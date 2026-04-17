"""
Marco_trader_24_hybrid.py  –  IMC Prosperity Round 1
====================================================
Hybrid version of trader 24.

What changed
────────────
INTARIAN_PEPPER_ROOT now has a failsafe:
  • It still uses the original linear drift / hold-long logic while the trend
    looks valid.
  • If the Pepper residual becomes erratic for several consecutive ticks, it
    latches into a fallback mode early.
  • Even if that never happens, it force-switches at a fixed late-session
    timestamp to an Osmium-style mean-reversion market-maker for Pepper.

This gives Pepper two behaviours:
  1) early-session trend capture,
  2) late-session / erratic-session adaptive mean reversion.
"""

try:
    from prosperity3bt.datamodel import OrderDepth, TradingState, Order
except ImportError:
    from datamodel import OrderDepth, TradingState, Order

from typing import Dict, List, Tuple, Optional
import json
import math


class Trader:
    # ──────────────────────── Constants ────────────────────────
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 50,
        "INTARIAN_PEPPER_ROOT": 50,
    }

    # OU parameters – calibrated via OLS on all 3 historical days
    OU_MU: float = 10000.20   # long-run mean
    OU_THETA: float = 0.2427  # mean-reversion speed (per tick)
    OU_SIGMA: float = 3.494   # per-tick noise std
    OU_STD: float = 5.014     # stationary std = σ / √(2θ)
    OU_MU_ALPHA: float = 5e-4 # very slow μ adaptation (handles regime drift)

    # Pepper linear trend model
    PEPPER_SLOPE: float = 0.001000
    KF_Q: float = 0.10
    KF_R: float = 5.00
    KF_P0: float = 500.0

    # Pepper fallback controls
    PEPPER_FORCE_FALLBACK_TS: int = 910_000
    PEPPER_HARD_FLAT_TS: int = 995_000
    PEPPER_ERRATIC_Z: float = 3.5
    PEPPER_ERRATIC_CONFIRM_TICKS: int = 3
    PEPPER_FALLBACK_MR_WINDOW: int = 18

    # ──────────────────────── State ────────────────────────────

    def default_state(self) -> Dict:
        return {
            "ou_mu": self.OU_MU,
            "kf_x": None,
            "kf_P": self.KF_P0,
            "pepper_fallback": False,
            "pepper_erratic_count": 0,
            "mid_hist": {p: [] for p in self.POSITION_LIMITS},
        }

    def load_state(self, trader_data: str) -> Dict:
        base = self.default_state()
        if not trader_data:
            return base
        try:
            s = json.loads(trader_data)
            for k, v in base.items():
                if k not in s:
                    s[k] = v
            for p in self.POSITION_LIMITS:
                if p not in s["mid_hist"]:
                    s["mid_hist"][p] = []
            return s
        except Exception:
            return base

    def dump_state(self, s: Dict) -> str:
        return json.dumps(s, separators=(",", ":"))

    # ──────────────────────── Market helpers ───────────────────

    def best_bid_ask(self, od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        return bb, ba

    def micro_price(self, od: OrderDepth) -> Optional[float]:
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None and ba is None:
            return None
        if bb is None:
            return float(ba)
        if ba is None:
            return float(bb)
        bid_vol = od.buy_orders[bb]
        ask_vol = -od.sell_orders[ba]
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.5 * (bb + ba)
        return (bb * ask_vol + ba * bid_vol) / total

    def order_book_imbalance(self, od: OrderDepth, levels: int = 3) -> float:
        bids = sorted(od.buy_orders.items(), reverse=True)[:levels]
        asks = sorted(od.sell_orders.items())[:levels]
        bid_vol = sum(max(0, qty) for _, qty in bids)
        ask_vol = sum(max(0, -qty) for _, qty in asks)
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def push_hist(self, hist: List[float], val: float, maxlen: int = 30) -> None:
        hist.append(val)
        if len(hist) > maxlen:
            del hist[0]

    def local_vol(self, hist: List[float], window: int = 12) -> float:
        n = min(len(hist), window)
        if n < 2:
            return self.OU_SIGMA
        diffs = [hist[-i] - hist[-i - 1] for i in range(1, n)]
        return math.sqrt(sum(d * d for d in diffs) / len(diffs))

    def rolling_mean(self, hist: List[float], window: int) -> Optional[float]:
        n = min(len(hist), window)
        if n <= 0:
            return None
        return sum(hist[-n:]) / n

    def clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    # ──────────────────────── OU helpers ───────────────────────

    def ou_dynamic_std(self, hist: List[float]) -> float:
        sigma_hat = self.local_vol(hist)
        std_hat = sigma_hat / math.sqrt(max(1e-6, 2.0 * self.OU_THETA))
        blended = 0.75 * self.OU_STD + 0.25 * std_hat
        return max(3.0, min(12.0, blended))

    def ou_z_score(self, mid: float, mu: float, std: float) -> float:
        std = max(1e-6, std)
        return (mid - mu) / std

    def ou_update_mu(self, mid: float, mem: Dict) -> float:
        mem["ou_mu"] = (1 - self.OU_MU_ALPHA) * mem["ou_mu"] + self.OU_MU_ALPHA * mid
        return mem["ou_mu"]

    # ──────────────────────── Kalman filter ────────────────────

    def kf_update(self, ts: int, mid: float, mem: Dict) -> float:
        obs = mid - self.PEPPER_SLOPE * ts

        x = mem["kf_x"]
        P = mem["kf_P"]

        if x is None:
            mem["kf_x"] = obs
            mem["kf_P"] = self.KF_R
            return mid

        x_pred = x
        P_pred = P + self.KF_Q

        K = P_pred / (P_pred + self.KF_R)
        x_new = x_pred + K * (obs - x_pred)
        P_new = (1.0 - K) * P_pred

        mem["kf_x"] = x_new
        mem["kf_P"] = P_new

        return x_new + self.PEPPER_SLOPE * ts

    # ──────────────────────── Order helpers ────────────────────

    def take_asks(
        self,
        product: str,
        od: OrderDepth,
        max_px: float,
        pos: int,
        limit: int,
        cap: Optional[int] = None,
    ) -> List[Order]:
        orders: List[Order] = []
        room = limit - pos
        if cap is not None:
            room = min(room, cap)
        for px in sorted(od.sell_orders):
            if room <= 0 or px > max_px:
                break
            qty = min(room, max(0, -od.sell_orders[px]))
            if qty > 0:
                orders.append(Order(product, px, qty))
                room -= qty
        return orders

    def take_bids(
        self,
        product: str,
        od: OrderDepth,
        min_px: float,
        pos: int,
        limit: int,
        cap: Optional[int] = None,
    ) -> List[Order]:
        orders: List[Order] = []
        room = limit + pos
        if cap is not None:
            room = min(room, cap)
        for px in sorted(od.buy_orders, reverse=True):
            if room <= 0 or px < min_px:
                break
            qty = min(room, max(0, od.buy_orders[px]))
            if qty > 0:
                orders.append(Order(product, px, -qty))
                room -= qty
        return orders

    # ──────────────────────── Osmium logic ─────────────────────

    def trade_osmium(
        self, state: TradingState, od: OrderDepth, pos: int, mem: Dict
    ) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        mid = self.micro_price(od)
        if mid is None:
            return orders

        ts = state.timestamp
        mu = self.ou_update_mu(mid, mem)
        dyn_std = self.ou_dynamic_std(mem["mid_hist"][product])
        z = self.ou_z_score(mid, mu, dyn_std)
        abs_z = abs(z)

        imbalance = self.order_book_imbalance(od, 3)
        imb = self.clamp(imbalance, -1.0, 1.0)

        mh = mem["mid_hist"][product]
        mom = 0.0
        if len(mh) >= 6:
            mom = mh[-1] - mh[-6]
        toxic = (abs(imb) > 0.55) and (mom * imb > 0.0)

        inv = pos / limit
        inv_k = 3.0 + (2.0 if dyn_std < 6.0 else 0.0)
        inv_skew = inv_k * inv
        z_skew = 1.4 * self.clamp(z, -2.5, 2.5)
        imb_nudge = 0.75 * imb
        fair = mu - inv_skew - z_skew + imb_nudge

        if ts > 930_000:
            if pos > 0 and bb is not None:
                orders.append(Order(product, bb, -pos))
            elif pos < 0 and ba is not None:
                orders.append(Order(product, ba, -pos))
            return orders

        spread = None
        if bb is not None and ba is not None:
            spread = max(0, ba - bb)

        if abs_z >= 2.2:
            cross_thresh = 1
        elif abs_z >= 1.4:
            cross_thresh = 2
        elif abs_z >= 0.9:
            cross_thresh = 3
        else:
            cross_thresh = 4

        if toxic and abs_z < 2.6:
            cross_thresh = max(4, cross_thresh)

        if spread is not None and spread >= 6 and not toxic:
            cross_thresh = max(1, cross_thresh - 1)

        orders.extend(self.take_asks(product, od, fair - cross_thresh, pos, limit))
        pos_a = pos + sum(o.quantity for o in orders)

        orders.extend(self.take_bids(product, od, fair + cross_thresh, pos_a, limit))
        pos_a = pos + sum(o.quantity for o in orders)

        half_spread = max(1, min(5, int(round(dyn_std * 0.35))))
        if toxic:
            half_spread = min(6, half_spread + 1)

        inv_a = pos_a / limit
        base_size = max(6, 18 - int(10 * abs(inv_a)))

        sig_boost = 0
        if abs_z >= 1.0:
            sig_boost = 2
        if abs_z >= 1.8:
            sig_boost = 4
        if toxic:
            sig_boost = 0
            base_size = max(4, int(base_size * 0.7))

        if (z < 0 and inv_a > 0.55) or (z > 0 and inv_a < -0.55):
            sig_boost = 0
        base_size = min(24, base_size + sig_boost)

        bid_px = math.floor(fair - half_spread)
        ask_px = math.ceil(fair + half_spread)

        if z <= -1.2 and not toxic:
            bid_px += 1
        elif z >= 1.2 and not toxic:
            ask_px -= 1

        if bb is not None:
            bid_px = min(bid_px, bb + 1)
        if ba is not None:
            ask_px = max(ask_px, ba - 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_cap = max(0, limit - pos_a)
        sell_cap = max(0, limit + pos_a)

        if (not toxic) and spread is not None and spread >= 2:
            inside_sz = 6
            if buy_cap > 0 and bb is not None:
                orders.append(Order(product, bb + 1, min(inside_sz, buy_cap)))
            if sell_cap > 0 and ba is not None:
                orders.append(Order(product, ba - 1, -min(inside_sz, sell_cap)))

        if buy_cap > 0:
            orders.append(Order(product, bid_px, min(base_size, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -min(base_size, sell_cap)))

        return orders

    # ──────────────────────── Pepper mode switch ───────────────

    def pepper_should_use_fallback(
        self,
        state: TradingState,
        od: OrderDepth,
        mem: Dict,
    ) -> bool:
        if mem["pepper_fallback"]:
            return True

        ts = state.timestamp
        if ts >= self.PEPPER_FORCE_FALLBACK_TS:
            mem["pepper_fallback"] = True
            return True

        mid = self.micro_price(od)
        if mid is None:
            return False

        fair = self.kf_update(ts, mid, mem)
        hist = mem["mid_hist"]["INTARIAN_PEPPER_ROOT"]
        resid_std = max(math.sqrt(self.KF_R), 0.9 * self.local_vol(hist, 12))
        resid_z = abs(mid - fair) / max(1e-6, resid_std)

        if ts > 100_000 and resid_z >= self.PEPPER_ERRATIC_Z:
            mem["pepper_erratic_count"] += 1
        else:
            mem["pepper_erratic_count"] = max(0, mem["pepper_erratic_count"] - 1)

        if mem["pepper_erratic_count"] >= self.PEPPER_ERRATIC_CONFIRM_TICKS:
            mem["pepper_fallback"] = True

        return mem["pepper_fallback"]

    # ──────────────────────── Pepper linear logic ──────────────

    def trade_pepper_linear(
        self, state: TradingState, od: OrderDepth, pos: int
    ) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        if bb is None and ba is None:
            return orders

        ts = state.timestamp

        if ts < 910_000:
            target = limit
        elif ts < 960_000:
            target = 35
        elif ts < 985_000:
            target = 15
        else:
            target = 0

        if pos < target and ba is not None:
            need = min(target - pos, limit - pos)
            cushion = 2 if ts < 25_000 else (1 if ts < 60_000 else 0)
            max_buy_px = ba + cushion
            new = self.take_asks(product, od, max_buy_px, pos, limit, cap=need)
            orders.extend(new)
            pos += sum(o.quantity for o in new)

        if pos > target and bb is not None:
            need = min(pos - target, pos + limit)
            cushion = 2 if ts > 970_000 else (1 if ts > 930_000 else 0)
            min_sell_px = bb - cushion
            new = self.take_bids(product, od, min_sell_px, pos, limit, cap=need)
            orders.extend(new)
            pos += sum(o.quantity for o in new)

        if ts > self.PEPPER_HARD_FLAT_TS:
            bb2, ba2 = self.best_bid_ask(od)
            if pos > 0 and bb2 is not None:
                orders.append(Order(product, bb2, -pos))
            elif pos < 0 and ba2 is not None:
                orders.append(Order(product, ba2, -pos))

        return orders

    # ──────────────────────── Pepper fallback logic ────────────

    def trade_pepper_fallback(
        self, state: TradingState, od: OrderDepth, pos: int, mem: Dict
    ) -> List[Order]:
        """
        Osmium-style fallback for Pepper.

        Uses a rolling mean + local-vol z-score instead of the fixed linear trend.
        This is meant to take over once the Pepper trend becomes unreliable or once
        the linear strategy's main hold phase has ended.
        """
        product = "INTARIAN_PEPPER_ROOT"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        mid = self.micro_price(od)
        if mid is None:
            return orders

        ts = state.timestamp
        hist = mem["mid_hist"][product]

        mu = self.rolling_mean(hist, self.PEPPER_FALLBACK_MR_WINDOW)
        if mu is None:
            mu = mid

        dyn_std = max(2.0, min(12.0, 1.35 * self.local_vol(hist, 12)))
        z = (mid - mu) / max(1e-6, dyn_std)
        abs_z = abs(z)

        imbalance = self.order_book_imbalance(od, 3)
        imb = self.clamp(imbalance, -1.0, 1.0)

        mom = 0.0
        if len(hist) >= 6:
            mom = hist[-1] - hist[-6]
        toxic = (abs(imb) > 0.55) and (mom * imb > 0.0)

        inv = pos / limit
        inv_skew = 2.8 * inv
        z_skew = 1.2 * self.clamp(z, -2.5, 2.5)
        fair = mu - inv_skew - z_skew + 0.60 * imb

        if ts > self.PEPPER_HARD_FLAT_TS:
            if pos > 0 and bb is not None:
                orders.append(Order(product, bb, -pos))
            elif pos < 0 and ba is not None:
                orders.append(Order(product, ba, -pos))
            return orders

        spread = None
        if bb is not None and ba is not None:
            spread = max(0, ba - bb)

        if abs_z >= 2.4:
            cross_thresh = 1
        elif abs_z >= 1.5:
            cross_thresh = 2
        elif abs_z >= 0.9:
            cross_thresh = 3
        else:
            cross_thresh = 4

        if toxic and abs_z < 2.6:
            cross_thresh = max(4, cross_thresh)

        orders.extend(self.take_asks(product, od, fair - cross_thresh, pos, limit))
        pos_a = pos + sum(o.quantity for o in orders)

        orders.extend(self.take_bids(product, od, fair + cross_thresh, pos_a, limit))
        pos_a = pos + sum(o.quantity for o in orders)

        half_spread = max(1, min(6, int(round(dyn_std * 0.40))))
        if toxic:
            half_spread = min(7, half_spread + 1)

        inv_a = pos_a / limit
        base_size = max(6, 16 - int(9 * abs(inv_a)))

        sig_boost = 0
        if abs_z >= 1.0:
            sig_boost = 2
        if abs_z >= 1.8:
            sig_boost = 4
        if toxic:
            sig_boost = 0
            base_size = max(4, int(base_size * 0.7))

        if (z < 0 and inv_a > 0.55) or (z > 0 and inv_a < -0.55):
            sig_boost = 0
        base_size = min(22, base_size + sig_boost)

        bid_px = math.floor(fair - half_spread)
        ask_px = math.ceil(fair + half_spread)

        if z <= -1.2 and not toxic:
            bid_px += 1
        elif z >= 1.2 and not toxic:
            ask_px -= 1

        if bb is not None:
            bid_px = min(bid_px, bb + 1)
        if ba is not None:
            ask_px = max(ask_px, ba - 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_cap = max(0, limit - pos_a)
        sell_cap = max(0, limit + pos_a)

        if (not toxic) and spread is not None and spread >= 2:
            inside_sz = 5
            if buy_cap > 0 and bb is not None:
                orders.append(Order(product, bb + 1, min(inside_sz, buy_cap)))
            if sell_cap > 0 and ba is not None:
                orders.append(Order(product, ba - 1, -min(inside_sz, sell_cap)))

        if buy_cap > 0:
            orders.append(Order(product, bid_px, min(base_size, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -min(base_size, sell_cap)))

        return orders

    def trade_pepper(
        self, state: TradingState, od: OrderDepth, pos: int, mem: Dict
    ) -> List[Order]:
        if self.pepper_should_use_fallback(state, od, mem):
            return self.trade_pepper_fallback(state, od, pos, mem)
        return self.trade_pepper_linear(state, od, pos)

    # ──────────────────────── Main run ─────────────────────────

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        mem = self.load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                result[product] = []
                continue

            mid = self.micro_price(od)
            if mid is not None:
                self.push_hist(mem["mid_hist"][product], mid)

            pos = state.position.get(product, 0)

            if product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_osmium(state, od, pos, mem)
            else:
                result[product] = self.trade_pepper(state, od, pos, mem)

        conversions = 0
        traderData = self.dump_state(mem)
        return result, conversions, traderData
