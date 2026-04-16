"""
Marco_trader_4.0.py  –  IMC Prosperity Round 1
================================================
Key improvements over v3.5
──────────────────────────
ASH_COATED_OSMIUM
  • Ornstein-Uhlenbeck (OU) process replaces ad-hoc autocorrelation blend.
    Calibrated from 3-day historical data: θ=0.2427, μ=10000.20, σ_ou=5.014.
  • z-score = (mid − μ) / σ_ou drives BOTH the cross threshold (how
    aggressively we take mispriced liquidity) AND the quote-centre skew.
  • Combined quote skew = inventory skew + OU signal skew (both lean in the
    profitable direction; opposing signals partially cancel).
  • Dynamic half-spread tied to realised local volatility.
  • End-of-day unwind starts a little earlier and flattens harder.

INTARIAN_PEPPER_ROOT
  • Kalman filter (1-D random-walk state) replaces the simple EMA intercept.
    Tracks the detrended price = mid − slope×t optimally, with Q=0.10,
    R=5.0 (calibrated: noise_std≈2.2, so R≈2.2²≈5).  Steady-state
    Kalman gain ≈ 0.12 – tighter than the old α=0.15 warm-up but more
    responsive than α=0.01 late phase.
  • Imbalance nudge kept but tightened.

Both products
  • Volume-weighted mid (micro-price) for sharper fair-value estimates.
  • Simplified state: removed regime strings; OU/KF provide quantitative
    signals instead.
"""

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

    # Pepper linear trend (exactly matches data across all days)
    PEPPER_SLOPE: float = 0.001000  # seashells per timestamp tick

    # Kalman filter hyperparameters for Pepper intercept tracking
    KF_Q: float = 0.10   # process noise variance (intercept random-walk step)
    KF_R: float = 5.00   # observation noise variance  (~2.2² from calibration)
    KF_P0: float = 500.0 # large initial uncertainty → fast early convergence

    # ──────────────────────── State ────────────────────────────

    def default_state(self) -> Dict:
        return {
            "ou_mu": self.OU_MU,
            "kf_x": None,        # Kalman intercept estimate
            "kf_P": self.KF_P0,  # Kalman error variance
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
        """
        Volume-weighted mid (micro-price).
        Uses best bid/ask weighted by the opposing side's volume so that a
        thick ask pulls the fair price toward the ask, and vice-versa.
        Falls back to simple mid when one side is missing.
        """
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is None and ba is None:
            return None
        if bb is None:
            return float(ba)
        if ba is None:
            return float(bb)
        bid_vol = od.buy_orders[bb]        # positive
        ask_vol = -od.sell_orders[ba]      # positive (stored negative)
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.5 * (bb + ba)
        return (bb * ask_vol + ba * bid_vol) / total  # microprice

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
        """RMS of recent mid differences as a proxy for instantaneous vol."""
        n = min(len(hist), window)
        if n < 2:
            return self.OU_SIGMA
        diffs = [hist[-i] - hist[-i - 1] for i in range(1, n)]
        return math.sqrt(sum(d * d for d in diffs) / len(diffs))

    # ──────────────────────── OU model ─────────────────────────

    def ou_z_score(self, mid: float, mu: float) -> float:
        return (mid - mu) / self.OU_STD

    def ou_update_mu(self, mid: float, mem: Dict) -> float:
        """
        Slowly adapt the long-run mean using a tiny EMA.
        Keeps μ anchored to calibrated value but can track very slow drifts.
        """
        mem["ou_mu"] = (1 - self.OU_MU_ALPHA) * mem["ou_mu"] + self.OU_MU_ALPHA * mid
        return mem["ou_mu"]

    # ──────────────────────── Kalman filter ────────────────────

    def kf_update(self, ts: int, mid: float, mem: Dict) -> float:
        """
        1-D Kalman filter for the Pepper trend intercept.

        Model
        ─────
          State  x_k  = intercept  (random walk with noise Q)
          Obs    z_k  = mid − slope × t  ≈  intercept + v,  v ~ N(0, R)

        Returns the Kalman-filtered fair price at the current timestamp.
        """
        obs = mid - self.PEPPER_SLOPE * ts   # detrend → should be ≈ intercept

        x = mem["kf_x"]
        P = mem["kf_P"]

        if x is None:
            # Bootstrap: trust first observation with observation-noise variance
            mem["kf_x"] = obs
            mem["kf_P"] = self.KF_R
            return mid

        # Predict (intercept drifts as a random walk)
        x_pred = x
        P_pred = P + self.KF_Q

        # Update (incorporate new price observation)
        K = P_pred / (P_pred + self.KF_R)       # Kalman gain
        x_new = x_pred + K * (obs - x_pred)     # posterior mean
        P_new = (1.0 - K) * P_pred              # posterior variance

        mem["kf_x"] = x_new
        mem["kf_P"] = P_new

        return x_new + self.PEPPER_SLOPE * ts    # re-add trend → fair price

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
        """Buy all resting ask orders at prices ≤ max_px, up to cap units."""
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
        """Sell into all resting bid orders at prices ≥ min_px, up to cap units."""
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
        """
        Mean-reversion market-making driven by the OU model.

        Quote centre:
            skewed_fair = μ
                        − inv_skew   (lean against inventory)
                        − z_skew     (lean in direction of OU signal)

        Cross threshold:
            Dynamically tightened when |z| is large so we lift cheap offers /
            hit expensive bids more aggressively when the OU signal is strongest.
        """
        product = "ASH_COATED_OSMIUM"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        mid = self.micro_price(od)
        if mid is None:
            return orders

        ts = state.timestamp

        # ── OU fair value & signal ──────────────────────────────────────────
        mu = self.ou_update_mu(mid, mem)
        z = self.ou_z_score(mid, mu)      # signed OU deviation in units of σ_ou
        abs_z = abs(z)

        # Imbalance: small nudge toward heavier side (short-term pressure)
        imbalance = self.order_book_imbalance(od, 3)
        imb_nudge = 0.6 * imbalance       # ≤ 0.6 ticks

        # Quote centre with combined skew
        inv = pos / limit                 # ∈ [−1, 1]
        inv_skew = 3.5 * inv             # lean away from inventory
        z_skew = 1.2 * max(-2.0, min(2.0, z))  # lean in direction of OU signal
        skewed_fair = mu - inv_skew - z_skew + imb_nudge

        # ── End-of-day unwind ──────────────────────────────────────────────
        if ts > 940_000:
            if pos > 0 and bb is not None:
                orders.append(Order(product, bb, -pos))
            elif pos < 0 and ba is not None:
                orders.append(Order(product, ba, -pos))
            return orders

        # ── Dynamic cross threshold (driven by OU z-score) ────────────────
        # The larger |z|, the more mispriced the market is → be more aggressive
        if abs_z >= 2.0:
            cross_thresh = 1
        elif abs_z >= 1.5:
            cross_thresh = 2
        elif abs_z >= 0.8:
            cross_thresh = 3
        else:
            cross_thresh = 4   # near equilibrium: conservative, let MM do the work

        # ── Aggressive crossing: take mispriced resting liquidity ──────────
        # Buy when ask is cheap (below our skewed fair minus threshold)
        orders.extend(
            self.take_asks(product, od, skewed_fair - cross_thresh, pos, limit)
        )
        pos_a = pos + sum(o.quantity for o in orders)

        # Sell when bid is expensive (above our skewed fair plus threshold)
        orders.extend(
            self.take_bids(product, od, skewed_fair + cross_thresh, pos_a, limit)
        )
        pos_a = pos + sum(o.quantity for o in orders)

        # ── Passive quoting ────────────────────────────────────────────────
        # Dynamic half-spread: wider when local vol is elevated
        lv = self.local_vol(mem["mid_hist"][product])
        half_spread = max(2, min(5, round(lv * 0.75)))

        # Size: larger when OU signal & inventory both support the direction,
        #       smaller when near the limit or when z is near 0.
        inv_a = pos_a / limit
        base_size = max(5, 16 - int(8 * abs(inv_a)))

        # Boost size when OU strongly supports the trade direction
        if (z < -1.0 and pos_a < limit * 0.6) or (z > 1.0 and pos_a > -limit * 0.6):
            base_size = min(base_size + 4, 22)

        bid_px = math.floor(skewed_fair - half_spread)
        ask_px = math.ceil(skewed_fair + half_spread)

        # Don't place orders behind the inside market (queue waste)
        if bb is not None:
            bid_px = min(bid_px, bb + 1)
        if ba is not None:
            ask_px = max(ask_px, ba - 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_cap = max(0, limit - pos_a)
        sell_cap = max(0, limit + pos_a)

        if buy_cap > 0:
            orders.append(Order(product, bid_px, min(base_size, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -min(base_size, sell_cap)))

        return orders

    # ──────────────────────── Pepper logic ─────────────────────

    def target_pos_pepper(self, ts: int) -> int:
        """
        Step-down target position for Pepper.
        We hold max long throughout the day (trend-riding) and unwind in stages
        so we're flat before the final mark-to-market.
        """
        if ts < 870_000:
            return 50
        if ts < 920_000:
            return 30
        if ts < 960_000:
            return 10
        return 0

    def trade_pepper(
        self, state: TradingState, od: OrderDepth, pos: int, mem: Dict
    ) -> List[Order]:
        """
        Trend-riding with Kalman-filtered fair value.

        Strategy
        ────────
        • Maintain max long position (50) throughout the uptrend.
        • Fair value from Kalman filter tracking the linear intercept.
        • Aggressive buying early (large threshold above fair) to build the book.
        • Step-down target near end-of-day for clean unwind.
        • Passive ask placed well above fair so we don't accidentally sell early.
        """
        product = "INTARIAN_PEPPER_ROOT"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        mid = self.micro_price(od)
        if mid is None:
            return orders

        ts = state.timestamp

        # ── Kalman-filtered fair value ─────────────────────────────────────
        fair = self.kf_update(ts, mid, mem)

        # Small imbalance nudge (keep it tiny – trend already supplies the edge)
        imbalance = self.order_book_imbalance(od, 3)
        fair += 0.25 * max(0.0, imbalance)   # only positive imbalance adds to buy fair

        target = self.target_pos_pepper(ts)
        pos_a = pos

        # ── Phase 1: Build long position up to target ──────────────────────
        need = max(0, target - pos_a)
        if need > 0:
            # Accept increasing price premiums early in the day because the
            # trend advantage far exceeds the crossing cost.
            if ts < 10_000:
                max_buy = fair + 12
            elif ts < 50_000:
                max_buy = fair + 6
            elif ts < 200_000:
                max_buy = fair + 3
            elif ts < 800_000:
                max_buy = fair + 2
            else:
                max_buy = fair + 1

            new = self.take_asks(product, od, max_buy, pos_a, limit, cap=need)
            orders.extend(new)
            pos_a += sum(o.quantity for o in new)

        # ── Phase 2: Trim excess above target ─────────────────────────────
        extra = max(0, pos_a - target)
        if extra > 0:
            # Never sell cheap mid-day; near end-of-day allow wider margin
            min_sell = fair - 1 if ts > 900_000 else fair + 8
            new = self.take_bids(product, od, min_sell, pos_a, limit, cap=extra)
            orders.extend(new)
            pos_a += sum(o.quantity for o in new)

        # ── Phase 3: Hard unwind at day end ───────────────────────────────
        if ts > 975_000:
            if pos_a > 0 and bb is not None:
                orders.append(Order(product, bb, -pos_a))
            elif pos_a < 0 and ba is not None:
                orders.append(Order(product, ba, -pos_a))
            return orders

        # ── Phase 4: Passive quoting ───────────────────────────────────────
        buy_cap = max(0, limit - pos_a)
        sell_cap = max(0, limit + pos_a)

        # Bid: just above Kalman fair to stay at top of queue
        bid_px = math.floor(fair + 0.5)
        if bb is not None:
            bid_px = min(bid_px, bb + 1)

        # Ask: well above fair unless we're reducing or near EoD
        if pos_a > target or ts > 880_000:
            ask_px = math.ceil(fair + 2)
        else:
            ask_px = math.ceil(fair + 22)   # won't fill → effectively hold position
        if ba is not None:
            ask_px = max(ask_px, ba - 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        if pos_a < target and buy_cap > 0:
            size = min(buy_cap, max(10, target - pos_a))
            orders.append(Order(product, bid_px, size))

        if (pos_a > target or ts > 900_000) and sell_cap > 0:
            size = min(sell_cap, max(5, pos_a - target))
            if size > 0:
                orders.append(Order(product, ask_px, -size))

        return orders

    # ──────────────────────── Run ───────────────────────────────

    def run(self, state: TradingState):
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

        return result, 0, self.dump_state(mem)
