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


    def ou_dynamic_std(self, hist: List[float]) -> float:
        """Estimate the current stationary OU std from local per-tick vol.
        We blend the calibrated stationary std with a rolling estimate so the
        z-score stays meaningful across volatility regimes.
        """
        sigma_hat = self.local_vol(hist)  # per-tick RMS
        std_hat = sigma_hat / math.sqrt(max(1e-6, 2.0 * self.OU_THETA))
        # Blend and clamp to keep behaviour stable
        blended = 0.75 * self.OU_STD + 0.25 * std_hat
        return max(3.0, min(12.0, blended))
    # ──────────────────────── OU model ─────────────────────────

    def ou_z_score(self, mid: float, mu: float, std: float) -> float:
        # Guard against divide-by-zero and overly tiny std from short windows
        std = max(1e-6, std)
        return (mid - mu) / std
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
        OSMIUM: Mean-reversion market-making with OU signal + microstructure safety.

        Goal: keep the strong baseline behaviour of Paveet_trader_F, but avoid the
        PnL drop from overly-tight quotes by:
          • detecting short-horizon "toxic" conditions (imbalance + momentum),
          • widening / shrinking size under toxicity,
          • adding a small inside-quote tier ONLY when conditions are benign.
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
        dyn_std = self.ou_dynamic_std(mem["mid_hist"][product])
        z = self.ou_z_score(mid, mu, dyn_std)
        abs_z = abs(z)

        # Microstructure features
        imbalance = self.order_book_imbalance(od, 3)  # [-1,1]
        imb = max(-1.0, min(1.0, imbalance))

        # Short-horizon momentum (very cheap)
        mh = mem["mid_hist"][product]
        mom = 0.0
        if len(mh) >= 6:
            mom = mh[-1] - mh[-6]  # ~5-step momentum
        # toxicity proxy: strong one-sided book + price moving same way
        toxic = (abs(imb) > 0.55) and (mom * imb > 0.0)

        # Quote centre with combined skew
        inv = pos / limit  # [-1,1]
        inv_k = 3.0 + (2.0 if dyn_std < 6.0 else 0.0)
        inv_skew = inv_k * inv
        z_skew = 1.4 * max(-2.5, min(2.5, z))
        imb_nudge = 0.75 * imb  # <= 0.75 ticks
        fair = mu - inv_skew - z_skew + imb_nudge

        # ── End-of-day unwind (keep baseline timing; do not "panic" early) ──
        if ts > 930_000:
            if pos > 0 and bb is not None:
                orders.append(Order(product, bb, -pos))
            elif pos < 0 and ba is not None:
                orders.append(Order(product, ba, -pos))
            return orders

        # ── Crossing policy ────────────────────────────────────────────────
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

        # Under toxicity: be less eager to cross (avoid getting picked off)
        if toxic and abs_z < 2.6:
            cross_thresh = max(4, cross_thresh)

        if spread is not None and spread >= 6 and not toxic:
            cross_thresh = max(1, cross_thresh - 1)

        # Aggressive crossing
        orders.extend(self.take_asks(product, od, fair - cross_thresh, pos, limit))
        pos_a = pos + sum(o.quantity for o in orders)

        orders.extend(self.take_bids(product, od, fair + cross_thresh, pos_a, limit))
        pos_a = pos + sum(o.quantity for o in orders)

        # ── Passive quoting ────────────────────────────────────────────────
        # Baseline width, widened slightly when toxic.
        half_spread = max(1, min(5, int(round(dyn_std * 0.35))))
        if toxic:
            half_spread = min(6, half_spread + 1)

        inv_a = pos_a / limit
        base_size = max(6, 18 - int(10 * abs(inv_a)))

        # Size boosts when signal strong, but not when toxicity suggests adverse selection.
        sig_boost = 0
        if abs_z >= 1.0:
            sig_boost = 2
        if abs_z >= 1.8:
            sig_boost = 4
        if toxic:
            sig_boost = 0
            base_size = max(4, int(base_size * 0.7))

        # Don't add risk if signal fights inventory
        if (z < 0 and inv_a > 0.55) or (z > 0 and inv_a < -0.55):
            sig_boost = 0
        base_size = min(24, base_size + sig_boost)

        bid_px = math.floor(fair - half_spread)
        ask_px = math.ceil(fair + half_spread)

        # small price shading when conviction is high (keep baseline)
        if z <= -1.2 and not toxic:
            bid_px += 1
        elif z >= 1.2 and not toxic:
            ask_px -= 1

        # Clamp to near top-of-book
        if bb is not None:
            bid_px = min(bid_px, bb + 1)
        if ba is not None:
            ask_px = max(ask_px, ba - 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_cap = max(0, limit - pos_a)
        sell_cap = max(0, limit + pos_a)

        # Tier-1: small inside quote when benign and spread allows.
        if (not toxic) and spread is not None and spread >= 2:
            inside_sz = 6
            if buy_cap > 0 and bb is not None:
                orders.append(Order(product, bb + 1, min(inside_sz, buy_cap)))
            if sell_cap > 0 and ba is not None:
                orders.append(Order(product, ba - 1, -min(inside_sz, sell_cap)))

        # Tier-2: main quotes around fair
        if buy_cap > 0:
            orders.append(Order(product, bid_px, min(base_size, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -min(base_size, sell_cap)))

        return orders
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
        """PEPPER drift-capture (error-safe).

        Policy:
          • get to +limit quickly early (small sweep of asks),
          • hold +limit most of the session (no profit-taking),
          • unwind in stages late,
          • hard flatten very late.
        """
        product = "INTARIAN_PEPPER_ROOT"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        if bb is None and ba is None:
            return orders

        ts = state.timestamp

        # Stage targets (late unwind). Uses full long most of the day.
        if ts < 910_000:
            target = limit
        elif ts < 960_000:
            target = 35
        elif ts < 985_000:
            target = 15
        else:
            target = 0

        # ---- Build to target (early aggressive) ----
        if pos < target and ba is not None:
            need = min(target - pos, limit - pos)
            # Early: allow sweeping slightly above best ask to guarantee fills.
            # Later: only take best ask.
            cushion = 2 if ts < 25_000 else (1 if ts < 60_000 else 0)
            max_buy_px = ba + cushion

            # Use existing helper (sweeps asks up to max price)
            new = self.take_asks(product, od, max_buy_px, pos, limit, cap=need)
            orders.extend(new)
            pos += sum(o.quantity for o in new)

        # ---- Reduce to target (late staged unwind) ----
        if pos > target and bb is not None:
            need = min(pos - target, pos + limit)
            # Closer to close, accept slightly worse to ensure we get out.
            cushion = 2 if ts > 970_000 else (1 if ts > 930_000 else 0)
            min_sell_px = bb - cushion

            new = self.take_bids(product, od, min_sell_px, pos, limit, cap=need)
            orders.extend(new)
            pos += sum(o.quantity for o in new)

        # ---- Hard flatten at the very end ----
        if ts > 995_000:
            bb2, ba2 = self.best_bid_ask(od)
            if pos > 0 and bb2 is not None:
                orders.append(Order(product, bb2, -pos))
            elif pos < 0 and ba2 is not None:
                orders.append(Order(product, ba2, -pos))

        return orders
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
