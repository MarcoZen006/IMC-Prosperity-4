from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json
import math

# ─────────────────────────────────────────────────────────────────────────────
#  KEY DATA INSIGHTS (from prices_round_1_day_-2/-1/0.csv analysis)
#
#  ASH_COATED_OSMIUM
#   • Stationary around 10000 — zero long-run drift across all days
#   • Return autocorrelation (lag-1) = EXACTLY –0.50 every day
#     → Using this signal cuts return-prediction MSE by ~25%
#   • Avg absolute tick return: 2.2  |  Best-bid-ask spread: ~16
#   • Market-maker half-spread of ±3 fits well inside the 8-tick half-spread
#     already in the book, so we quote inside and attract flow.
#
#  INTARIAN_PEPPER_ROOT
#   • Slope = EXACTLY 0.001000 per timestamp (±2 residual std — nearly perfect)
#   • Total drift per day ≈ 1 001.  Being long 50 all day = ~50 000 profit.
#   • Available ask volume ≥ 50 units by timestamp 300–500 (essentially t=0)
#     → We can reach max position almost instantly and ride the full drift.
# ─────────────────────────────────────────────────────────────────────────────


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 50,
        "INTARIAN_PEPPER_ROOT": 50,
    }

    OSMIUM_LONG_MEAN = 10000.0
    PEPPER_SLOPE = 0.001000          # Rock-solid across all 3 days

    # ──────────────────────────────────────────────────────────────────────────
    #  State management
    # ──────────────────────────────────────────────────────────────────────────

    def default_state(self) -> Dict:
        return {
            "last_mid": {},
            "pepper_intercept": None,
            "osmium_adj_fair": self.OSMIUM_LONG_MEAN,
        }

    def load_state(self, trader_data: str) -> Dict:
        if not trader_data:
            return self.default_state()
        try:
            s = json.loads(trader_data)
            for k, v in self.default_state().items():
                if k not in s:
                    s[k] = v
            return s
        except Exception:
            return self.default_state()

    def dump_state(self, s: Dict) -> str:
        return json.dumps(s, separators=(",", ":"))

    # ──────────────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def best_bid_ask(self, od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        return bb, ba

    def mid(self, od: OrderDepth) -> Optional[float]:
        bb, ba = self.best_bid_ask(od)
        if bb is not None and ba is not None:
            return 0.5 * (bb + ba)
        return float(bb) if bb is not None else (float(ba) if ba is not None else None)

    def take_asks(self, product, od, max_px, pos, limit, cap=None) -> List[Order]:
        orders, room = [], limit - pos
        if cap is not None:
            room = min(room, cap)
        for px in sorted(od.sell_orders):
            if room <= 0 or px > max_px:
                break
            qty = min(room, -od.sell_orders[px])
            if qty > 0:
                orders.append(Order(product, px, qty))
                room -= qty
        return orders

    def take_bids(self, product, od, min_px, pos, limit, cap=None) -> List[Order]:
        orders, room = [], limit + pos
        if cap is not None:
            room = min(room, cap)
        for px in sorted(od.buy_orders, reverse=True):
            if room <= 0 or px < min_px:
                break
            qty = min(room, od.buy_orders[px])
            if qty > 0:
                orders.append(Order(product, px, -qty))
                room -= qty
        return orders

    # ──────────────────────────────────────────────────────────────────────────
    #  ASH_COATED_OSMIUM  —  Mean-reversion market making
    # ──────────────────────────────────────────────────────────────────────────

    def trade_osmium(self, state: TradingState, od: OrderDepth,
                     pos: int, mem: Dict) -> List[Order]:
        """
        Strategy: tight market making, anchored to 10 000, with a -0.5
        autocorrelation adjustment on every tick.

        The -0.5 autocorr tells us:
            E[mid_{t+1}] = mid_t - 0.5 * (mid_t - mid_{t-1})
                         = 0.5 * (mid_t + mid_{t-1})

        We blend this short-term prediction (80 %) with the long-run anchor
        10 000 (20 %) to get our fair value, then smooth it with a fast EMA
        (α=0.40) so quotes don't swing wildly on outlier ticks.

        We quote a ±3-tick half-spread around inventory-skewed fair — well
        inside the 8-tick half-spread already in the book — to attract fills.
        """
        product = "ASH_COATED_OSMIUM"
        limit   = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        mid = self.mid(od)
        if mid is None:
            return orders

        last_mid = mem["last_mid"].get(product, mid)
        last_ret = mid - last_mid                  # tick-level return

        # ── Fair value ──────────────────────────────────────────────────────
        # Short-term: fade last return (exploits -0.5 autocorr)
        autocorr_fair = 0.5 * (mid + last_mid)    # = mid - 0.5*last_ret
        # Blend with long-run mean, then EMA-smooth for quote stability
        raw_fair  = 0.80 * autocorr_fair + 0.20 * self.OSMIUM_LONG_MEAN
        prev_fair = mem.get("osmium_adj_fair", raw_fair)
        fair      = 0.60 * raw_fair + 0.40 * prev_fair
        mem["osmium_adj_fair"] = fair

        # Inventory skew: push fair DOWN if long, UP if short
        # This widens the effective sell-side and tightens buy-side when long.
        inv  = pos / limit                         # ∈ [–1, +1]
        skew = 4.0 * inv
        skewed_fair = fair - skew

        # ── Take obvious mispricings ────────────────────────────────────────
        # Threshold = 3 ticks: only cross the spread when conviction is high.
        orders.extend(self.take_asks(product, od, skewed_fair - 3, pos, limit))
        pos_a = pos + sum(o.quantity for o in orders)
        orders.extend(self.take_bids(product, od, skewed_fair + 3, pos_a, limit))
        pos_a = pos + sum(o.quantity for o in orders)

        # ── Endgame flatten ─────────────────────────────────────────────────
        if state.timestamp > 930_000:
            if pos_a > 0 and bb is not None:
                orders.append(Order(product, bb, -min(limit + pos_a, pos_a)))
            elif pos_a < 0 and ba is not None:
                orders.append(Order(product, ba,  min(limit - pos_a, -pos_a)))
            return orders

        # ── Passive quotes ──────────────────────────────────────────────────
        inv_a = pos_a / limit
        # Quote size: larger when near neutral, smaller at extremes.
        size = max(5, 14 - int(8 * abs(inv_a)))

        # Tight ±3 quotes, leaned by inventory
        bid_px = math.floor(skewed_fair - 3)
        ask_px = math.ceil (skewed_fair + 3)

        # Never cross existing book (be at best or improve by 1)
        if bb is not None: bid_px = min(bid_px, bb + 1)
        if ba is not None: ask_px = max(ask_px, ba - 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_cap  = max(0, limit - pos_a)
        sell_cap = max(0, limit + pos_a)

        if buy_cap  > 0: orders.append(Order(product, bid_px,  min(size, buy_cap)))
        if sell_cap > 0: orders.append(Order(product, ask_px, -min(size, sell_cap)))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    #  INTARIAN_PEPPER_ROOT  —  Trend-riding with optimal acquisition & unwind
    # ──────────────────────────────────────────────────────────────────────────

    def pepper_fair(self, ts: int, mid: float, mem: Dict) -> float:
        """
        True fair = intercept + 0.001000 * timestamp.
        Intercept is estimated via a fast-decaying EMA (aggressive at start
        of day when we have few data points, slow later once locked in).
        """
        obs_intercept = mid - self.PEPPER_SLOPE * ts
        stored = mem.get("pepper_intercept")
        if stored is None:
            mem["pepper_intercept"] = obs_intercept
            return mid
        # Decay: α=0.15 early (< 20k ts), α=0.01 once settled
        alpha = 0.15 if ts < 20_000 else 0.01
        mem["pepper_intercept"] = (1 - alpha) * stored + alpha * obs_intercept
        return mem["pepper_intercept"] + self.PEPPER_SLOPE * ts

    def target_pos_pepper(self, ts: int) -> int:
        """
        Be max-long (+50) as early as possible and hold until the final unwind.

        Every timestamp we're NOT at +50 we forfeit:
            50 × 0.001 = 0.05 PnL/timestamp.

        So we tolerate paying up to fair+10 at the very start to get there fast.
        Unwind schedule is tuned so we don't dump too early (losing drift) or
        too late (getting stuck without bids to sell into).
        """
        if ts < 870_000: return 50
        if ts < 930_000: return 30
        if ts < 965_000: return 10
        return 0

    def trade_pepper(self, state: TradingState, od: OrderDepth,
                     pos: int, mem: Dict) -> List[Order]:
        """
        Three phases:
          1. ACQUIRE (ts ≈ 0–500): blast into the ask book at any price ≤ fair+10.
             By ts=500 we're already at +50.  We forgo ≤ 500 × 0.05 = 25 in
             over-pay, capturing the full 50 050 in drift.
          2. HOLD (ts 500–870k): passive bid near fair to maintain +50.
             Ask side wide so we don't accidentally sell the trend.
          3. UNWIND (ts 870k–1M): step down target and sell aggressively.
        """
        product = "INTARIAN_PEPPER_ROOT"
        limit   = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        mid = self.mid(od)
        if mid is None:
            return orders

        fair   = self.pepper_fair(state.timestamp, mid, mem)
        target = self.target_pos_pepper(state.timestamp)
        ts     = state.timestamp
        pos_a  = pos

        # ── PHASE 1/2: Buy to reach target ─────────────────────────────────
        need = max(0, target - pos_a)
        if need > 0:
            if   ts <  10_000: max_buy = fair + 10   # Pay anything at open
            elif ts <  50_000: max_buy = fair +  6
            elif ts < 200_000: max_buy = fair +  4
            elif ts < 800_000: max_buy = fair +  2
            else:              max_buy = fair +  1
            new = self.take_asks(product, od, max_buy, pos_a, limit, cap=need)
            orders.extend(new)
            pos_a += sum(o.quantity for o in new)

        # ── PHASE 3: Sell excess / late unwind ────────────────────────────
        extra = max(0, pos_a - target)
        if extra > 0:
            # Late in day: sell at anything; earlier: only sell if very expensive
            min_sell = fair - 2 if ts > 900_000 else fair + 10
            new = self.take_bids(product, od, min_sell, pos_a, limit, cap=extra)
            orders.extend(new)
            pos_a += sum(o.quantity for o in new)

        # ── Final hard flatten at very end ─────────────────────────────────
        if ts > 975_000:
            if pos_a > 0 and bb is not None:
                orders.append(Order(product, bb, -min(limit + pos_a, pos_a)))
            elif pos_a < 0 and ba is not None:
                orders.append(Order(product, ba,  min(limit - pos_a, -pos_a)))
            return orders

        buy_cap  = max(0, limit - pos_a)
        sell_cap = max(0, limit + pos_a)

        # ── Passive limit orders ────────────────────────────────────────────
        # Bid at fair+1 (just inside the spread) to maintain long position.
        bid_px = math.floor(fair + 1)
        if bb is not None: bid_px = min(bid_px, bb + 1)

        # Ask very wide while below target (we DON'T want to sell the drift).
        # Once above target or in unwind phase, tighten.
        if pos_a > target or ts > 880_000:
            ask_px = math.ceil(fair + 2)
        else:
            ask_px = math.ceil(fair + 20)         # Effectively never fills
        if ba is not None: ask_px = max(ask_px, ba - 1)

        if bid_px >= ask_px:
            bid_px = ask_px - 1

        # Post bid to accumulate / maintain +50
        if pos_a < target and buy_cap > 0:
            size = min(buy_cap, max(10, target - pos_a))
            orders.append(Order(product, bid_px, size))

        # Post ask to reduce when above target or in unwind
        if (pos_a > target or ts > 900_000) and sell_cap > 0:
            size = min(sell_cap, max(5, pos_a - target))
            if size > 0:
                orders.append(Order(product, ask_px, -size))

        return orders

    # ──────────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        mem    = self.load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                result[product] = []
                continue

            pos = state.position.get(product, 0)

            if product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_osmium(state, od, pos, mem)
            else:
                result[product] = self.trade_pepper(state, od, pos, mem)

            m = self.mid(od)
            if m is not None:
                mem["last_mid"][product] = m

        return result, 0, self.dump_state(mem)
