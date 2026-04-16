from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json
import math


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 50,
        "INTARIAN_PEPPER_ROOT": 50,
    }

    OSMIUM_LONG_MEAN = 10000.0
    PEPPER_SLOPE = 0.001000

    def default_state(self) -> Dict:
        return {
            "last_mid": {},
            "pepper_intercept": None,
            "osmium_adj_fair": self.OSMIUM_LONG_MEAN,
            "mid_hist": {
                "ASH_COATED_OSMIUM": [],
                "INTARIAN_PEPPER_ROOT": [],
            },
            "spread_hist": {
                "ASH_COATED_OSMIUM": [],
                "INTARIAN_PEPPER_ROOT": [],
            },
            "imb_hist": {
                "ASH_COATED_OSMIUM": [],
                "INTARIAN_PEPPER_ROOT": [],
            },
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
            for bucket in ["mid_hist", "spread_hist", "imb_hist", "last_mid"]:
                if bucket not in s or not isinstance(s[bucket], dict):
                    s[bucket] = base[bucket]
            for p in self.POSITION_LIMITS:
                if p not in s["mid_hist"]:
                    s["mid_hist"][p] = []
                if p not in s["spread_hist"]:
                    s["spread_hist"][p] = []
                if p not in s["imb_hist"]:
                    s["imb_hist"][p] = []
            return s
        except Exception:
            return base

    def dump_state(self, s: Dict) -> str:
        return json.dumps(s, separators=(",", ":"))

    def best_bid_ask(self, od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        return bb, ba

    def mid(self, od: OrderDepth) -> Optional[float]:
        bb, ba = self.best_bid_ask(od)
        if bb is not None and ba is not None:
            return 0.5 * (bb + ba)
        return float(bb) if bb is not None else (float(ba) if ba is not None else None)

    def top_levels(self, od: OrderDepth, levels: int = 3) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = sorted(od.buy_orders.items(), reverse=True)[:levels]
        asks = sorted(od.sell_orders.items())[:levels]
        return bids, asks

    def order_book_imbalance(self, od: OrderDepth, levels: int = 3) -> float:
        bids, asks = self.top_levels(od, levels)
        bid_vol = sum(max(0, qty) for _, qty in bids)
        ask_vol = sum(max(0, -qty) for _, qty in asks)
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def push_hist(self, hist: List[float], value: float, maxlen: int = 25) -> None:
        hist.append(value)
        if len(hist) > maxlen:
            del hist[0]

    def mean(self, arr: List[float]) -> float:
        return sum(arr) / len(arr) if arr else 0.0

    def stdev(self, arr: List[float]) -> float:
        n = len(arr)
        if n < 2:
            return 0.0
        mu = self.mean(arr)
        return math.sqrt(sum((x - mu) * (x - mu) for x in arr) / (n - 1))

    def slope_per_tick(self, mids: List[float]) -> float:
        n = len(mids)
        if n < 2:
            return 0.0
        x_mean = 0.5 * (n - 1)
        y_mean = self.mean(mids)
        num = 0.0
        den = 0.0
        for i, y in enumerate(mids):
            dx = i - x_mean
            num += dx * (y - y_mean)
            den += dx * dx
        if den == 0:
            return 0.0
        return num / den

    def regime(self, product: str, mem: Dict) -> str:
        mids = mem["mid_hist"][product]
        spreads = mem["spread_hist"][product]
        imbs = mem["imb_hist"][product]

        if len(mids) < 8:
            return "normal"

        diffs = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        vol = self.stdev(diffs[-12:]) if len(diffs) >= 2 else 0.0
        slope = self.slope_per_tick(mids[-12:])
        avg_spread = self.mean(spreads[-8:]) if spreads else 0.0
        imb_strength = abs(self.mean(imbs[-5:])) if imbs else 0.0

        if product == "INTARIAN_PEPPER_ROOT":
            if slope > 0.45:
                return "strong_uptrend"
            if slope > 0.15:
                return "uptrend"
            if vol > 4.5 or avg_spread > 15:
                return "volatile"
            return "normal"

        if vol > 5.5 or avg_spread > 18:
            return "volatile"
        if imb_strength > 0.30:
            return "pressure"
        return "mean_revert"

    def take_asks(self, product, od, max_px, pos, limit, cap=None) -> List[Order]:
        orders, room = [], limit - pos
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

    def take_bids(self, product, od, min_px, pos, limit, cap=None) -> List[Order]:
        orders, room = [], limit + pos
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

    def trade_osmium(self, state: TradingState, od: OrderDepth, pos: int, mem: Dict) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        mid = self.mid(od)
        if mid is None:
            return orders

        last_mid = mem["last_mid"].get(product, mid)
        autocorr_fair = 0.5 * (mid + last_mid)
        raw_fair = 0.82 * autocorr_fair + 0.18 * self.OSMIUM_LONG_MEAN

        imbalance = self.order_book_imbalance(od, 3)
        regime = self.regime(product, mem)

        imb_shift = 0.0
        if regime == "pressure":
            imb_shift = 1.2 * imbalance
        elif regime == "mean_revert":
            imb_shift = 0.5 * imbalance

        raw_fair += imb_shift
        prev_fair = mem.get("osmium_adj_fair", raw_fair)
        fair = 0.62 * raw_fair + 0.38 * prev_fair
        mem["osmium_adj_fair"] = fair

        inv = pos / limit
        skew = 4.0 * inv
        if regime == "volatile":
            skew += 0.8 * inv
            cross_thresh = 4
            half_spread = 4
        else:
            cross_thresh = 3
            half_spread = 3

        skewed_fair = fair - skew

        orders.extend(self.take_asks(product, od, skewed_fair - cross_thresh, pos, limit))
        pos_a = pos + sum(o.quantity for o in orders)
        orders.extend(self.take_bids(product, od, skewed_fair + cross_thresh, pos_a, limit))
        pos_a = pos + sum(o.quantity for o in orders)

        if state.timestamp > 930_000:
            if pos_a > 0 and bb is not None:
                orders.append(Order(product, bb, -min(pos_a, limit + pos_a)))
            elif pos_a < 0 and ba is not None:
                orders.append(Order(product, ba, min(-pos_a, limit - pos_a)))
            return orders

        inv_a = pos_a / limit
        size = max(5, 14 - int(8 * abs(inv_a)))
        if regime == "volatile":
            size = max(4, size - 2)

        bid_px = math.floor(skewed_fair - half_spread)
        ask_px = math.ceil(skewed_fair + half_spread)
        if bb is not None:
            bid_px = min(bid_px, bb + 1)
        if ba is not None:
            ask_px = max(ask_px, ba - 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_cap = max(0, limit - pos_a)
        sell_cap = max(0, limit + pos_a)

        if buy_cap > 0:
            orders.append(Order(product, bid_px, min(size, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -min(size, sell_cap)))

        return orders

    def pepper_fair(self, ts: int, mid: float, mem: Dict) -> float:
        obs_intercept = mid - self.PEPPER_SLOPE * ts
        stored = mem.get("pepper_intercept")
        if stored is None:
            mem["pepper_intercept"] = obs_intercept
            return mid
        alpha = 0.15 if ts < 20_000 else 0.01
        mem["pepper_intercept"] = (1 - alpha) * stored + alpha * obs_intercept
        return mem["pepper_intercept"] + self.PEPPER_SLOPE * ts

    def target_pos_pepper(self, ts: int, regime: str) -> int:
        if ts < 870_000:
            return 50
        if ts < 930_000:
            return 30
        if ts < 965_000:
            return 10
        return 0

    def trade_pepper(self, state: TradingState, od: OrderDepth, pos: int, mem: Dict) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        bb, ba = self.best_bid_ask(od)
        mid = self.mid(od)
        if mid is None:
            return orders

        ts = state.timestamp
        regime = self.regime(product, mem)
        imbalance = self.order_book_imbalance(od, 3)

        fair = self.pepper_fair(ts, mid, mem)
        if regime in ("strong_uptrend", "uptrend"):
            fair += 0.5 * max(0.0, imbalance)
        elif regime == "volatile":
            fair += 0.2 * imbalance

        target = self.target_pos_pepper(ts, regime)
        pos_a = pos

        need = max(0, target - pos_a)
        if need > 0:
            if ts < 10_000:
                max_buy = fair + 10
            elif ts < 50_000:
                max_buy = fair + 6
            elif ts < 200_000:
                max_buy = fair + 4
            elif ts < 800_000:
                max_buy = fair + 2
            else:
                max_buy = fair + 1

            if regime == "strong_uptrend" and ts < 200_000:
                max_buy += 1
            if regime == "volatile":
                max_buy -= 1

            new = self.take_asks(product, od, max_buy, pos_a, limit, cap=need)
            orders.extend(new)
            pos_a += sum(o.quantity for o in new)

        extra = max(0, pos_a - target)
        if extra > 0:
            min_sell = fair - 2 if ts > 900_000 else fair + 10
            if regime == "volatile" and ts < 900_000:
                min_sell = fair + 6
            new = self.take_bids(product, od, min_sell, pos_a, limit, cap=extra)
            orders.extend(new)
            pos_a += sum(o.quantity for o in new)

        if ts > 975_000:
            if pos_a > 0 and bb is not None:
                orders.append(Order(product, bb, -min(pos_a, limit + pos_a)))
            elif pos_a < 0 and ba is not None:
                orders.append(Order(product, ba, min(-pos_a, limit - pos_a)))
            return orders

        buy_cap = max(0, limit - pos_a)
        sell_cap = max(0, limit + pos_a)

        bid_px = math.floor(fair + 1)
        if bb is not None:
            bid_px = min(bid_px, bb + 1)

        if pos_a > target or ts > 880_000:
            ask_px = math.ceil(fair + 2)
        else:
            ask_px = math.ceil(fair + 20)
        if ba is not None:
            ask_px = max(ask_px, ba - 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        if pos_a < target and buy_cap > 0:
            size = min(buy_cap, max(10, target - pos_a))
            if regime == "volatile":
                size = max(5, size - 4)
            orders.append(Order(product, bid_px, size))

        if (pos_a > target or ts > 900_000) and sell_cap > 0:
            size = min(sell_cap, max(5, pos_a - target))
            if size > 0:
                orders.append(Order(product, ask_px, -size))

        return orders

    def run(self, state: TradingState):
        mem = self.load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                result[product] = []
                continue

            mid = self.mid(od)
            bb, ba = self.best_bid_ask(od)
            if mid is not None:
                self.push_hist(mem["mid_hist"][product], mid)
            if bb is not None and ba is not None:
                self.push_hist(mem["spread_hist"][product], ba - bb)
            self.push_hist(mem["imb_hist"][product], self.order_book_imbalance(od, 3))

            pos = state.position.get(product, 0)
            if product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_osmium(state, od, pos, mem)
            else:
                result[product] = self.trade_pepper(state, od, pos, mem)

            if mid is not None:
                mem["last_mid"][product] = mid

        return result, 0, self.dump_state(mem)
