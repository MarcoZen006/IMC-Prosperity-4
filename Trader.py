from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import json
import math


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 50,
        "INTARIAN_PEPPER_ROOT": 50,
    }

    OSMIUM_ANCHOR = 10000.0
    PEPPER_SLOPE_PER_TS = 0.001  # +1000 over a full day of 1,000,000 timestamp units

    def load_state(self, trader_data: str) -> Dict:
        if not trader_data:
            return {
                "last_mid": {},
                "base": {},
                "ret_ema": {},
                "dev_ema": {},
            }
        try:
            return json.loads(trader_data)
        except Exception:
            return {
                "last_mid": {},
                "base": {},
                "ret_ema": {},
                "dev_ema": {},
            }

    def dump_state(self, state: Dict) -> str:
        return json.dumps(state, separators=(",", ":"))

    def best_bid_ask(self, od: OrderDepth) -> Tuple[int, int]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    def top_sizes(self, od: OrderDepth) -> Tuple[int, int]:
        best_bid, best_ask = self.best_bid_ask(od)
        bid_sz = od.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        ask_sz = -od.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        return bid_sz, ask_sz

    def mid_price(self, od: OrderDepth):
        best_bid, best_ask = self.best_bid_ask(od)
        if best_bid is not None and best_ask is not None:
            return 0.5 * (best_bid + best_ask)
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def spread(self, od: OrderDepth) -> float:
        best_bid, best_ask = self.best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return 2.0
        return float(best_ask - best_bid)

    def imbalance(self, od: OrderDepth) -> float:
        bid_sz, ask_sz = self.top_sizes(od)
        denom = bid_sz + ask_sz
        if denom <= 0:
            return 0.0
        return (bid_sz - ask_sz) / denom

    def microprice(self, od: OrderDepth):
        best_bid, best_ask = self.best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return self.mid_price(od)
        bid_sz, ask_sz = self.top_sizes(od)
        denom = bid_sz + ask_sz
        if denom <= 0:
            return 0.5 * (best_bid + best_ask)
        return (best_ask * bid_sz + best_bid * ask_sz) / denom

    def level_take(self, product: str, fair: float, od: OrderDepth, pos: int, limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_cap = limit - pos
        sell_cap = limit + pos

        for ask in sorted(od.sell_orders.keys()):
            ask_vol = -od.sell_orders[ask]
            edge = fair - ask
            if buy_cap <= 0:
                break
            if edge >= 2.0:
                qty = min(buy_cap, ask_vol)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap -= qty
            else:
                break

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            bid_vol = od.buy_orders[bid]
            edge = bid - fair
            if sell_cap <= 0:
                break
            if edge >= 2.0:
                qty = min(sell_cap, bid_vol)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
            else:
                break

        return orders

    def fair_osmium(self, timestamp: int, mid: float, od: OrderDepth, mem: Dict) -> float:
        last_mid = mem["last_mid"].get("ASH_COATED_OSMIUM", mid)
        ret = mid - last_mid
        prev_ret_ema = mem["ret_ema"].get("ASH_COATED_OSMIUM", 0.0)
        ret_ema = 0.80 * prev_ret_ema + 0.20 * ret

        dev = mid - self.OSMIUM_ANCHOR
        prev_dev_ema = mem["dev_ema"].get("ASH_COATED_OSMIUM", dev)
        dev_ema = 0.92 * prev_dev_ema + 0.08 * dev

        imb = self.imbalance(od)
        mp = self.microprice(od)
        mp_edge = 0.0 if mp is None else (mp - mid)

        # Mean-reversion dominates; microprice/imbalance provides short-horizon alpha.
        fair = (
            self.OSMIUM_ANCHOR
            - 0.55 * dev_ema
            - 0.35 * ret_ema
            + 0.60 * mp_edge
            + 0.80 * imb
        )

        mem["ret_ema"]["ASH_COATED_OSMIUM"] = ret_ema
        mem["dev_ema"]["ASH_COATED_OSMIUM"] = dev_ema
        return fair

    def fair_pepper(self, timestamp: int, mid: float, od: OrderDepth, mem: Dict) -> float:
        trend_base_now = mid - self.PEPPER_SLOPE_PER_TS * timestamp
        prev_base = mem["base"].get("INTARIAN_PEPPER_ROOT", trend_base_now)
        base = 0.995 * prev_base + 0.005 * trend_base_now

        last_mid = mem["last_mid"].get("INTARIAN_PEPPER_ROOT", mid)
        ret = mid - last_mid
        prev_ret_ema = mem["ret_ema"].get("INTARIAN_PEPPER_ROOT", 0.0)
        ret_ema = 0.85 * prev_ret_ema + 0.15 * ret

        imb = self.imbalance(od)
        mp = self.microprice(od)
        mp_edge = 0.0 if mp is None else (mp - mid)

        fair = (
            base
            + self.PEPPER_SLOPE_PER_TS * timestamp
            + 0.35 * ret_ema
            + 0.55 * mp_edge
            + 0.90 * imb
        )

        mem["base"]["INTARIAN_PEPPER_ROOT"] = base
        mem["ret_ema"]["INTARIAN_PEPPER_ROOT"] = ret_ema
        return fair

    def make_quotes(
        self,
        product: str,
        fair: float,
        od: OrderDepth,
        pos: int,
        limit: int,
        timestamp: int,
    ) -> List[Order]:
        orders: List[Order] = []
        best_bid, best_ask = self.best_bid_ask(od)
        spread = max(1.0, self.spread(od))
        imb = self.imbalance(od)
        mp = self.microprice(od)
        mid = self.mid_price(od)
        alpha = 0.0 if (mp is None or mid is None) else (mp - mid) + 0.5 * imb

        # Dynamic width: wider when spread is wider or signal is toxic.
        base_half = 1 if spread <= 10 else 2
        half = base_half + (1 if abs(alpha) > 0.7 else 0)

        inv_ratio = pos / limit if limit else 0.0
        reservation = fair - 2.2 * inv_ratio

        bid_px = math.floor(reservation - half)
        ask_px = math.ceil(reservation + half)

        # Dynamic execution from imbalance: avoid being picked off on the wrong side.
        if alpha > 1.2:
            ask_px += 1
            bid_px += 1
        elif alpha < -1.2:
            ask_px -= 1
            bid_px -= 1

        # Join or lightly improve the book where sensible.
        if best_bid is not None:
            bid_px = min(bid_px, best_bid + 1)
        if best_ask is not None:
            ask_px = max(ask_px, best_ask - 1)

        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_cap = limit - pos
        sell_cap = limit + pos

        time_left = max(0, 1_000_000 - timestamp)
        unwind = time_left < 120_000
        severe_unwind = time_left < 40_000

        size_base = 8 if product == "ASH_COATED_OSMIUM" else 10
        size = max(3, int(size_base * (1.0 - 0.35 * abs(inv_ratio))))
        if unwind:
            size = max(4, size + 2)

        post_bid = buy_cap > 0
        post_ask = sell_cap > 0

        if alpha > 1.5:
            post_ask = False
        elif alpha < -1.5:
            post_bid = False

        if severe_unwind:
            if pos > 0:
                post_bid = False
                ask_px = min(ask_px, (best_bid if best_bid is not None else ask_px) + 1)
                size = min(sell_cap, max(size, abs(pos) // 2 + 1))
            elif pos < 0:
                post_ask = False
                bid_px = max(bid_px, (best_ask if best_ask is not None else bid_px) - 1)
                size = min(buy_cap, max(size, abs(pos) // 2 + 1))

        if post_bid and buy_cap > 0:
            orders.append(Order(product, int(bid_px), min(size, buy_cap)))
        if post_ask and sell_cap > 0:
            orders.append(Order(product, int(ask_px), -min(size, sell_cap)))

        return orders

    def run(self, state: TradingState):
        mem = self.load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                result[product] = []
                continue

            limit = self.POSITION_LIMITS[product]
            pos = state.position.get(product, 0)
            mid = self.mid_price(od)
            if mid is None:
                result[product] = []
                continue

            if product == "ASH_COATED_OSMIUM":
                fair = self.fair_osmium(state.timestamp, mid, od, mem)
            else:
                fair = self.fair_pepper(state.timestamp, mid, od, mem)

            orders: List[Order] = []
            orders.extend(self.level_take(product, fair, od, pos, limit))

            temp_pos = pos + sum(o.quantity for o in orders)
            orders.extend(self.make_quotes(product, fair, od, temp_pos, limit, state.timestamp))

            result[product] = orders
            mem["last_mid"][product] = mid

        return result, 0, self.dump_state(mem)
