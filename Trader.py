
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json
import math


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 50,
        "INTARIAN_PEPPER_ROOT": 50,
    }

    OSMIUM_FAIR = 10000.0
    PEPPER_SLOPE = 0.001  # from round-1 data: ~+1000 over a day

    def default_state(self) -> Dict:
        return {
            "pepper_base": None,
            "last_mid": {},
            "osmium_dev_ema": 0.0,
            "osmium_ret_ema": 0.0,
        }

    def load_state(self, trader_data: str) -> Dict:
        if not trader_data:
            return self.default_state()
        try:
            state = json.loads(trader_data)
            default = self.default_state()
            for k, v in default.items():
                if k not in state:
                    state[k] = v
            return state
        except Exception:
            return self.default_state()

    def dump_state(self, state: Dict) -> str:
        return json.dumps(state, separators=(",", ":"))

    def best_bid_ask(self, od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    def mid_price(self, od: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(od)
        if best_bid is not None and best_ask is not None:
            return 0.5 * (best_bid + best_ask)
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def clamp(self, x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, x))

    def take_asks(
        self,
        product: str,
        od: OrderDepth,
        max_price: float,
        pos: int,
        limit: int,
        max_qty: Optional[int] = None,
    ) -> List[Order]:
        orders: List[Order] = []
        buy_cap = limit - pos
        if max_qty is not None:
            buy_cap = min(buy_cap, max_qty)

        for ask in sorted(od.sell_orders.keys()):
            if buy_cap <= 0:
                break
            ask_qty = -od.sell_orders[ask]
            if ask <= max_price:
                qty = min(buy_cap, ask_qty)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap -= qty
            else:
                break
        return orders

    def take_bids(
        self,
        product: str,
        od: OrderDepth,
        min_price: float,
        pos: int,
        limit: int,
        max_qty: Optional[int] = None,
    ) -> List[Order]:
        orders: List[Order] = []
        sell_cap = limit + pos
        if max_qty is not None:
            sell_cap = min(sell_cap, max_qty)

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if sell_cap <= 0:
                break
            bid_qty = od.buy_orders[bid]
            if bid >= min_price:
                qty = min(sell_cap, bid_qty)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
            else:
                break
        return orders

    def quote_order(
        self,
        product: str,
        price: int,
        qty: int,
    ) -> Optional[Order]:
        if qty == 0:
            return None
        return Order(product, int(price), int(qty))

    def trade_osmium(
        self,
        state: TradingState,
        od: OrderDepth,
        pos: int,
        mem: Dict,
    ) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        best_bid, best_ask = self.best_bid_ask(od)
        mid = self.mid_price(od)
        if mid is None:
            return orders

        last_mid = mem["last_mid"].get(product, mid)
        ret = mid - last_mid
        dev = mid - self.OSMIUM_FAIR

        mem["osmium_ret_ema"] = 0.85 * mem.get("osmium_ret_ema", 0.0) + 0.15 * ret
        mem["osmium_dev_ema"] = 0.92 * mem.get("osmium_dev_ema", 0.0) + 0.08 * dev

        fair = (
            self.OSMIUM_FAIR
            - 0.55 * mem["osmium_dev_ema"]
            - 0.20 * mem["osmium_ret_ema"]
        )

        # Only take clear mispricings. The round-1 data shows osmium is near-stationary.
        orders.extend(self.take_asks(product, od, fair - 2, pos, limit))
        pos_after_take = pos + sum(o.quantity for o in orders)
        orders.extend(self.take_bids(product, od, fair + 2, pos_after_take, limit))

        pos_after_take = pos + sum(o.quantity for o in orders)
        inv = pos_after_take / limit

        # Inventory-skewed passive quotes around fair.
        bid_px = math.floor(fair - 3 - 2.0 * inv)
        ask_px = math.ceil(fair + 3 - 2.0 * inv)

        if best_bid is not None:
            bid_px = min(bid_px, best_bid + 1)
        if best_ask is not None:
            ask_px = max(ask_px, best_ask - 1)
        if bid_px >= ask_px:
            bid_px = ask_px - 1

        buy_cap = max(0, limit - pos_after_take)
        sell_cap = max(0, limit + pos_after_take)
        size = max(4, 10 - int(6 * abs(inv)))

        # Endgame flattening.
        if state.timestamp > 930_000:
            if pos_after_take > 0:
                if best_bid is not None:
                    qty = min(sell_cap, max(5, abs(pos_after_take)))
                    orders.append(Order(product, best_bid, -qty))
            elif pos_after_take < 0:
                if best_ask is not None:
                    qty = min(buy_cap, max(5, abs(pos_after_take)))
                    orders.append(Order(product, best_ask, qty))
            return orders

        if buy_cap > 0:
            orders.append(Order(product, bid_px, min(size, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(product, ask_px, -min(size, sell_cap)))
        return orders

    def pepper_fair(self, timestamp: int, mid: float, mem: Dict) -> float:
        instantaneous_base = mid - self.PEPPER_SLOPE * timestamp
        base = mem.get("pepper_base")
        if base is None:
            base = instantaneous_base
        else:
            base = 0.98 * base + 0.02 * instantaneous_base
        mem["pepper_base"] = base
        return base + self.PEPPER_SLOPE * timestamp

    def target_pepper_position(self, timestamp: int) -> int:
        # The data shows a near-linear upward drift all day.
        # Stay max long for most of the day, then unwind late.
        if timestamp < 820_000:
            return 50
        if timestamp < 900_000:
            return 35
        if timestamp < 960_000:
            return 15
        return 0

    def trade_pepper(
        self,
        state: TradingState,
        od: OrderDepth,
        pos: int,
        mem: Dict,
    ) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        best_bid, best_ask = self.best_bid_ask(od)
        mid = self.mid_price(od)
        if mid is None:
            return orders

        fair = self.pepper_fair(state.timestamp, mid, mem)
        target_pos = self.target_pepper_position(state.timestamp)

        # Strong directional edge: buy below/near fair aggressively.
        # Because the slope is positive, we tolerate paying slightly through fair early.
        if state.timestamp < 850_000:
            max_buy_price = fair + 2
        elif state.timestamp < 930_000:
            max_buy_price = fair + 1
        else:
            max_buy_price = fair

        need_to_buy = max(0, target_pos - pos)
        if need_to_buy > 0:
            orders.extend(self.take_asks(product, od, max_buy_price, pos, limit, max_qty=need_to_buy))

        pos_after_take = pos + sum(o.quantity for o in orders)

        # If clearly over fair or late in day, allow selling.
        min_sell_price = fair + 3
        if state.timestamp > 900_000:
            min_sell_price = fair - 1

        extra_long = max(0, pos_after_take - target_pos)
        if extra_long > 0:
            orders.extend(self.take_bids(product, od, min_sell_price, pos_after_take, limit, max_qty=extra_long))

        pos_after_take = pos + sum(o.quantity for o in orders)
        buy_cap = max(0, limit - pos_after_take)
        sell_cap = max(0, limit + pos_after_take)

        # Passive logic:
        # - mostly quote bid-side to maintain long inventory
        # - only quote ask-side when above target or late in the day
        inv = pos_after_take / limit if limit else 0.0

        # Bid quote: slightly aggressive if below target.
        bid_px = math.floor(fair - 1)
        if best_bid is not None:
            bid_px = min(int(bid_px), best_bid + 1)

        # Ask quote: keep wide unless reducing inventory.
        ask_px = math.ceil(fair + 6)
        if pos_after_take > target_pos or state.timestamp > 900_000:
            ask_px = math.ceil(fair + 1)
        if best_ask is not None:
            ask_px = max(int(ask_px), best_ask - 1)

        if bid_px >= ask_px:
            bid_px = ask_px - 1

        if state.timestamp > 970_000:
            # Final flatten.
            if pos_after_take > 0 and best_bid is not None:
                qty = min(sell_cap, abs(pos_after_take))
                orders.append(Order(product, best_bid, -qty))
            elif pos_after_take < 0 and best_ask is not None:
                qty = min(buy_cap, abs(pos_after_take))
                orders.append(Order(product, best_ask, qty))
            return orders

        # Maintain long bias through most of the day.
        if pos_after_take < target_pos and buy_cap > 0:
            size = min(buy_cap, max(6, target_pos - pos_after_take))
            orders.append(Order(product, bid_px, size))

        if (pos_after_take > target_pos or state.timestamp > 880_000) and sell_cap > 0:
            size = min(sell_cap, max(4, pos_after_take - target_pos))
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

            pos = state.position.get(product, 0)

            if product == "ASH_COATED_OSMIUM":
                orders = self.trade_osmium(state, od, pos, mem)
            else:
                orders = self.trade_pepper(state, od, pos, mem)

            result[product] = orders

            mid = self.mid_price(od)
            if mid is not None:
                mem["last_mid"][product] = mid

        return result, 0, self.dump_state(mem)
