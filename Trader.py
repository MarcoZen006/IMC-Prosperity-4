from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 50,
        "INTARIAN_PEPPER_ROOT": 50,
    }

    FAIR_VALUES = {
        "ASH_COATED_OSMIUM": 10000,
        "INTARIAN_PEPPER_ROOT": 12000,
    }

    def get_best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def get_mid_price(self, order_depth: OrderDepth):
        best_bid, best_ask = self.get_best_bid_ask(order_depth)

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        elif best_bid is not None:
            return best_bid
        elif best_ask is not None:
            return best_ask
        return None

    def calc_fair_value(self, product: str, mid_price: float, timestamp: int):
        """
        Product-specific alpha logic
        """

        if product == "ASH_COATED_OSMIUM":
            # Strong mean reversion to 10000
            return 10000

        elif product == "INTARIAN_PEPPER_ROOT":
            # Historical drift + short-term signal
            drift = timestamp / 10000 * 5
            return max(12000 + drift, mid_price)

        return self.FAIR_VALUES[product]

    def run(self, state: TradingState):
<<<<<<< HEAD
=======
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
##TEST
        # Orders to be placed on exchange matching engine
>>>>>>> 9dda59c3d62c1bacacd7880affb8a54335457123
        result = {}

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []

            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS[product]

            best_bid, best_ask = self.get_best_bid_ask(order_depth)
            mid_price = self.get_mid_price(order_depth)

            if mid_price is None:
                result[product] = orders
                continue

            fair_value = self.calc_fair_value(
                product,
                mid_price,
                state.timestamp
            )

            buy_capacity = limit - position
            sell_capacity = limit + position

            # ========================
            # AGGRESSIVE EDGE TAKING
            # ========================
            if best_ask is not None and best_ask < fair_value - 1:
                volume = min(
                    -order_depth.sell_orders[best_ask],
                    buy_capacity
                )
                if volume > 0:
                    orders.append(Order(product, best_ask, volume))

            if best_bid is not None and best_bid > fair_value + 1:
                volume = min(
                    order_depth.buy_orders[best_bid],
                    sell_capacity
                )
                if volume > 0:
                    orders.append(Order(product, best_bid, -volume))

            # ========================
            # MARKET MAKING LAYER
            # ========================
            inventory_skew = position * 0.2

            bid_quote = int(fair_value - 1 - inventory_skew)
            ask_quote = int(fair_value + 1 - inventory_skew)

            mm_size = 10

            if buy_capacity >= mm_size:
                orders.append(Order(product, bid_quote, mm_size))

            if sell_capacity >= mm_size:
                orders.append(Order(product, ask_quote, -mm_size))

            result[product] = orders

        traderData = ""
        conversions = 0
<<<<<<< HEAD

        return result, conversions, traderData
=======
        return result, conversions, traderData
>>>>>>> 9dda59c3d62c1bacacd7880affb8a54335457123
