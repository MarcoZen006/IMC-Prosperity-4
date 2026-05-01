import math
import json
from typing import Dict, List, Tuple, Optional
from datamodel import OrderDepth, TradingState, Order


class Trader:
    """
    Active trading bot round 5.
    Trades all available products using microprice, volatility, and reservation
    price logic, with light risk controls and safe default limits for unknown products.
    """

    KNOWN_LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    def __init__(self):
        self.past_microprices: Dict[str, float] = {}
        self.volatility_ema: Dict[str, float] = {}
        self.mid_ema: Dict[str, float] = {}
        self.momentum_ema: Dict[str, float] = {}
        self.fill_bias: Dict[str, float] = {}

    def limit_for(self, product: str) -> int:
        """Return the position limit for a product."""
        if product in self.KNOWN_LIMITS:
            return self.KNOWN_LIMITS[product]
        name = product.upper()
        if "VOUCHER" in name or name.startswith("VEV_"):
            return 300
        if "BASKET" in name:
            return 60
        return 50

    def load_state(self, trader_data: str) -> None:
        """Load saved values from the previous run."""
        if not trader_data:
            return
        try:
            data = json.loads(trader_data)
            self.past_microprices.update({k: float(v) for k, v in data.get("past", {}).items()})
            self.volatility_ema.update({k: float(v) for k, v in data.get("var", {}).items()})
            self.mid_ema.update({k: float(v) for k, v in data.get("ema", {}).items()})
            self.momentum_ema.update({k: float(v) for k, v in data.get("mom", {}).items()})
            self.fill_bias.update({k: float(v) for k, v in data.get("bias", {}).items()})
        except Exception:
            pass

    def save_state(self) -> str:
        """Save values needed for the next run."""
        try:
            return json.dumps({
                "past": self.past_microprices,
                "var": self.volatility_ema,
                "ema": self.mid_ema,
                "mom": self.momentum_ema,
                "bias": self.fill_bias,
            })
        except Exception:
            return ""

    def best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        """Return the best bid and best ask prices."""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def microprice(self, order_depth: OrderDepth) -> Optional[float]:
        """Calculate the volume-weighted microprice."""
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        bid_vol = abs(order_depth.buy_orders[best_bid])
        ask_vol = abs(order_depth.sell_orders[best_ask])
        total = bid_vol + ask_vol
        if total <= 0:
            return (best_bid + best_ask) / 2.0
        return (best_bid * ask_vol + best_ask * bid_vol) / total

    def mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        """Calculate the mid price from the best bid and ask."""
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def imbalance(self, order_depth: OrderDepth) -> float:
        """Measure order book imbalance at the best bid and ask."""
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return 0.0
        bid_vol = abs(order_depth.buy_orders[best_bid])
        ask_vol = abs(order_depth.sell_orders[best_ask])
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return max(-1.0, min(1.0, (bid_vol - ask_vol) / total))

    def update_models(self, product: str, micro: float, mid: float) -> Tuple[float, float, float]:
        """Update the stored volatility, mid-price EMA, and momentum."""
        if product not in self.past_microprices:
            self.past_microprices[product] = micro
            self.volatility_ema[product] = 1.0
            self.mid_ema[product] = mid
            self.momentum_ema[product] = 0.0
            return 1.0, mid, 0.0

        diff = micro - self.past_microprices[product]
        self.past_microprices[product] = micro

        old_var = self.volatility_ema.get(product, 1.0)
        new_var = 0.10 * diff * diff + 0.90 * old_var
        new_var = max(0.25, min(new_var, 10000.0))
        self.volatility_ema[product] = new_var

        old_ema = self.mid_ema.get(product, mid)
        new_ema = 0.04 * mid + 0.96 * old_ema
        self.mid_ema[product] = new_ema

        old_mom = self.momentum_ema.get(product, 0.0)
        new_mom = 0.20 * diff + 0.80 * old_mom
        self.momentum_ema[product] = new_mom

        return new_var, new_ema, new_mom

    def time_fraction(self, timestamp: int) -> float:
        """Convert the timestamp into a value between 0 and 1."""
        if timestamp <= 0:
            return 0.0
        # Some backtesters end at 100k, while others end at 1m.
        denom = 100000.0 if timestamp <= 100000 else 1000000.0
        return max(0.0, min(1.0, timestamp / denom))

    def position(self, state: TradingState, product: str) -> int:
        """Return the current position for a product."""
        return state.position.get(product, 0)

    def add_order(
        self,
        orders: List[Order],
        product: str,
        price: int,
        quantity: int,
        position_after: int,
        limit: int,
    ) -> int:
        """Add an order while keeping the position inside the limit."""
        if quantity == 0 or price < 0:
            return position_after

        if quantity > 0:
            quantity = min(quantity, limit - position_after)
        else:
            quantity = -min(-quantity, limit + position_after)

        if quantity != 0:
            orders.append(Order(product, int(price), int(quantity)))
            position_after += quantity

        return position_after

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        self.load_state(state.traderData)
        result: Dict[str, List[Order]] = {}
        conversions = 0
        t = self.time_fraction(state.timestamp)

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []

            best_bid, best_ask = self.best_bid_ask(order_depth)
            micro = self.microprice(order_depth)
            mid = self.mid_price(order_depth)

            if best_bid is None or best_ask is None or micro is None or mid is None:
                result[product] = orders
                continue

            limit = self.limit_for(product)
            pos0 = self.position(state, product)
            pos_after = pos0
            inv_ratio = pos0 / limit if limit > 0 else 0.0

            variance, ema, momentum = self.update_models(product, micro, mid)
            vol = math.sqrt(max(variance, 0.25))
            spread = max(1, best_ask - best_bid)
            imb = self.imbalance(order_depth)

            # Keep fair value close to microprice so the bot stays active.
            momentum_lean = max(-1.20, min(1.20, 0.10 * momentum))
            imbalance_lean = 0.20 * imb * max(1.0, min(2.0, spread / 2.0))
            ema_lean = max(-1.0, min(1.0, 0.06 * (ema - micro)))
            fair = micro + momentum_lean + imbalance_lean + ema_lean

            # Shift the reservation price based on inventory and late-game risk.
            risk_mult = 1.0 + 0.55 * (t ** 2)
            inventory_penalty = pos0 * 0.050 * variance * risk_mult
            soft_limit_skew = math.tanh(2.0 * inv_ratio) * (0.85 + 0.12 * vol) * risk_mult
            reservation = fair - inventory_penalty - soft_limit_skew

            half_spread = 2.0 + 0.50 * vol
            half_spread = max(1.0, min(half_spread, 24.0))

            # Trade smaller when volatility or inventory is high.
            size = max(1, int(10 / max(1.0, vol)))
            size = max(1, int(size * (1.0 - min(0.60, abs(inv_ratio)))))

            # Reduce size near the end, but keep trading.
            if t > 0.88:
                size = max(1, int(size * 0.80))
            if t > 0.94:
                size = max(1, int(size * 0.60))

            # Take clear profitable prices before placing passive quotes.
            if best_ask < reservation - half_spread:
                if not (t > 0.90 and pos0 > 0):
                    qty = min(abs(order_depth.sell_orders[best_ask]), size)
                    pos_after = self.add_order(orders, product, best_ask, qty, pos_after, limit)

            if best_bid > reservation + half_spread:
                if not (t > 0.90 and pos0 < 0):
                    qty = min(abs(order_depth.buy_orders[best_bid]), size)
                    pos_after = self.add_order(orders, product, best_bid, -qty, pos_after, limit)

            # Place quotes close enough to the market to avoid sitting inactive.
            opt_bid = math.floor(reservation - half_spread)
            opt_ask = math.ceil(reservation + half_spread)

            # Improve the quote when the spread gives enough room.
            if spread >= 3:
                quote_bid = min(best_ask - 1, max(best_bid + 1, opt_bid))
                quote_ask = max(best_bid + 1, min(best_ask - 1, opt_ask))
            else:
                quote_bid = min(best_bid, opt_bid)
                quote_ask = max(best_ask, opt_ask)

            if quote_bid >= quote_ask:
                quote_bid = best_bid
                quote_ask = best_ask

            # Lean order sizes toward reducing inventory.
            buy_size = size
            sell_size = size
            if pos0 > 0:
                sell_size = max(1, int(size * (1.0 + min(1.2, abs(inv_ratio) * 2.0))))
                buy_size = max(1, int(size * (1.0 - min(0.5, abs(inv_ratio)))))
            elif pos0 < 0:
                buy_size = max(1, int(size * (1.0 + min(1.2, abs(inv_ratio) * 2.0))))
                sell_size = max(1, int(size * (1.0 - min(0.5, abs(inv_ratio)))))

            allow_buy = pos_after < limit and not (t > 0.92 and pos0 > 0)
            allow_sell = pos_after > -limit and not (t > 0.92 and pos0 < 0)

            if allow_buy:
                pos_after = self.add_order(orders, product, quote_bid, buy_size, pos_after, limit)
            if allow_sell:
                pos_after = self.add_order(orders, product, quote_ask, -sell_size, pos_after, limit)

            # Release some inventory near the end if the position is too large.
            if t > 0.90 and abs(pos0) > max(5, int(0.06 * limit)):
                exit_qty = max(1, min(abs(pos0) // 3, size * 3))
                if pos0 > 0:
                    pos_after = self.add_order(orders, product, best_bid, -exit_qty, pos_after, limit)
                elif pos0 < 0:
                    pos_after = self.add_order(orders, product, best_ask, exit_qty, pos_after, limit)

            result[product] = orders

        return result, conversions, self.save_state()
