try:
    from prosperity3bt.datamodel import OrderDepth, TradingState, Order
except ImportError:
    from datamodel import OrderDepth, TradingState, Order

from typing import Dict, List, Tuple
import json


HYDRO = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"

# Position limits from the Round 3 statement.
POSITION_LIMITS = {
    HYDRO: 200,
    VELVET: 200,
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

# Empirical Round 3 first-session structure:
# the public logs showed that the old bot lost mostly by being long Hydrogel and
# by churn/hedging in Velvetfruit. This version does the opposite: enter one
# controlled early short, then buy it back only after a measured move in our
# favour. No Black-Scholes hedge, no passive market making, no repeated churn.
SCALP_CONFIG = {
    HYDRO: {
        "target": -200,
        "entry_until": 5_000,
        "take_profit": 70.0,
    },
    VELVET: {
        "target": -200,
        "entry_until": 5_000,
        "take_profit": 12.0,
    },
    "VEV_5000": {
        "target": -300,
        "entry_until": 5_000,
        "take_profit": 10.0,
    },
    "VEV_5100": {
        "target": -300,
        "entry_until": 5_000,
        "take_profit": 10.0,
    },
    "VEV_5200": {
        "target": -300,
        "entry_until": 5_000,
        "take_profit": 7.0,
    },
    "VEV_5300": {
        "target": -300,
        "entry_until": 5_000,
        "take_profit": 7.0,
    },
}


class Trader:
    def __init__(self):
        self.memory: Dict[str, Dict[str, float | bool]] = {}

    # ------------------------------------------------------------------
    # State handling
    # ------------------------------------------------------------------

    def _default_memory(self) -> Dict[str, Dict[str, float | bool]]:
        mem: Dict[str, Dict[str, float | bool]] = {}
        for product in SCALP_CONFIG:
            mem[product] = {
                "short_qty": 0.0,
                "short_notional": 0.0,
                "done": False,
            }
        return mem

    def _load_memory(self, trader_data: str) -> None:
        mem = self._default_memory()
        if trader_data:
            try:
                loaded = json.loads(trader_data)
                if isinstance(loaded, dict):
                    for product in SCALP_CONFIG:
                        if product in loaded and isinstance(loaded[product], dict):
                            for key in mem[product]:
                                if key in loaded[product]:
                                    mem[product][key] = loaded[product][key]
            except Exception:
                pass
        self.memory = mem

    def _dump_memory(self) -> str:
        return json.dumps(self.memory, separators=(",", ":"))

    # ------------------------------------------------------------------
    # Book helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sorted_bids(od: OrderDepth) -> List[Tuple[int, int]]:
        return sorted(
            ((price, qty) for price, qty in od.buy_orders.items() if qty > 0),
            key=lambda x: -x[0],
        )

    @staticmethod
    def _sorted_asks(od: OrderDepth) -> List[Tuple[int, int]]:
        return sorted(
            ((price, -qty) for price, qty in od.sell_orders.items() if -qty > 0),
            key=lambda x: x[0],
        )

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------

    def _short_average(self, product: str) -> float | None:
        short_qty = float(self.memory[product].get("short_qty", 0.0))
        if short_qty <= 0.0:
            return None
        short_notional = float(self.memory[product].get("short_notional", 0.0))
        return short_notional / short_qty

    def _trade_short_scalp(
        self,
        product: str,
        od: OrderDepth,
        current_position: int,
        timestamp: int,
    ) -> List[Order]:
        cfg = SCALP_CONFIG[product]
        limit = POSITION_LIMITS[product]
        target = int(cfg["target"])
        entry_until = int(cfg["entry_until"])
        take_profit = float(cfg["take_profit"])
        mem = self.memory[product]

        orders: List[Order] = []
        estimated_position = current_position

        if bool(mem.get("done", False)):
            return orders

        # 1) Enter the short only at the very start of the session. This avoids
        # repeated re-shorting after a profitable cover.
        if timestamp <= entry_until and estimated_position > target:
            sell_room = min(estimated_position - target, limit + estimated_position)
            for price, volume in self._sorted_bids(od):
                if sell_room <= 0:
                    break
                qty = min(volume, sell_room)
                if qty <= 0:
                    continue
                orders.append(Order(product, price, -qty))
                estimated_position -= qty
                sell_room -= qty

                # These are crossing orders into visible bid volume, so they are
                # treated as filled immediately for average-entry tracking.
                mem["short_qty"] = float(mem.get("short_qty", 0.0)) + qty
                mem["short_notional"] = float(mem.get("short_notional", 0.0)) + qty * price

        # 2) Buy back only after the book has moved far enough in our favour.
        # The average entry is based on actual generated short fills above.
        avg_short = self._short_average(product)
        if estimated_position < 0 and avg_short is not None:
            cover_trigger = avg_short - take_profit
            buy_room = -estimated_position
            for price, volume in self._sorted_asks(od):
                if buy_room <= 0 or price > cover_trigger:
                    break
                qty = min(volume, buy_room)
                if qty <= 0:
                    continue
                orders.append(Order(product, price, qty))
                estimated_position += qty
                buy_room -= qty

            if estimated_position >= 0:
                mem["done"] = True

        return orders

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, state: TradingState):
        self._load_memory(state.traderData)

        result: Dict[str, List[Order]] = {}
        for product, od in state.order_depths.items():
            if product in SCALP_CONFIG:
                pos = state.position.get(product, 0)
                result[product] = self._trade_short_scalp(
                    product,
                    od,
                    pos,
                    state.timestamp,
                )
            else:
                result[product] = []

        return result, 0, self._dump_memory()
