from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    """
    Round 4 trader.
    Trades HYDROGEL and VEV vouchers. VELVET is disabled for now.
    Uses low-vol BSM pricing, inventory control, and passive quotes.
    """

    HYDROGEL = "HYDROGEL_PACK"
    VELVET = "VELVETFRUIT_EXTRACT"

    VOUCHERS = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    LIMITS = {
        HYDROGEL: 200,
        VELVET: 200,
        "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300,
        "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300,
        "VEV_5400": 300, "VEV_5500": 300, "VEV_6000": 300,
        "VEV_6500": 300,
    }

    # These vouchers usually sit at the floor, so skip trading them.
    FLOOR_VOUCHERS = {"VEV_6000", "VEV_6500"}

    # Low BSM volatility used to value vouchers conservatively.
    OPTION_SIGMA = 0.10

    EMA_ALPHA = {
        HYDROGEL: 0.06,
        VELVET: 0.08,
        "default": 0.10,
    }

    # HYDROGEL mean-reversion settings.
    HYDRO_MU = 9995.4
    HYDRO_OU_PULL = 0.10

    ACF1 = {
        HYDROGEL: -0.124,
        VELVET: -0.160,
    }

    # Edge needed before crossing the spread.
    TAKE_EDGE = {
        HYDROGEL: 4.0,
        VELVET: 1.5,
        "VEV_4000": 3.0,
        "VEV_4500": 2.5,
        "VEV_5000": 1.1,
        "VEV_5100": 0.9,
        "VEV_5200": 0.65,
        "VEV_5300": 0.45,
        "VEV_5400": 0.35,
        "VEV_5500": 0.35,
        "VEV_6000": 999.0,
        "VEV_6500": 999.0,
    }

    # Distance used for passive quotes.
    MAKE_EDGE = {
        HYDROGEL: 4.0,
        VELVET: 1.5,
        "VEV_4000": 4.0,
        "VEV_4500": 3.0,
        "VEV_5000": 1.5,
        "VEV_5100": 1.1,
        "VEV_5200": 0.8,
        "VEV_5300": 0.55,
        "VEV_5400": 0.45,
        "VEV_5500": 0.45,
        "VEV_6000": 999.0,
        "VEV_6500": 999.0,
    }

    # Push voucher quote fair lower to favour selling options.
    VOL_SELL_BIAS = {
        "VEV_4000": -7.5,
        "VEV_4500": -5.625,
        "VEV_5000": -4.375,
        "VEV_5100": -5.0,
        "VEV_5200": -4.375,
        "VEV_5300": -4.375,
        "VEV_5400": -2.5,
        "VEV_5500": -1.25,
    }

    # Base size for passive quotes.
    BASE_PASSIVE_SIZE = {
        HYDROGEL: 25,
        VELVET: 8,
        "VEV_4000": 10, "VEV_4500": 10, "VEV_5000": 18,
        "VEV_5100": 20, "VEV_5200": 25, "VEV_5300": 25,
        "VEV_5400": 20, "VEV_5500": 15,
    }

    # Max size for aggressive orders.
    MAX_TAKE_SIZE = {
        HYDROGEL: 24,
        VELVET: 30,
        "VEV_4000": 24, "VEV_4500": 28, "VEV_5000": 44,
        "VEV_5100": 44, "VEV_5200": 42, "VEV_5300": 34,
        "VEV_5400": 24, "VEV_5500": 16,
    }

    # Hedge cap is zero, so VELVET hedging is disabled.
    VELVET_HEDGE_CAP = 0

    # Stronger inventory skew for HYDROGEL.
    HYDRO_INVENTORY_SKEW_MULT = 2.0

    # Stop aggressive HYDROGEL takes near the end.
    EOD_COOLDOWN_TICKS = 1000

    DAY_TIMESTAMPS = 100_000

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        self._update_day_counter(state, data)

        result: Dict[str, List[Order]] = {}
        working_pos = dict(state.position)
        conversions = 0

        mids = self._update_mids_and_ema(state, data)

        velvet_mid = mids.get(self.VELVET)
        if velvet_mid is None:
            return {}, conversions, self._dump_data(data)

        days_passed = float(data.get("day_index", 0)) + state.timestamp / 1_000_000.0
        tte_days = max(0.0, 7.0 - days_passed)
        T = max(1e-6, tte_days / 252.0)

        # Trade vouchers and HYDROGEL first.
        for product, depth in state.order_depths.items():
            if product not in self.LIMITS or product == self.VELVET:
                continue
            if product in self.FLOOR_VOUCHERS:
                result[product] = []
                continue
            best_bid, _, best_ask, _ = self._best_quotes(depth)
            if best_bid is None or best_ask is None:
                continue
            fair = self._fair_value(product, depth, mids, data, velvet_mid, T)
            orders = self._trade_product(
                product, depth, fair, working_pos, state.timestamp,
            )
            if orders:
                result[product] = orders

        # VELVET trading is disabled in this version.
        return result, conversions, self._dump_data(data)

    def _load_data(self, trader_data: str) -> Dict:
        """Load saved trader data."""
        if trader_data:
            try:
                data = json.loads(trader_data)
                if isinstance(data, dict):
                    data.setdefault("ema", {})
                    data.setdefault("prev_mid", {})
                    data.setdefault("last_mid", {})
                    data.setdefault("day_index", 0)
                    data.setdefault("last_timestamp", None)
                    return data
            except Exception:
                pass

        return {
            "ema": {},
            "prev_mid": {},
            "last_mid": {},
            "day_index": 0,
            "last_timestamp": None,
        }

    def _dump_data(self, data: Dict) -> str:
        """Save trader data compactly."""
        return json.dumps(data, separators=(",", ":"))

    def _update_day_counter(self, state: TradingState, data: Dict) -> None:
        """Track when the backtest rolls into a new day."""
        last = data.get("last_timestamp")
        if last is not None and state.timestamp < int(last):
            data["day_index"] = int(data.get("day_index", 0)) + 1
        data["last_timestamp"] = state.timestamp

    def _update_mids_and_ema(
        self, state: TradingState, data: Dict
    ) -> Dict[str, float]:
        """Update mid prices and EMAs."""
        mids: Dict[str, float] = {}

        for product, depth in state.order_depths.items():
            mid = self._vwap_mid(depth)
            if mid is None:
                continue

            mids[product] = mid
            alpha = self.EMA_ALPHA.get(product, self.EMA_ALPHA["default"])
            old_ema = data["ema"].get(product)

            if old_ema is None:
                ema = mid
            else:
                ema = alpha * mid + (1.0 - alpha) * float(old_ema)

            data["ema"][product] = ema
            prev_mid = data.get("last_mid", {}).get(product)
            if prev_mid is not None:
                data["prev_mid"][product] = prev_mid
            data.setdefault("last_mid", {})[product] = mid

        return mids

    def _fair_value(
        self,
        product: str,
        depth: OrderDepth,
        mids: Dict[str, float],
        data: Dict,
        velvet_mid: float,
        T: float,
    ) -> float:
        """Calculate fair value for one product."""
        if product in self.VOUCHERS:
            strike = self.VOUCHERS[product]
            fair = self._bsm_call(velvet_mid, strike, T, self.OPTION_SIGMA)

            # Keep deep ITM calls above intrinsic value.
            intrinsic = max(0.0, velvet_mid - strike)
            if strike <= 5000:
                fair = max(fair, intrinsic + 0.5)
            return fair

        mid = mids.get(product)
        ema = float(data.get("ema", {}).get(product, mid if mid is not None else 0.0))
        obi = self._order_book_imbalance(depth)

        prev = data.get("prev_mid", {}).get(product)
        acf_adj = 0.0
        if prev is not None and mid is not None:
            acf_adj = self.ACF1.get(product, 0.0) * (mid - float(prev))

        if product == self.VELVET:
            return ema + 2.0 * obi + acf_adj
        if product == self.HYDROGEL:
            ou_adj = self.HYDRO_OU_PULL * (self.HYDRO_MU - ema)
            return ema + ou_adj + 3.0 * obi + acf_adj

        return ema + 2.0 * obi

    def _trade_product(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        working_pos: Dict[str, int],
        timestamp: int,
    ) -> List[Order]:
        """Create orders for HYDROGEL or a voucher."""
        best_bid, bid_vol, best_ask, ask_vol = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return []

        limit = self.LIMITS[product]
        pos = working_pos.get(product, 0)
        orders: List[Order] = []

        take_edge = self.TAKE_EDGE.get(product, 1.0)
        make_edge = self.MAKE_EDGE.get(product, 1.5)
        base_size = self.BASE_PASSIVE_SIZE.get(product, 12)
        max_take = self.MAX_TAKE_SIZE.get(product, 24)

        # Skip aggressive HYDROGEL takes near the end.
        eod_cooldown = (
            product == self.HYDROGEL
            and timestamp >= (self.DAY_TIMESTAMPS - self.EOD_COOLDOWN_TICKS)
        )

        if not eod_cooldown:
            # Buy when the ask is cheap enough.
            buy_edge = fair - best_ask
            if buy_edge > take_edge and pos < limit:
                qty = min(
                    ask_vol,
                    limit - pos,
                    max_take,
                    max(1, int(base_size + buy_edge)),
                )
                self._append_order(orders, working_pos, product, best_ask, qty)
                pos = working_pos.get(product, 0)

            # Sell when the bid is high enough.
            sell_edge = best_bid - fair
            if sell_edge > take_edge and pos > -limit:
                qty = min(
                    bid_vol,
                    limit + pos,
                    max_take,
                    max(1, int(base_size + sell_edge)),
                )
                self._append_order(orders, working_pos, product, best_bid, -qty)
                pos = working_pos.get(product, 0)

        # Passive quotes stay active when the spread is wide.
        spread = best_ask - best_bid
        if spread <= 1:
            return orders

        inv = pos / limit if limit > 0 else 0.0

        # Skew quotes away from adding too much inventory.
        skew_mult = (
            self.HYDRO_INVENTORY_SKEW_MULT
            if product == self.HYDROGEL else 2.0
        )
        inventory_skew = inv * make_edge * skew_mult
        vol_bias = self.VOL_SELL_BIAS.get(product, 0.0)
        quote_fair = fair - inventory_skew + vol_bias

        buy_price = int(math.floor(quote_fair - make_edge))
        sell_price = int(math.ceil(quote_fair + make_edge))

        buy_price = min(buy_price, best_bid + 1, best_ask - 1)
        sell_price = max(sell_price, best_ask - 1, best_bid + 1)

        if buy_price > 0 and buy_price < best_ask and pos < limit:
            qty = min(base_size, limit - pos)
            qty = int(qty * max(0.25, 1.0 - max(0.0, inv)))
            if qty > 0:
                self._append_order(orders, working_pos, product, buy_price, qty)
                pos = working_pos.get(product, 0)

        if sell_price > 0 and sell_price > best_bid and pos > -limit:
            qty = min(base_size, limit + pos)
            qty = int(qty * max(0.25, 1.0 + min(0.0, inv)))
            if qty > 0:
                self._append_order(orders, working_pos, product, sell_price, -qty)

        return orders

    def _velvet_hedge_target(
        self,
        working_pos: Dict[str, int],
        velvet_mid: float,
        T: float,
    ) -> int:
        """Return the VELVET target needed to offset option delta."""
        net_call_delta = 0.0
        for product, strike in self.VOUCHERS.items():
            if product in self.FLOOR_VOUCHERS:
                continue
            qty = working_pos.get(product, 0)
            if qty == 0:
                continue
            d = self._bsm_delta(velvet_mid, strike, T, self.OPTION_SIGMA)
            net_call_delta += qty * d

        target = int(round(-net_call_delta))
        cap = self.VELVET_HEDGE_CAP
        return max(-cap, min(cap, target))

    def _trade_velvet_hedge(
        self,
        depth: OrderDepth,
        fair: float,
        working_pos: Dict[str, int],
        target: int,
    ) -> List[Order]:
        """Move VELVET closer to the hedge target."""
        best_bid, bid_vol, best_ask, ask_vol = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return []

        pos = working_pos.get(self.VELVET, 0)
        diff = target - pos
        orders: List[Order] = []

        take_edge = self.TAKE_EDGE[self.VELVET]
        make_edge = self.MAKE_EDGE[self.VELVET]
        base_size = self.BASE_PASSIVE_SIZE[self.VELVET]
        max_take = self.MAX_TAKE_SIZE[self.VELVET]

        # Take only when price and hedge direction both line up.
        if diff > 0:
            buy_edge = fair - best_ask
            if buy_edge > take_edge:
                qty = min(diff, ask_vol, max_take, base_size + 4)
                if qty > 0:
                    self._append_order(orders, working_pos, self.VELVET, best_ask, qty)
                    pos = working_pos.get(self.VELVET, 0)
        elif diff < 0:
            sell_edge = best_bid - fair
            if sell_edge > take_edge:
                qty = min(-diff, bid_vol, max_take, base_size + 4)
                if qty > 0:
                    self._append_order(orders, working_pos, self.VELVET, best_bid, -qty)
                    pos = working_pos.get(self.VELVET, 0)

        # Use passive quotes for smaller hedge moves.
        spread = best_ask - best_bid
        if spread <= 1:
            return orders

        # Skew quote fair slightly toward the hedge target.
        bias = max(-1.0, min(1.0, diff / max(1, self.VELVET_HEDGE_CAP)))
        quote_fair = fair + bias * 0.5

        buy_price = int(math.floor(quote_fair - make_edge))
        sell_price = int(math.ceil(quote_fair + make_edge))
        buy_price = min(buy_price, best_bid + 1, best_ask - 1)
        sell_price = max(sell_price, best_ask - 1, best_bid + 1)

        # Only bid when buying is still acceptable.
        if diff > -base_size and buy_price > 0 and buy_price < best_ask:
            limit_room = self.LIMITS[self.VELVET] - pos
            qty = min(base_size, max(0, limit_room))
            if pos > target:
                qty = int(qty * 0.4)
            if qty > 0:
                self._append_order(orders, working_pos, self.VELVET, buy_price, qty)
                pos = working_pos.get(self.VELVET, 0)

        if diff < base_size and sell_price > 0 and sell_price > best_bid:
            limit_room = self.LIMITS[self.VELVET] + pos
            qty = min(base_size, max(0, limit_room))
            if pos < target:
                qty = int(qty * 0.4)
            if qty > 0:
                self._append_order(orders, working_pos, self.VELVET, sell_price, -qty)

        return orders

    def _append_order(
        self,
        orders: List[Order],
        working_pos: Dict[str, int],
        product: str,
        price: int,
        qty: int,
    ) -> None:
        """Add an order while respecting position limits."""
        if qty == 0:
            return

        limit = self.LIMITS[product]
        current = working_pos.get(product, 0)
        qty = int(qty)

        if current + qty > limit:
            qty = limit - current
        elif current + qty < -limit:
            qty = -limit - current

        if qty == 0:
            return

        orders.append(Order(product, int(price), qty))
        working_pos[product] = current + qty

    def _best_quotes(
        self, depth: OrderDepth
    ) -> Tuple[Optional[int], int, Optional[int], int]:
        """Return best bid, bid size, best ask, and ask size."""
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        bid_vol = depth.buy_orders[best_bid] if best_bid is not None else 0
        ask_vol = -depth.sell_orders[best_ask] if best_ask is not None else 0
        return best_bid, bid_vol, best_ask, ask_vol

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """Return the simple midpoint."""
        best_bid, _, best_ask, _ = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return None
        return 0.5 * (best_bid + best_ask)

    def _vwap_mid(self, depth: OrderDepth) -> Optional[float]:
        """Return a volume-weighted midpoint from visible book levels."""
        if not depth.buy_orders or not depth.sell_orders:
            return self._mid(depth)

        bid_denom = sum(abs(v) for v in depth.buy_orders.values())
        ask_denom = sum(abs(v) for v in depth.sell_orders.values())
        if bid_denom == 0 or ask_denom == 0:
            return self._mid(depth)

        bid_vwap = sum(price * abs(volume) for price, volume in depth.buy_orders.items()) / bid_denom
        ask_vwap = sum(price * abs(volume) for price, volume in depth.sell_orders.items()) / ask_denom

        return 0.5 * (bid_vwap + ask_vwap)

    def _order_book_imbalance(self, depth: OrderDepth) -> float:
        """Return total order book imbalance."""
        bid_vol = sum(depth.buy_orders.values()) if depth.buy_orders else 0
        ask_vol = sum(abs(v) for v in depth.sell_orders.values()) if depth.sell_orders else 0
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _bsm_call(self, S: float, K: float, T: float, sigma: float) -> float:
        """Return the Black-Scholes call price."""
        if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
            return max(0.0, S - K)

        total_vol = sigma * math.sqrt(T)
        if total_vol <= 1e-9:
            return max(0.0, S - K)

        d1 = (math.log(S / K) + 0.5 * total_vol * total_vol) / total_vol
        d2 = d1 - total_vol
        return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

    def _bsm_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """Return the Black-Scholes call delta."""
        if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        total_vol = sigma * math.sqrt(T)
        if total_vol <= 1e-9:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * total_vol * total_vol) / total_vol
        return self._norm_cdf(d1)

    def _norm_cdf(self, x: float) -> float:
        """Return the standard normal CDF."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
