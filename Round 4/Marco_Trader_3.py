from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
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

    FLOOR_VOUCHERS = {"VEV_6000", "VEV_6500"}

    # Keep the profitable short-vol bias from Trader 1, but make it explicit.
    # Test grid around: 0.04, 0.05, 0.06, 0.08, 0.10.
    OPTION_SIGMA = 0.05

    # EMA alpha can be tuned. 0.10 has half-life about 6.6 ticks.
    EMA_ALPHA = {
        HYDROGEL: 0.06,
        VELVET: 0.08,
        "default": 0.10,
    }

    # Product-specific trading filters. These replace Trader 1's single 1.5 edge.
    TAKE_EDGE = {
        HYDROGEL: 4.0,
        VELVET: 1.2,
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

    BASE_PASSIVE_SIZE = {
        HYDROGEL: 10,
        VELVET: 18,
        "VEV_4000": 10,
        "VEV_4500": 12,
        "VEV_5000": 18,
        "VEV_5100": 18,
        "VEV_5200": 18,
        "VEV_5300": 16,
        "VEV_5400": 12,
        "VEV_5500": 8,
    }

    MAX_TAKE_SIZE = {
        HYDROGEL: 24,
        VELVET: 40,
        "VEV_4000": 24,
        "VEV_4500": 28,
        "VEV_5000": 44,
        "VEV_5100": 44,
        "VEV_5200": 42,
        "VEV_5300": 34,
        "VEV_5400": 24,
        "VEV_5500": 16,
    }

    MARK_LEARN_LAG = 5000
    MARK_LEARN_RATE = 0.015
    MAX_PENDING_MARK_TRADES = 250

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        self._update_day_counter(state, data)

        result: Dict[str, List[Order]] = {}
        working_pos = dict(state.position)
        conversions = 0

        mids = self._update_mids_and_ema(state, data)
        self._learn_mark_scores(state, data, mids)

        velvet_mid = mids.get(self.VELVET)
        if velvet_mid is None:
            return {}, conversions, self._dump_data(data)

        days_passed = float(data.get("day_index", 0)) + state.timestamp / 1_000_000.0
        tte_days = max(0.0, 7.0 - days_passed)
        T = max(1e-6, tte_days / 252.0)

        for product, depth in state.order_depths.items():
            if product not in self.LIMITS:
                continue

            if product in self.FLOOR_VOUCHERS:
                # These are flat/tick-floor instruments in the supplied data.
                # Avoid wasting inventory or placing invalid negative bids.
                result[product] = []
                continue

            best_bid, bid_vol, best_ask, ask_vol = self._best_quotes(depth)
            if best_bid is None or best_ask is None:
                continue

            fair = self._fair_value(product, depth, mids, data, velvet_mid, T)
            orders = self._trade_product(
                product=product,
                depth=depth,
                fair=fair,
                working_pos=working_pos,
            )

            if orders:
                result[product] = orders

        return result, conversions, self._dump_data(data)

    def _load_data(self, trader_data: str) -> Dict:
        if trader_data:
            try:
                data = json.loads(trader_data)
                if isinstance(data, dict):
                    data.setdefault("ema", {})
                    data.setdefault("prev_mid", {})
                    data.setdefault("ema_ret", {})
                    data.setdefault("mark_score", {})
                    data.setdefault("pending_mark", [])
                    data.setdefault("day_index", 0)
                    data.setdefault("last_timestamp", None)
                    return data
            except Exception:
                pass

        return {
            "ema": {},
            "prev_mid": {},
            "ema_ret": {},
            "mark_score": {},
            "pending_mark": [],
            "day_index": 0,
            "last_timestamp": None,
        }

    def _dump_data(self, data: Dict) -> str:
        # Keep traderData small enough for the sandbox.
        pending = data.get("pending_mark", [])
        if len(pending) > self.MAX_PENDING_MARK_TRADES:
            data["pending_mark"] = pending[-self.MAX_PENDING_MARK_TRADES:]
        return json.dumps(data, separators=(",", ":"))

    def _update_day_counter(self, state: TradingState, data: Dict) -> None:
        last_timestamp = data.get("last_timestamp")
        if last_timestamp is not None and state.timestamp < int(last_timestamp):
            data["day_index"] = int(data.get("day_index", 0)) + 1
            data["pending_mark"] = []
        data["last_timestamp"] = state.timestamp

    def _update_mids_and_ema(self, state: TradingState, data: Dict) -> Dict[str, float]:
        mids: Dict[str, float] = {}

        for product, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is None:
                continue

            mids[product] = mid
            alpha = self.EMA_ALPHA.get(product, self.EMA_ALPHA["default"])
            old_ema = data["ema"].get(product)

            if old_ema is None:
                ema = mid
            else:
                ema = alpha * mid + (1.0 - alpha) * float(old_ema)

            prev_mid = data["prev_mid"].get(product)
            if prev_mid is None:
                ret = 0.0
            else:
                ret = mid - float(prev_mid)

            old_ret = float(data["ema_ret"].get(product, 0.0))
            data["ema_ret"][product] = 0.85 * old_ret + 0.15 * ret
            data["ema"][product] = ema
            data["prev_mid"][product] = mid

        return mids

    def _learn_mark_scores(
        self,
        state: TradingState,
        data: Dict,
        mids: Dict[str, float],
    ) -> None:
        now_abs = int(data.get("day_index", 0)) * 1_000_000 + state.timestamp
        new_pending = []
        mark_score = data.setdefault("mark_score", {})

        for item in data.get("pending_mark", []):
            try:
                entry_time, product, buyer, seller, entry_mid, qty = item
            except ValueError:
                continue

            if now_abs - int(entry_time) < self.MARK_LEARN_LAG:
                new_pending.append(item)
                continue

            current_mid = mids.get(product)
            if current_mid is None:
                continue

            move = float(current_mid) - float(entry_mid)
            scaled_move = self._clip(move / max(1.0, float(entry_mid) * 0.001), -3.0, 3.0)
            update = self.MARK_LEARN_RATE * float(qty) * scaled_move

            if buyer:
                key = product + "|" + str(buyer)
                mark_score[key] = self._clip(float(mark_score.get(key, 0.0)) + update, -5.0, 5.0)
            if seller:
                key = product + "|" + str(seller)
                mark_score[key] = self._clip(float(mark_score.get(key, 0.0)) - update, -5.0, 5.0)

        data["pending_mark"] = new_pending[-self.MAX_PENDING_MARK_TRADES:]

        for product, trades in state.market_trades.items():
            if product not in mids:
                continue
            for trade in trades:
                buyer = getattr(trade, "buyer", None)
                seller = getattr(trade, "seller", None)
                qty = int(getattr(trade, "quantity", 0))
                if qty <= 0:
                    continue
                data["pending_mark"].append(
                    [now_abs, product, buyer, seller, float(mids[product]), qty]
                )

    def _fair_value(
        self,
        product: str,
        depth: OrderDepth,
        mids: Dict[str, float],
        data: Dict,
        velvet_mid: float,
        T: float,
    ) -> float:
        mark_adjust = self._mark_adjustment(product, data)

        if product in self.VOUCHERS:
            strike = self.VOUCHERS[product]
            fair = self._bsm_call(velvet_mid, strike, T, self.OPTION_SIGMA)
            fair += mark_adjust * 0.25

            # For deep ITM calls, keep the fair close to intrinsic plus a small time value.
            intrinsic = max(0.0, velvet_mid - strike)
            if strike <= 5000:
                fair = max(fair, intrinsic + 0.5)

            return fair

        mid = mids.get(product)
        ema = float(data.get("ema", {}).get(product, mid if mid is not None else 0.0))
        obi = self._order_book_imbalance(depth)
        ema_ret = float(data.get("ema_ret", {}).get(product, 0.0))

        if product == self.VELVET:
            return ema + 2.0 * obi + 0.35 * ema_ret + mark_adjust * 0.35

        if product == self.HYDROGEL:
            return ema + 3.0 * obi + 0.20 * ema_ret + mark_adjust * 0.20

        return ema + 2.0 * obi

    def _mark_adjustment(self, product: str, data: Dict) -> float:
        trades_signal = 0.0
        mark_score = data.get("mark_score", {})

        # Use only the latest tape, but with learned Mark quality.
        # This avoids the Trader 1 issue where all Mark-vs-Mark trades cancel to zero.
        # The caller keeps this small, so it cannot dominate the base fair model.
        for item in data.get("pending_mark", [])[-40:]:
            try:
                _, p, buyer, seller, _, qty = item
            except ValueError:
                continue
            if p != product:
                continue
            buy_key = product + "|" + str(buyer)
            sell_key = product + "|" + str(seller)
            trades_signal += (float(mark_score.get(buy_key, 0.0)) - float(mark_score.get(sell_key, 0.0))) * float(qty)

        return self._clip(0.015 * trades_signal, -4.0, 4.0)

    def _trade_product(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        working_pos: Dict[str, int],
    ) -> List[Order]:
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

        buy_edge = fair - best_ask
        if buy_edge > take_edge and pos < limit:
            qty = min(ask_vol, limit - pos, max_take, max(1, int(base_size + buy_edge)))
            self._append_order(orders, working_pos, product, best_ask, qty)
            pos = working_pos.get(product, 0)

        sell_edge = best_bid - fair
        if sell_edge > take_edge and pos > -limit:
            qty = min(bid_vol, limit + pos, max_take, max(1, int(base_size + sell_edge)))
            self._append_order(orders, working_pos, product, best_bid, -qty)
            pos = working_pos.get(product, 0)

        # Passive quoting, with inventory skew and no crossing.
        spread = best_ask - best_bid
        if spread <= 1:
            return orders

        inv = pos / limit if limit > 0 else 0.0
        inventory_skew = inv * make_edge * 2.0
        quote_fair = fair - inventory_skew

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

    def _append_order(
        self,
        orders: List[Order],
        working_pos: Dict[str, int],
        product: str,
        price: int,
        qty: int,
    ) -> None:
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

    def _best_quotes(self, depth: OrderDepth) -> Tuple[Optional[int], int, Optional[int], int]:
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        bid_vol = depth.buy_orders[best_bid] if best_bid is not None else 0
        ask_vol = -depth.sell_orders[best_ask] if best_ask is not None else 0
        return best_bid, bid_vol, best_ask, ask_vol

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        best_bid, _, best_ask, _ = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return None
        return 0.5 * (best_bid + best_ask)

    def _order_book_imbalance(self, depth: OrderDepth) -> float:
        bid_vol = sum(depth.buy_orders.values()) if depth.buy_orders else 0
        ask_vol = sum(abs(v) for v in depth.sell_orders.values()) if depth.sell_orders else 0
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _bsm_call(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0:
            return max(0.0, S - K)
        if S <= 0 or K <= 0 or sigma <= 0:
            return max(0.0, S - K)

        total_vol = sigma * math.sqrt(T)
        if total_vol <= 1e-9:
            return max(0.0, S - K)

        d1 = (math.log(S / K) + 0.5 * total_vol * total_vol) / total_vol
        d2 = d1 - total_vol
        return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _clip(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
