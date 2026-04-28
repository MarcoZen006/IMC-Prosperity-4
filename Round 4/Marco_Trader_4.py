from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    """
    Trader 4 — vol-aware market maker with delta-hedged VELVET.

    Edge model (unchanged from T3 in spirit):
      Sell OTM VEV vouchers because realized VELVET vol (~11%) is far below
      market-implied vol (~22%). BSM with a low sigma -> fair ~ intrinsic ->
      every OTM call looks overpriced -> sell.

    Changes from Trader 3:
      1. VELVET no longer trades on its own MM signal.
         It's now driven only by a delta-hedge target against current short
         option positions, capped at +-80 (down from +-200). This addresses
         the -5,354 VELVET loss in 490207, which came from the standalone
         take-edge logic pinning VELVET at +200 from ts=19,200 onwards.

      2. HYDROGEL has an end-of-session cooldown.
         No aggressive cross-the-spread takes in the last 1,000 ticks.
         Only passive quotes. This prevents the ts=99,000-99,400 spree that
         bought 65 contracts at 10,025-10,029 right before the close at 10,017.

      3. Bidirectional voucher take-logic.
         Old code only crossed asks (buy at ask) when fair was high; sells
         only via passive maker quotes. Now also crosses bids (sell to bid)
         when fair > best_bid + edge AND we have inventory to sell. This
         is mostly defensive (capture profit faster, allow exit).

      4. Removed the mark-counterparty learning subsystem.
         Lag was 5,000 ticks vs ~1,000 timestamps/day with ~50 mature samples
         per session. Effectively noise. Removing it simplifies the file and
         reduces traderData size.

      5. Removed ema_ret directional bias on HYDROGEL/VELVET fair value.
         This was injecting a momentum bet that compounded the VELVET
         long-bias problem. Fair = EMA + OBI weight only.

      6. HYDROGEL inventory skew bumped from 2.0x to 2.5x make_edge.
         Pushes us out of inventory faster.
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

    # 0.5 floor instruments — never edge available
    FLOOR_VOUCHERS = {"VEV_6000", "VEV_6500"}

    # BSM sigma for fair-value calculation. The strategy works for any small
    # sigma here because the edge comes from realized vs implied gap, not
    # from the absolute level. 0.10 gives nontrivial time-value so the
    # buy-back leg works for ATM-ish strikes if vol cools mid-day.
    OPTION_SIGMA = 0.10

    EMA_ALPHA = {
        HYDROGEL: 0.06,
        VELVET: 0.08,
        "default": 0.10,
    }

    # HYDROGEL mean-reversion / short-term reversal model from Trader 13.
    # This only changes HYDROGEL fair value; voucher logic and VELVET hedge
    # settings are left unchanged so this can be tested in isolation.
    HYDRO_MU = 9995.4
    # Trader 25: drop from 0.15 -> 0.10 to test weaker mean-reversion pull
    # (Trader 24 confirmed 0.20 is too strong, -200 PNL).
    HYDRO_OU_PULL = 0.10
    ACF1 = {
        HYDROGEL: -0.124,
        VELVET: -0.160,
    }

    # Take edge (cross-the-spread threshold)
    TAKE_EDGE = {
        HYDROGEL: 4.0,
        VELVET: 1.5,         # used by hedge-only logic
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

    # Make edge (passive quote distance)
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


    # Passive option quote bias from Trader 13.
    # Negative values push option quote_fair lower:
    #   - passive bids become less likely to fill
    #   - passive asks move closer to ask-1 and are more likely to fill
    # Trader 29: scale all entries to 125% of Trader 25 baseline to test
    # whether stronger passive short-vol bias adds more PNL.
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

    # Larger passive sizes from Trader 13.
    # This keeps the same fair-value / quoting logic, but increases maker
    # participation where the previous tests showed useful edge.
    BASE_PASSIVE_SIZE = {
        HYDROGEL: 15,
        VELVET: 8,           # VELVET remains hedge-driven
        "VEV_4000": 10, "VEV_4500": 10, "VEV_5000": 18,
        "VEV_5100": 20, "VEV_5200": 25, "VEV_5300": 25,
        "VEV_5400": 20, "VEV_5500": 15,
    }

    MAX_TAKE_SIZE = {
        HYDROGEL: 24,
        VELVET: 30,
        "VEV_4000": 24, "VEV_4500": 28, "VEV_5000": 44,
        "VEV_5100": 44, "VEV_5200": 42, "VEV_5300": 34,
        "VEV_5400": 24, "VEV_5500": 16,
    }

    # Hedge target cap. With 5 vouchers x -300 max short, total short delta
    # at ATM-ish ~2.5, so net delta ~ -750. Limit is +200 on VELVET. Even
    # full hedging would be partial. Capping at +-80 keeps the hedge cost
    # bounded in trending sessions while still providing some directional
    # protection against a vol spike.
    # Trader 32: drop hedge cap from 80 -> 70.
    # Prior tests: 80->120 was -100, 80->180 was -2000 (more hedge worse).
    # Testing if even less hedging helps.
    VELVET_HEDGE_CAP = 70

    HYDRO_INVENTORY_SKEW_MULT = 2.5

    # In last X ticks of session, no aggressive crossing on HYDROGEL.
    # Prevents the kind of late-day adverse selection that cost ~600 in
    # 490207 (ts=99,000-99,400 buying spree).
    EOD_COOLDOWN_TICKS = 1000

    DAY_TIMESTAMPS = 100_000  # one day spans timestamps 0 -> 99,900

    # ---- main loop ------------------------------------------------------

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

        # 1) Trade vouchers and HYDROGEL first (these set the option positions).
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

        # 2) Now compute hedge target for VELVET against the resulting
        #    voucher position (using working_pos which reflects new orders).
        velvet_depth = state.order_depths.get(self.VELVET)
        if velvet_depth is not None:
            best_bid, _, best_ask, _ = self._best_quotes(velvet_depth)
            if best_bid is not None and best_ask is not None:
                target = self._velvet_hedge_target(
                    working_pos, velvet_mid, T,
                )
                fair = self._fair_value(
                    self.VELVET, velvet_depth, mids, data, velvet_mid, T,
                )
                orders = self._trade_velvet_hedge(
                    velvet_depth, fair, working_pos, target,
                )
                if orders:
                    result[self.VELVET] = orders

        return result, conversions, self._dump_data(data)

    # ---- state ----------------------------------------------------------

    def _load_data(self, trader_data: str) -> Dict:
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
        return json.dumps(data, separators=(",", ":"))

    def _update_day_counter(self, state: TradingState, data: Dict) -> None:
        last = data.get("last_timestamp")
        if last is not None and state.timestamp < int(last):
            data["day_index"] = int(data.get("day_index", 0)) + 1
        data["last_timestamp"] = state.timestamp

    def _update_mids_and_ema(
        self, state: TradingState, data: Dict
    ) -> Dict[str, float]:
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

    # ---- fair value -----------------------------------------------------

    def _fair_value(
        self,
        product: str,
        depth: OrderDepth,
        mids: Dict[str, float],
        data: Dict,
        velvet_mid: float,
        T: float,
    ) -> float:
        if product in self.VOUCHERS:
            strike = self.VOUCHERS[product]
            fair = self._bsm_call(velvet_mid, strike, T, self.OPTION_SIGMA)
            # Floor for deep ITM
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

    # ---- voucher / HYDROGEL trading ------------------------------------

    def _trade_product(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        working_pos: Dict[str, int],
        timestamp: int,
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

        # End-of-session cooldown for HYDROGEL: skip aggressive takes only.
        # Passive quoting still runs.
        eod_cooldown = (
            product == self.HYDROGEL
            and timestamp >= (self.DAY_TIMESTAMPS - self.EOD_COOLDOWN_TICKS)
        )

        if not eod_cooldown:
            # Buy: cross ask if fair >> ask
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

            # Sell: cross bid if fair << bid
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

        # Passive quoting always runs (this is how we enter positions
        # cheaply when the spread is wide).
        spread = best_ask - best_bid
        if spread <= 1:
            return orders

        inv = pos / limit if limit > 0 else 0.0

        # Stronger inventory skew on HYDROGEL specifically.
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

    # ---- VELVET hedge logic --------------------------------------------

    def _velvet_hedge_target(
        self,
        working_pos: Dict[str, int],
        velvet_mid: float,
        T: float,
    ) -> int:
        """
        Compute desired VELVET position to hedge net option delta.

        net_call_position = sum_k (position_k * delta_k)
          if positive, we are long calls -> long delta -> hedge by short VELVET
          if negative, we are short calls -> short delta -> hedge by long VELVET

        Capped at +-VELVET_HEDGE_CAP (80).
        """
        net_call_delta = 0.0
        for product, strike in self.VOUCHERS.items():
            if product in self.FLOOR_VOUCHERS:
                continue
            qty = working_pos.get(product, 0)
            if qty == 0:
                continue
            d = self._bsm_delta(velvet_mid, strike, T, self.OPTION_SIGMA)
            net_call_delta += qty * d

        # Hedge = -net_call_delta (offset)
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
        """
        Move VELVET toward hedge target. Use take-orders aggressively only
        when the move is in our direction AND the price is favorable. Use
        passive quotes for the rest.
        """
        best_bid, bid_vol, best_ask, ask_vol = self._best_quotes(depth)
        if best_bid is None or best_ask is None:
            return []

        pos = working_pos.get(self.VELVET, 0)
        diff = target - pos  # positive = need to buy, negative = need to sell
        orders: List[Order] = []

        take_edge = self.TAKE_EDGE[self.VELVET]
        make_edge = self.MAKE_EDGE[self.VELVET]
        base_size = self.BASE_PASSIVE_SIZE[self.VELVET]
        max_take = self.MAX_TAKE_SIZE[self.VELVET]

        # Aggressive take only if (a) book is favorable AND (b) move aligns
        # with hedge target.
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

        # Passive quotes biased toward target.
        spread = best_ask - best_bid
        if spread <= 1:
            return orders

        # Pull bid up / push ask up if we want to BUY (diff > 0)
        # Pull ask down / push bid down if we want to SELL (diff < 0)
        bias = max(-1.0, min(1.0, diff / max(1, self.VELVET_HEDGE_CAP)))
        quote_fair = fair + bias * 0.5  # small skew toward target

        buy_price = int(math.floor(quote_fair - make_edge))
        sell_price = int(math.ceil(quote_fair + make_edge))
        buy_price = min(buy_price, best_bid + 1, best_ask - 1)
        sell_price = max(sell_price, best_ask - 1, best_bid + 1)

        # Only place a passive bid if we want to (or are willing to) be long.
        if diff > -base_size and buy_price > 0 and buy_price < best_ask:
            limit_room = self.LIMITS[self.VELVET] - pos
            qty = min(base_size, max(0, limit_room))
            # Reduce size if already past target on the long side
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

    # ---- low-level helpers ---------------------------------------------

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

    def _best_quotes(
        self, depth: OrderDepth
    ) -> Tuple[Optional[int], int, Optional[int], int]:
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

    def _vwap_mid(self, depth: OrderDepth) -> Optional[float]:
        """
        Volume-weighted midpoint using all visible bid and ask levels.

        This replaces the simple best bid/ask midpoint in the EMA update.
        It is still non-hardcoded: it only uses current order-book prices
        and displayed volumes.
        """
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
        bid_vol = sum(depth.buy_orders.values()) if depth.buy_orders else 0
        ask_vol = sum(abs(v) for v in depth.sell_orders.values()) if depth.sell_orders else 0
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _bsm_call(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
            return max(0.0, S - K)

        total_vol = sigma * math.sqrt(T)
        if total_vol <= 1e-9:
            return max(0.0, S - K)

        d1 = (math.log(S / K) + 0.5 * total_vol * total_vol) / total_vol
        d2 = d1 - total_vol
        return S * self._norm_cdf(d1) - K * self._norm_cdf(d2)

    def _bsm_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        """N(d1) — call delta in BSM."""
        if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
            return 1.0 if S > K else 0.0
        total_vol = sigma * math.sqrt(T)
        if total_vol <= 1e-9:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S / K) + 0.5 * total_vol * total_vol) / total_vol
        return self._norm_cdf(d1)

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
