from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState

OSM = "ASH_COATED_OSMIUM"
PEP = "INTARIAN_PEPPER_ROOT"
POS_LIMIT = {OSM: 80, PEP: 80}

# Osmium settings.
OSM_FAIR = 10000
OSM_MICRO_WEIGHT = 0.34
OSM_BASE_TAKE_EDGE = 0.50
OSM_JUMP_TAKE_EDGE = 0.25
OSM_BASE_MAKE_EDGE_1 = 4
OSM_BASE_MAKE_EDGE_2 = 5
OSM_JUMP_MAKE_EDGE_1 = 3
OSM_JUMP_MAKE_EDGE_2 = 5
OSM_MM_SIZE_1 = 10
OSM_MM_SIZE_2 = 30
OSM_JUMP_MM_SIZE_1 = 15
OSM_POS_SKEW = 0.04
OSM_RET_REV = 0.12
OSM_JUMP_THRESHOLD = 3.0

# Osmium inventory control.
OSM_INV_WEAK_PRESSURE = 0.20
OSM_INV_BAND_Q1 = 20
OSM_INV_BAND_Q2 = 45
OSM_INV_CAUTION_SAME_SIDE_TAKE_PENALTY = 0.15
OSM_INV_DANGER_SAME_SIDE_TAKE_PENALTY = 0.40
OSM_INV_DANGER_OPPOSITE_TAKE_BONUS = 0.25

# Hold-mode settings for Osmium.
OSM_HOLD_PRESSURE_ON = 0.18
OSM_HOLD_PRESSURE_OFF = 0.08
OSM_HOLD_SLOPE_ON = 0.08
OSM_HOLD_SLOPE_OFF = 0.03
OSM_HOLD_UNWIND_PENALTY = 0.18
OSM_HOLD_UNWIND_QUOTE_WIDEN = 1
OSM_HOLD_FLIP_UNWIND_BONUS = 0.20
OSM_HOLD_FLIP_UNWIND_QUOTE_TIGHTEN = 1
OSM_LARGE_UNWIND_QUOTE_TIGHTEN = 1

# Pepper settings.
PEP_PRIOR_DRIFT = 0.10
PEP_HOLD_HORIZON = 80
PEP_MAKE_EDGE_BID = 1
PEP_MAKE_EDGE_ASK = 5
PEP_MM_SIZE = 40
PEP_TAKE_EDGE = 0
PEP_ADAPT_MIN_OBS = 300
PEP_ADAPT_WINDOW = 400
PEP_ADAPT_BLEND = 0.5


class Trader:
    def bid(self) -> int:
        return 5000

    def __init__(self):
        self._pep_mid_history: List[float] = []
        self._osm_last_mid: float | None = None
        self._osm_last_micro: float | None = None
        self._osm_long_hold_mode = False
        self._osm_short_hold_mode = False

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        if OSM in state.order_depths:
            pos = state.position.get(OSM, 0)
            result[OSM] = self._trade_osmium(state.order_depths[OSM], pos)

        if PEP in state.order_depths:
            pos = state.position.get(PEP, 0)
            result[PEP] = self._trade_pepper(state.order_depths[PEP], pos)

        return result, 0, ""

    @staticmethod
    def _top_of_book(od: OrderDepth):
        if not od.buy_orders or not od.sell_orders:
            return None
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bbv = od.buy_orders[best_bid]
        bav = -od.sell_orders[best_ask]
        if bbv <= 0 or bav <= 0:
            return None
        return best_bid, bbv, best_ask, bav

    @staticmethod
    def _two_level_microprice(od: OrderDepth):
        bids = [(p, v) for p, v in sorted(od.buy_orders.items(), key=lambda x: -x[0])[:2] if v > 0]
        asks = [(p, -v) for p, v in sorted(od.sell_orders.items(), key=lambda x: x[0])[:2] if -v > 0]
        if not bids or not asks:
            return None

        bid_vol = sum(v for _, v in bids)
        ask_vol = sum(v for _, v in asks)
        if bid_vol <= 0 or ask_vol <= 0:
            return None

        bid_vwap = sum(p * v for p, v in bids) / bid_vol
        ask_vwap = sum(p * v for p, v in asks) / ask_vol

        return (bid_vol * ask_vwap + ask_vol * bid_vwap) / (bid_vol + ask_vol)

    @staticmethod
    def _three_level_microprice(od: OrderDepth):
        bids = [(p, v) for p, v in sorted(od.buy_orders.items(), key=lambda x: -x[0])[:3] if v > 0]
        asks = [(p, -v) for p, v in sorted(od.sell_orders.items(), key=lambda x: x[0])[:3] if -v > 0]
        if not bids or not asks:
            return None

        bid_vol = sum(v for _, v in bids)
        ask_vol = sum(v for _, v in asks)
        if bid_vol <= 0 or ask_vol <= 0:
            return None

        bid_vwap = sum(p * v for p, v in bids) / bid_vol
        ask_vwap = sum(p * v for p, v in asks) / ask_vol

        return (bid_vol * ask_vwap + ask_vol * bid_vwap) / (bid_vol + ask_vol)

    @staticmethod
    def _sorted_asks(od: OrderDepth):
        return sorted(((p, -v) for p, v in od.sell_orders.items()), key=lambda x: x[0])

    @staticmethod
    def _sorted_bids(od: OrderDepth):
        return sorted(od.buy_orders.items(), key=lambda x: -x[0])

    def _update_osmium_hold_modes(self, position: int, pressure: float, micro_slope: float):
        prev_long_hold = self._osm_long_hold_mode
        prev_short_hold = self._osm_short_hold_mode

        if position > 0:
            self._osm_short_hold_mode = False
            if self._osm_long_hold_mode:
                if pressure < OSM_HOLD_PRESSURE_OFF or micro_slope < OSM_HOLD_SLOPE_OFF:
                    self._osm_long_hold_mode = False
            elif pressure > OSM_HOLD_PRESSURE_ON and micro_slope > OSM_HOLD_SLOPE_ON:
                self._osm_long_hold_mode = True
        elif position < 0:
            self._osm_long_hold_mode = False
            if self._osm_short_hold_mode:
                if pressure > -OSM_HOLD_PRESSURE_OFF or micro_slope > -OSM_HOLD_SLOPE_OFF:
                    self._osm_short_hold_mode = False
            elif pressure < -OSM_HOLD_PRESSURE_ON and micro_slope < -OSM_HOLD_SLOPE_ON:
                self._osm_short_hold_mode = True
        else:
            self._osm_long_hold_mode = False
            self._osm_short_hold_mode = False

        long_lost_support = prev_long_hold and not self._osm_long_hold_mode
        short_lost_support = prev_short_hold and not self._osm_short_hold_mode

        return self._osm_long_hold_mode, self._osm_short_hold_mode, long_lost_support, short_lost_support

    def _trade_osmium(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        tob = self._top_of_book(od)
        if tob is None:
            return orders
        best_bid, bbv, best_ask, bav = tob

        micro = self._three_level_microprice(od)
        if micro is None:
            micro = self._two_level_microprice(od)
            if micro is None:
                micro = (bbv * best_ask + bav * best_bid) / (bbv + bav)
        mid = (best_bid + best_ask) / 2.0
        last_move = 0.0 if self._osm_last_mid is None else (mid - self._osm_last_mid)
        micro_slope = 0.0 if self._osm_last_micro is None else (micro - self._osm_last_micro)

        fair = (1 - OSM_MICRO_WEIGHT) * OSM_FAIR + OSM_MICRO_WEIGHT * micro - OSM_RET_REV * last_move

        if abs(last_move) >= OSM_JUMP_THRESHOLD:
            take_edge = OSM_JUMP_TAKE_EDGE
            make_edge_1 = OSM_JUMP_MAKE_EDGE_1
            make_edge_2 = OSM_JUMP_MAKE_EDGE_2
            mm_size_1 = OSM_JUMP_MM_SIZE_1
        else:
            take_edge = OSM_BASE_TAKE_EDGE
            make_edge_1 = OSM_BASE_MAKE_EDGE_1
            make_edge_2 = OSM_BASE_MAKE_EDGE_2
            mm_size_1 = OSM_MM_SIZE_1

        limit = POS_LIMIT[OSM]
        buy_cap = limit - position
        sell_cap = limit + position

        pressure = micro - mid
        abs_pos = abs(position)

        buy_take_edge = take_edge
        sell_take_edge = take_edge
        allow_inner_bid = True
        allow_outer_bid = True
        allow_inner_ask = True
        allow_outer_ask = True
        ask_quote_shift = 0
        bid_quote_shift = 0

        hold_small = abs_pos < OSM_INV_BAND_Q1
        hold_medium = OSM_INV_BAND_Q1 <= abs_pos <= OSM_INV_BAND_Q2
        hold_large = abs_pos > OSM_INV_BAND_Q2

        long_hold_mode, short_hold_mode, long_lost_support, short_lost_support = self._update_osmium_hold_modes(
            position, pressure, micro_slope
        )

        # Do not hold large inventory.
        if hold_large:
            long_hold_mode = False
            short_hold_mode = False
            long_lost_support = False
            short_lost_support = False

        if position > 0:
            if OSM_INV_BAND_Q1 < abs_pos <= OSM_INV_BAND_Q2:
                allow_inner_bid = False
                if pressure < OSM_INV_WEAK_PRESSURE:
                    buy_take_edge += OSM_INV_CAUTION_SAME_SIDE_TAKE_PENALTY
            elif abs_pos > OSM_INV_BAND_Q2:
                allow_inner_bid = False
                allow_outer_bid = False
                if pressure < OSM_INV_WEAK_PRESSURE:
                    buy_take_edge += OSM_INV_DANGER_SAME_SIDE_TAKE_PENALTY
                sell_take_edge = max(0.0, sell_take_edge - OSM_INV_DANGER_OPPOSITE_TAKE_BONUS)

            if hold_medium:
                if long_hold_mode:
                    sell_take_edge += OSM_HOLD_UNWIND_PENALTY
                    ask_quote_shift += OSM_HOLD_UNWIND_QUOTE_WIDEN
                elif long_lost_support:
                    sell_take_edge = max(0.0, sell_take_edge - OSM_HOLD_FLIP_UNWIND_BONUS)
                    ask_quote_shift -= OSM_HOLD_FLIP_UNWIND_QUOTE_TIGHTEN
            elif hold_large:
                ask_quote_shift -= OSM_LARGE_UNWIND_QUOTE_TIGHTEN

        elif position < 0:
            if OSM_INV_BAND_Q1 < abs_pos <= OSM_INV_BAND_Q2:
                allow_inner_ask = False
                if pressure > -OSM_INV_WEAK_PRESSURE:
                    sell_take_edge += OSM_INV_CAUTION_SAME_SIDE_TAKE_PENALTY
            elif abs_pos > OSM_INV_BAND_Q2:
                allow_inner_ask = False
                allow_outer_ask = False
                if pressure > -OSM_INV_WEAK_PRESSURE:
                    sell_take_edge += OSM_INV_DANGER_SAME_SIDE_TAKE_PENALTY
                buy_take_edge = max(0.0, buy_take_edge - OSM_INV_DANGER_OPPOSITE_TAKE_BONUS)

            if hold_medium:
                if short_hold_mode:
                    buy_take_edge += OSM_HOLD_UNWIND_PENALTY
                    bid_quote_shift -= OSM_HOLD_UNWIND_QUOTE_WIDEN
                elif short_lost_support:
                    buy_take_edge = max(0.0, buy_take_edge - OSM_HOLD_FLIP_UNWIND_BONUS)
                    bid_quote_shift += OSM_HOLD_FLIP_UNWIND_QUOTE_TIGHTEN
            elif hold_large:
                bid_quote_shift += OSM_LARGE_UNWIND_QUOTE_TIGHTEN

        for p, v in self._sorted_asks(od):
            if buy_cap <= 0:
                break
            if p < fair - buy_take_edge + 1e-6:
                q = min(v, buy_cap)
                if q > 0:
                    orders.append(Order(OSM, p, q))
                    buy_cap -= q
                    position += q
            else:
                break

        for p, v in self._sorted_bids(od):
            if sell_cap <= 0:
                break
            if p > fair + sell_take_edge - 1e-6:
                q = min(v, sell_cap)
                if q > 0:
                    orders.append(Order(OSM, p, -q))
                    sell_cap -= q
                    position -= q
            else:
                break

        inner_bid = int(round(fair - make_edge_1 + bid_quote_shift))
        outer_bid = int(round(fair - make_edge_2 + bid_quote_shift))
        inner_ask = int(round(fair + make_edge_1 + ask_quote_shift))
        outer_ask = int(round(fair + make_edge_2 + ask_quote_shift))

        inner_bid = min(inner_bid, best_ask - 1)
        outer_bid = min(outer_bid, best_ask - 1)
        inner_ask = max(inner_ask, best_bid + 1)
        outer_ask = max(outer_ask, best_bid + 1)

        if buy_cap > 0 and allow_inner_bid:
            q1 = min(mm_size_1, buy_cap)
            if q1 > 0:
                orders.append(Order(OSM, inner_bid, q1))
                buy_cap -= q1
        if buy_cap > 0 and allow_outer_bid and outer_bid < inner_bid:
            q2 = min(OSM_MM_SIZE_2, buy_cap)
            if q2 > 0:
                orders.append(Order(OSM, outer_bid, q2))
                buy_cap -= q2

        if sell_cap > 0 and allow_inner_ask:
            q1 = min(mm_size_1, sell_cap)
            if q1 > 0:
                orders.append(Order(OSM, inner_ask, -q1))
                sell_cap -= q1
        if sell_cap > 0 and allow_outer_ask and outer_ask > inner_ask:
            q2 = min(OSM_MM_SIZE_2, sell_cap)
            if q2 > 0:
                orders.append(Order(OSM, outer_ask, -q2))

        self._osm_last_mid = mid
        self._osm_last_micro = micro
        return orders

    def _estimate_pepper_drift(self) -> float:
        hist = self._pep_mid_history
        if len(hist) < PEP_ADAPT_MIN_OBS:
            return PEP_PRIOR_DRIFT
        window = hist[-PEP_ADAPT_WINDOW:] if len(hist) > PEP_ADAPT_WINDOW else hist
        observed = (window[-1] - window[0]) / max(1, len(window) - 1)
        return PEP_ADAPT_BLEND * observed + (1 - PEP_ADAPT_BLEND) * PEP_PRIOR_DRIFT

    def _trade_pepper(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        tob = self._top_of_book(od)
        if tob is None:
            return orders
        best_bid, bbv, best_ask, bav = tob

        micro = (bbv * best_ask + bav * best_bid) / (bbv + bav)
        mid = (best_bid + best_ask) / 2.0
        self._pep_mid_history.append(mid)

        drift = self._estimate_pepper_drift()
        fair = micro + drift * PEP_HOLD_HORIZON

        limit = POS_LIMIT[PEP]
        buy_cap = limit - position
        sell_cap = limit + position

        for p, v in self._sorted_asks(od):
            if buy_cap <= 0:
                break
            if p <= fair - PEP_TAKE_EDGE:
                q = min(v, buy_cap)
                if q > 0:
                    orders.append(Order(PEP, p, q))
                    buy_cap -= q
                    position += q
            else:
                break

        for p, v in self._sorted_bids(od):
            if sell_cap <= 0:
                break
            if p >= fair + PEP_TAKE_EDGE:
                q = min(v, sell_cap)
                if q > 0:
                    orders.append(Order(PEP, p, -q))
                    sell_cap -= q
                    position -= q
            else:
                break

        our_bid = int(round(fair - PEP_MAKE_EDGE_BID))
        our_ask = int(round(fair + PEP_MAKE_EDGE_ASK))
        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)

        if drift < 0.03:
            our_bid = int(round(fair - max(PEP_MAKE_EDGE_BID, 2)))
            our_ask = int(round(fair + 1))
            our_ask = max(our_ask, best_bid + 1)

        if buy_cap > 0:
            orders.append(Order(PEP, our_bid, min(PEP_MM_SIZE, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(PEP, our_ask, -min(PEP_MM_SIZE, sell_cap)))

        return orders
