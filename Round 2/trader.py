from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState

OSM = "ASH_COATED_OSMIUM"
PEP = "INTARIAN_PEPPER_ROOT"
POS_LIMIT = {OSM: 80, PEP: 80}

# Close-to-original, data-driven tweaks only
OSM_FAIR = 10000
OSM_MICRO_WEIGHT = 0.20
OSM_TAKE_EDGE = 1.0
OSM_MAKE_EDGE_1 = 4
OSM_MAKE_EDGE_2 = 5
OSM_MM_SIZE_1 = 10
OSM_MM_SIZE_2 = 30
OSM_POS_SKEW = 0.04
OSM_RET_REV = 0.15

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
    def _sorted_asks(od: OrderDepth):
        return sorted(((p, -v) for p, v in od.sell_orders.items()), key=lambda x: x[0])

    @staticmethod
    def _sorted_bids(od: OrderDepth):
        return sorted(od.buy_orders.items(), key=lambda x: -x[0])

    def _trade_osmium(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        tob = self._top_of_book(od)
        if tob is None:
            return orders
        best_bid, bbv, best_ask, bav = tob

        micro = (bbv * best_ask + bav * best_bid) / (bbv + bav)
        mid = (best_bid + best_ask) / 2.0
        fair = (1 - OSM_MICRO_WEIGHT) * OSM_FAIR + OSM_MICRO_WEIGHT * micro
        if self._osm_last_mid is not None:
            fair -= OSM_RET_REV * (mid - self._osm_last_mid)

        limit = POS_LIMIT[OSM]
        buy_cap = limit - position
        sell_cap = limit + position

        for p, v in self._sorted_asks(od):
            if buy_cap <= 0:
                break
            if p < fair - OSM_TAKE_EDGE + 1e-6:
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
            if p > fair + OSM_TAKE_EDGE - 1e-6:
                q = min(v, sell_cap)
                if q > 0:
                    orders.append(Order(OSM, p, -q))
                    sell_cap -= q
                    position -= q
            else:
                break

        pos_skew = position * OSM_POS_SKEW
        inner_bid = int(round(fair - OSM_MAKE_EDGE_1 - pos_skew))
        outer_bid = int(round(fair - OSM_MAKE_EDGE_2 - pos_skew))
        inner_ask = int(round(fair + OSM_MAKE_EDGE_1 - pos_skew))
        outer_ask = int(round(fair + OSM_MAKE_EDGE_2 - pos_skew))

        inner_bid = min(inner_bid, best_ask - 1)
        outer_bid = min(outer_bid, best_ask - 1)
        inner_ask = max(inner_ask, best_bid + 1)
        outer_ask = max(outer_ask, best_bid + 1)

        if buy_cap > 0:
            q1 = min(OSM_MM_SIZE_1, buy_cap)
            orders.append(Order(OSM, inner_bid, q1))
            buy_cap -= q1
        if buy_cap > 0 and outer_bid < inner_bid:
            q2 = min(OSM_MM_SIZE_2, buy_cap)
            orders.append(Order(OSM, outer_bid, q2))

        if sell_cap > 0:
            q1 = min(OSM_MM_SIZE_1, sell_cap)
            orders.append(Order(OSM, inner_ask, -q1))
            sell_cap -= q1
        if sell_cap > 0 and outer_ask > inner_ask:
            q2 = min(OSM_MM_SIZE_2, sell_cap)
            orders.append(Order(OSM, outer_ask, -q2))

        self._osm_last_mid = mid
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
