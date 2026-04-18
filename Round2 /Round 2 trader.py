"""
IMC Prosperity — Round 2 Trading Bot
====================================

Strategy Summary
----------------
Two products, two very different dynamics:

1) ASH_COATED_OSMIUM
   - Strong mean-reverter around a hard anchor of 10000.
     * Fitted OU process: alpha ≈ 0.26, half-life ≈ 2.65 timesteps.
     * Daily mean stays within 10000.2–10001.6 across 3 days.
     * 95% of mid-price deviations land within ±8.
   - Level-1 imbalance (microprice - mid) is strongly predictive:
     corr ≈ +0.58 with 1-step forward mid-change, persistent out to 20 steps.
   - Approach: aggressive take-liquidity on any ask strictly below fair or
     bid strictly above fair; market-make passive quotes inside the book
     with a small microprice skew and a small position-based skew.

2) INTARIAN_PEPPER_ROOT
   - Deterministic linear drift of +0.1 per timestep, i.e. +1000 per day,
     replicated across all three provided days to within 0.2 XIRECs.
   - Detrended residual is a fast OU (alpha ≈ 1.0, half-life ≈ 0.7 steps,
     std ≈ 2.2–2.5).
   - Approach: since drift dominates noise over any reasonable hold horizon,
     carry a maximum-long position throughout the day.  Pay asks up to
     (fair + drift*H), keep asks far above, collect both the drift and any
     spread the MM activity captures.
   - Safety: an adaptive drift estimator watches actual observed drift for
     the first 300–500 timesteps.  If the sign / magnitude disagrees strongly
     with the prior, the bot softens its long bias.

Market Access Fee (MAF)
-----------------------
bid() returns 3000 XIRECs.  This is deliberately moderate:
  * Base expected PnL per day ≈ 86–90k, so we *want* the extra 25% flow.
  * Hint from the brief: "you could save XIRECs by bidding less while
    staying in the top 50% of bidders". 3000 balances winning the auction
    comfortably vs. not overpaying.

Position Limits: 80 for both products.
"""

from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState


# ---- product constants ------------------------------------------------------
OSM = "ASH_COATED_OSMIUM"
PEP = "INTARIAN_PEPPER_ROOT"
POS_LIMIT = {OSM: 80, PEP: 80}

# ---- osmium parameters (fixed, mean-reverting) ------------------------------
# Layered quotes beat single-layer by ~700/day in backtest (5313 vs 4605).
# Tuning sweep over te ∈ {0.5,1,1.5}, me1 ∈ {2,3}, me2 ∈ {3,4,5}, sizes ∈ {10..30}
# settled on the values below (evaluated on all 3 days with 50% passive fill).
OSM_FAIR = 10000              # hard anchor from OU fit (mean 10000.2–10001.6)
OSM_MICRO_WEIGHT = 0.15       # blend weight for microprice signal
OSM_TAKE_EDGE = 1.0           # take any ask at fair-1 or better (tuned)
OSM_MAKE_EDGE_1 = 3           # inner passive quote offset (close to fair)
OSM_MAKE_EDGE_2 = 4           # outer passive quote offset (deeper in book)
OSM_MM_SIZE_1 = 15            # size on the inner quote
OSM_MM_SIZE_2 = 30            # size on the outer quote (larger — catches big moves)
OSM_POS_SKEW = 0.04           # per-unit-position fair-value skew (small)

# ---- pepper parameters (trending) -------------------------------------------
PEP_PRIOR_DRIFT = 0.10        # +1 per 10 timesteps, +1000 per day
PEP_HOLD_HORIZON = 100        # implied hold window in timesteps
PEP_MAKE_EDGE_BID = 1         # aggressive passive bid
PEP_MAKE_EDGE_ASK = 5         # conservative passive ask (we want to stay long)
PEP_MM_SIZE = 40
PEP_TAKE_EDGE = 0             # take any ask at-or-below fair
# adaptive drift
PEP_ADAPT_MIN_OBS = 300       # don't adapt until we have this many mids
PEP_ADAPT_WINDOW = 400        # rolling window for drift estimate
PEP_ADAPT_BLEND = 0.5         # weight on observed drift vs prior


class Trader:
    # ------------------------------------------------------------------ #
    #  Market Access Fee bid (one-time, only applies in final sim run)   #
    # ------------------------------------------------------------------ #
    def bid(self) -> int:
        # 3000 XIRECs — likely sits above the median bid, leaving comfortable
        # margin given base expected profit of ~85k.  Not aggressive enough
        # to torch upside; not timid enough to risk missing extra flow.
        return 3000

    def __init__(self):
        # Persistent state (Trader is instantiated once per sim run)
        self._pep_mid_history: List[float] = []

    # ==================================================================
    #                           MAIN LOOP
    # ==================================================================
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # --- ASH_COATED_OSMIUM ---
        if OSM in state.order_depths:
            pos = state.position.get(OSM, 0)
            result[OSM] = self._trade_osmium(state.order_depths[OSM], pos)

        # --- INTARIAN_PEPPER_ROOT ---
        if PEP in state.order_depths:
            pos = state.position.get(PEP, 0)
            result[PEP] = self._trade_pepper(state.order_depths[PEP], pos)

        # No conversions / simple traderData
        conversions = 0
        traderData = ""
        return result, conversions, traderData

    # ==================================================================
    #                           HELPERS
    # ==================================================================
    @staticmethod
    def _top_of_book(od: OrderDepth):
        """
        Return (best_bid, best_bid_vol, best_ask, best_ask_vol) or None.
        NOTE: Prosperity represents ask volumes as *negative* integers in
        od.sell_orders — we flip the sign here to always work with positives.
        """
        if not od.buy_orders or not od.sell_orders:
            return None
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        bbv = od.buy_orders[best_bid]
        bav = -od.sell_orders[best_ask]  # flip to positive
        if bbv <= 0 or bav <= 0:
            return None
        return best_bid, bbv, best_ask, bav

    @staticmethod
    def _sorted_asks(od: OrderDepth):
        """Return list of (price, positive_volume) sorted ascending by price."""
        return sorted(((p, -v) for p, v in od.sell_orders.items()),
                      key=lambda x: x[0])

    @staticmethod
    def _sorted_bids(od: OrderDepth):
        """Return list of (price, volume) sorted descending by price."""
        return sorted(od.buy_orders.items(), key=lambda x: -x[0])

    # ==================================================================
    #                         OSMIUM STRATEGY
    # ==================================================================
    def _trade_osmium(self, od: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        tob = self._top_of_book(od)
        if tob is None:
            return orders
        best_bid, bbv, best_ask, bav = tob

        # Microprice = volume-weighted fair.  Small bid volume → micro closer
        # to bid (price likely to fall), small ask volume → micro closer to
        # ask (price likely to rise).
        micro = (bbv * best_ask + bav * best_bid) / (bbv + bav)
        # Blend the known long-run fair (10000) with the short-term micro.
        # Weight is small because OSM_FAIR is a *known* anchor.
        fair = (1 - OSM_MICRO_WEIGHT) * OSM_FAIR + OSM_MICRO_WEIGHT * micro

        limit = POS_LIMIT[OSM]
        buy_cap = limit - position
        sell_cap = limit + position

        # ---------- TAKE: walk the book on both sides ------------------
        for p, v in self._sorted_asks(od):
            if buy_cap <= 0:
                break
            # Take any ask strictly below fair (mean-reversion will pull it up)
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

        # ---------- MAKE: layered passive quotes inside the spread ------
        # Two layers per side:
        #   inner (close to fair, smaller size): high hit rate, small edge
        #   outer (deeper, larger size): catches dislocations, larger edge
        # Position-based skew: when long, nudge quotes down to encourage
        # selling and discourage more buying (and vice versa).
        pos_skew = position * OSM_POS_SKEW

        inner_bid = int(round(fair - OSM_MAKE_EDGE_1 - pos_skew))
        outer_bid = int(round(fair - OSM_MAKE_EDGE_2 - pos_skew))
        inner_ask = int(round(fair + OSM_MAKE_EDGE_1 - pos_skew))
        outer_ask = int(round(fair + OSM_MAKE_EDGE_2 - pos_skew))

        # Never cross the existing book.
        inner_bid = min(inner_bid, best_ask - 1)
        outer_bid = min(outer_bid, best_ask - 1)
        inner_ask = max(inner_ask, best_bid + 1)
        outer_ask = max(outer_ask, best_bid + 1)

        # Place inner first (eats into caps first), then outer only if caps remain.
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

        return orders

    # ==================================================================
    #                         PEPPER STRATEGY
    # ==================================================================
    def _estimate_pepper_drift(self) -> float:
        """Adaptive drift estimate, softened toward the prior +0.1."""
        hist = self._pep_mid_history
        if len(hist) < PEP_ADAPT_MIN_OBS:
            return PEP_PRIOR_DRIFT
        window = hist[-PEP_ADAPT_WINDOW:] if len(hist) > PEP_ADAPT_WINDOW else hist
        observed = (window[-1] - window[0]) / max(1, len(window) - 1)
        # Blend observed drift with the strong prior
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

        # "Fair" here is the expected mid PEP_HOLD_HORIZON steps ahead.
        # A positive drift pushes fair UP, so we happily buy at today's mid
        # (which is below future fair) and stay wary of selling at today's mid.
        fair = micro + drift * PEP_HOLD_HORIZON

        limit = POS_LIMIT[PEP]
        buy_cap = limit - position
        sell_cap = limit + position

        # ---------- TAKE side -----------------------------------------
        # Sweep asks aggressively: any ask <= fair is +EV
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

        # Only lift bids that exceed fair — with +drift this is rare, which
        # is exactly what we want (don't prematurely dump long inventory)
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

        # ---------- MAKE side (asymmetric skew) -----------------------
        # Tight bid (near fair) to stay topped up long; wide ask so we only
        # get hit on real spikes.
        our_bid = int(round(fair - PEP_MAKE_EDGE_BID))
        our_ask = int(round(fair + PEP_MAKE_EDGE_ASK))
        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)

        # Safety: if drift has collapsed or gone negative, tighten the ask
        # and widen the bid to avoid being stuck long
        if drift < 0.03:
            our_bid = int(round(fair - max(PEP_MAKE_EDGE_BID, 2)))
            our_ask = int(round(fair + 1))
            our_ask = max(our_ask, best_bid + 1)

        if buy_cap > 0:
            orders.append(Order(PEP, our_bid, min(PEP_MM_SIZE, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(PEP, our_ask, -min(PEP_MM_SIZE, sell_cap)))

        return orders
