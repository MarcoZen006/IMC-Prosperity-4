# Round 2 — Techniques Used

Round 2 trades the same two products as round 1 — `ASH_COATED_OSMIUM` and
`INTARIAN_PEPPER_ROOT` — but with a doubled position limit (80 vs 50) and
a redesigned strategy. Osmium is treated as a stationary asset around a
known fair (`OSM_FAIR = 10000`) with regime-switching "hold modes," while
Pepper is treated as a directionally drifting asset whose drift is
estimated online and used to forecast a holding-horizon fair.

## Stochastic / Time-Series Models

- **Empirical drift estimation for Pepper.** The per-tick drift is
  estimated as `(P_T − P_0) / (T − 1)` over a rolling window of 400
  observations once at least 300 mids have been seen — i.e., an
  endpoints-OLS slope estimator on the last window of mids.
- **Bayesian-style blend with a prior**:
  `drift = β · observed + (1 − β) · prior`, with `β = 0.5` and
  `prior = 0.10`. This is a fixed-weight shrinkage estimator that
  prevents the empirical slope from being noise-dominated early on.
- **Holding-horizon expected-value forecast**:
  `fair = micro + drift · H`, with `H = 80` ticks. Equivalent to taking
  the expectation of a Brownian-with-drift process at horizon `H` and
  quoting around it.
- **Regime detection via two-threshold hysteresis** (Osmium hold modes).
  Long-hold mode arms when both `pressure > 0.18` and `slope > 0.08`,
  and disarms when either falls below tighter exit thresholds (0.08 /
  0.03). This is a Schmitt-trigger style state machine that prevents
  rapid mode chatter.

## Quantitative / Microstructure Signals

- **Multi-level microprice** computed at three depths:
  - Top-of-book microprice: `(bidVol·ask + askVol·bid) / (bidVol+askVol)`.
  - Two-level VWAP-microprice: bid VWAP and ask VWAP across the top two
    levels, then size-cross weighted.
  - Three-level VWAP-microprice: same construction across the top three.
  - Final estimate uses the three-level form, falling back to two-level,
    then to top.
- **Pressure** signal: `pressure = micro − mid` — captures one-sided
  size imbalance scaled to price units.
- **Micro slope**: `Δmicro` between consecutive ticks — a short-horizon
  velocity signal used (with pressure) to flip the bot in / out of hold
  mode.
- **Inventory bands** (Q1 = 20, Q2 = 45, max = 80) categorise the book
  position into safe / cautious / dangerous regimes, each with its own
  set of take penalties, take bonuses, and quote suppression rules.

## Economic / Market-Making Logic

- **Linear-blend fair value** for Osmium:
  `fair = (1 − w) · 10000 + w · micro − γ · last_move`, with
  `w = 0.34` and `γ = 0.12`. The `last_move` term is a return-reversal
  shrinkage that pulls fair against very recent moves (a microstructure
  bounce-back assumption).
- **Jump regime widening**. When `|last_move| ≥ 3.0`, the bot switches
  to a tighter take edge but wider make edges and a different quote
  size. This is a discrete-state stochastic-volatility approximation:
  the recent absolute return is used as a one-shot vol proxy.
- **Two-tier passive market making**: an inner quote (size 10, edge 4)
  and an outer quote (size 30, edge 5) on each side. The inner provides
  liquidity at tight edge; the outer captures size on momentum-driven
  retracements.
- **Asymmetric edges and penalties as functions of inventory**:
  - Inside Q1: symmetric.
  - Q1–Q2 (medium): same-side inner quote disabled; same-side take edge
    penalised by 0.15 unless pressure agrees.
  - Above Q2 (large): both inner and outer same-side quotes disabled;
    same-side take edge penalised by 0.40, opposite-side take edge
    discounted by 0.25 (encourages liquidating).
- **Hold-mode quote distortion**:
  - Inside hold mode: penalise the unwinding take edge (don't dump),
    widen the unwinding quote.
  - When hold support is "lost": flip to an unwind bonus and tighten
    the unwinding quote (race to exit).
  - Large inventory: always tighten the unwinding quote regardless.
- **Pepper asymmetric quoting**. Make edges are 1 (bid) and 5 (ask),
  reflecting the long-positive drift assumption — quote tightly on the
  side you want fills and far on the side you don't. When estimated
  drift collapses (`< 0.03`), revert to a more symmetric quote.

## Implementation Notes

- The Osmium hold-mode flags (`_osm_long_hold_mode`, `_osm_short_hold_mode`)
  and the Pepper mid-history are kept as object state, but the trader
  is recreated each backtest run, so persistence is in-process only
  (no `traderData` JSON serialisation).
- Take loops are short-circuited as soon as the next price level fails
  the edge condition, exploiting that order books are sorted.
- The `bid()` method returning 5000 is a sealed-bid auction stub for
  the manual round; it isn't used by the algorithmic loop.
