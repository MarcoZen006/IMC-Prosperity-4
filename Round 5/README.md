# Round 5 — Techniques Used

Round 5 is the final round, where the bot must trade **every product on
the exchange** including ones it has never seen. The strategy is a
single, generic, parameter-light **Avellaneda–Stoikov-style market
maker** applied uniformly across all products, with adaptive position
limits, end-of-day inventory release, and lightweight per-product
volatility / momentum tracking.

## Stochastic / Time-Series Models

- **EWMA realised variance** (RiskMetrics-style) per product:
  `Var_t = 0.10·Δ² + 0.90·Var_{t−1}`, where `Δ = micro_t − micro_{t−1}`.
  Bounded to `[0.25, 10000]`. Realised volatility is `σ̂ = √Var`.
- **EMA of mid price** per product: `EMA_t = 0.04·mid + 0.96·EMA_{t−1}`
  — a slow single-pole IIR low-pass providing a long-horizon anchor.
- **Momentum EMA** on first differences:
  `mom_t = 0.20·Δ + 0.80·mom_{t−1}` — a faster IIR filter on micro
  returns, used as a directional lean.
- **Time-decay risk multiplier**: `risk_mult = 1 + 0.55·t²`, with
  `t ∈ [0,1]` the normalised intraday clock. Quadratic in time so risk
  aversion grows sharply near the close.

## Quantitative / Microstructure Signals

- **Microprice**: `(bid·askVol + ask·bidVol) / (bidVol+askVol)` — the
  size-weighted fair-price estimator at the top of book.
- **Top-of-book imbalance**:
  `OBI = clip((bidVol − askVol) / total, ±1)`.
- **Composite fair**:
  `fair = micro + 0.10·clip(mom, ±12) + 0.20·OBI·spreadFactor + 0.06·(EMA − micro, clipped ±1)`.
  The three nudges combine momentum, microstructure pressure, and a
  long-horizon mean-reversion pull, all bounded to prevent any single
  signal from dominating.

## Economic / Market-Making Logic

- **Avellaneda–Stoikov reservation price**:
  `r = fair − q · γ · σ² · risk_mult − tanh(2·invFrac) · (0.85 + 0.12σ) · risk_mult`,
  with `γ = 0.05` and `q = position`. The first term is the standard
  inventory-skew, linear in inventory and proportional to variance;
  the second term is a **soft-limit `tanh` skew** that saturates near
  the position limit and prevents the bot from adding to a near-full
  book.
- **Optimal half-spread** scaled by realised vol: `h = clip(2 + 0.5·σ,
  1, 24)` — a minimum spread of 2 plus a vol-scaling term.
- **Vol-scaled and inventory-scaled order sizing**:
  `size = max(1, ⌊10/σ⌋ · (1 − min(0.6, |invFrac|)))` — smaller when
  vol is high or the position is already large.
- **End-of-day size taper**: ×0.80 after `t > 0.88`, ×0.60 after
  `t > 0.94` — gradual shrink in size as expiry approaches.
- **Take then make**:
  - Take when `best_ask < r − h` (or `best_bid > r + h`), but suppress
    same-side takes near the close that would deepen the position.
  - Then post a passive quote at `r ± h`, snapped to one inside the
    spread when `spread ≥ 3`, otherwise at the same level as the best
    bid / ask.
- **Asymmetric quote sizing for inventory exit**: when long, the ask
  size is scaled by `1 + min(1.2, 2·|invFrac|)` and the bid size by
  `1 − min(0.5, |invFrac|)` (and symmetrically for short). Encourages
  inventory mean-reversion through quote sizing, not just price.
- **End-of-day inventory release**: in the last 10 % of the session,
  if `|position| > 6 % of limit`, the bot actively dumps `≈ |pos| / 3`
  at the touch — a bounded liquidation schedule designed to flatten
  before close without a single market order.

## Robustness Features

- **Adaptive position limits** for unknown products: `300` if the
  product name contains `VOUCHER` / starts with `VEV_`, `60` for
  baskets, otherwise `50`. Lets the bot trade products it has never
  seen.
- **Adaptive time-fraction normalisation**: handles both 100 k and
  1 M timestamp episodes by switching the denominator at runtime.
- **Stateful across ticks**: variance, mid EMA, momentum, microprice
  history, and a `fill_bias` accumulator are all JSON-serialised
  through `traderData`. Failures during deserialisation degrade
  silently to defaults rather than crashing.
- **Self-cross protection**: when computed `quote_bid ≥ quote_ask`,
  the bot falls back to the visible `best_bid / best_ask`.
