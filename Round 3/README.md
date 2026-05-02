# Round 3 — Techniques Used

Round 3 introduces an **options chain** on top of the spot market.
Products: `HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT` (the underlying), and ten
European call **vouchers** `VEV_4000 … VEV_6500` whose names encode their
strikes. The strategy is structured as (i) a Black–Scholes-based options
market maker that fits the implied-volatility surface every tick, plus
(ii) an Avellaneda–Stoikov-style market maker for the spot legs, plus
(iii) a soft delta-hedge of the option book through the underlying.

## Stochastic / Time-Series Models

- **Black–Scholes–Merton call pricing** under zero rates and zero time
  decay (only "total volatility" `w = σ√T` enters, treated as a single
  parameter):
  `C = S·Φ(d₁) − K·Φ(d₂)`,
  `d₁ = (ln(S/K) + ½w²) / w`, `d₂ = d₁ − w`.
- **Black–Scholes call delta** `Δ = Φ(d₁)`, used both for fair-value
  inventory skew and for computing the residual delta exposure that the
  spot leg should hedge.
- **Cross-sectional implied-volatility fit.** Every tick the bot
  estimates `w` by minimising a weighted, clipped chi-squared:
  `Σ wᵢ · clip((C_model(K, w) − C_market) / spread, ±5)²`,
  via a 60-point linear grid search on `w ∈ [0.005, 0.220]`. Weights
  are `1 / (1 + spread)²`, deep ITM observations are down-weighted by
  ×0.35, and deep OTM "floor" vouchers (`mid ≤ 0.75` and far OTM) are
  excluded — a robustified weighted-least-squares vol-surface fit.
- **EMA of total volatility** (`α = 0.15` on the global level): the new
  cross-sectional fit is blended into a slow EMA so the surface doesn't
  jump, and clipped to `[0.005, 0.220]`.
- **EMA volatility** (per product) on microprice squared changes:
  `Var_t = α·Δ² + (1−α)·Var_{t−1}`, with floor 0.25 and product-specific
  cap. This is RiskMetrics-style EWMA variance.
- **Trend EMA** (per product): `slope_t = 0.20·Δ + 0.80·slope_{t−1}` —
  a single-pole IIR filter on first differences, used as a directional
  signal in the market maker.

## Quantitative / Microstructure Signals

- **Blended microprice** = 70 % top-of-book microprice + 30 % depth
  microprice, where the depth term is a VWAP-cross over the top three
  levels of each side. Robust to thin top-of-book sizes.
- **Order-book imbalance** across the top three levels:
  `OBI = (bidQty − askQty) / (bidQty + askQty)`. Used as a take
  confirmation signal (only cross-the-spread on weak edge if imbalance
  agrees with direction).
- **Spread-scaled minimum take edge** for vouchers:
  `min_take = max(1, 0.55·spread, 0.006·max(20, fair))` — the larger
  of an absolute floor, half the spread, or 60 bps of fair. Edge-sized
  quantity scaling (`base_size · (1 + clip(edge/min_take, 0, 2))`)
  takes more when the mispricing is larger.

## Economic / Market-Making Logic

- **Avellaneda–Stoikov reservation price** for spot products:
  `r = micro − q·γ·σ²`, with the bot blending a global γ with a
  product-specific γ (45 / 55). A trend term `0.15 · clip(slope, ±1.5σ)`
  is added.
- **Convex inventory penalty.** Beyond an inventory fraction of 0.60,
  an extra `((|inv|−0.60)/0.40)² · (0.60 + 0.25σ)` term pushes the
  reservation harder away from the limit — quadratic soft-wall.
- **Half-spread = base + 0.5·σ + 0.35·invFrac** plus a convex term past
  60 % inventory, mirroring the Avellaneda–Stoikov optimal spread up to
  the constant.
- **Voucher fair = max(BSM(S, K, w), intrinsic)** — never quotes below
  intrinsic value, enforcing the no-arbitrage lower bound on a call.
- **Voucher inventory shift**:
  `shift = invFrac · max(1, 0.45·spread) · (1 + 2·Δ)` — scales with
  delta, so deep ITM positions (high delta) are skewed harder than deep
  OTM (low delta), reflecting the larger directional risk per contract.
- **Eligibility filter** (`should_trade_voucher`): a voucher is only
  traded if intrinsic value ≥ 250 **and** market mid ≥ 0.75·intrinsic.
  Otherwise the bot enters a passive **inventory liquidation** mode at
  best bid / ask.
- **Soft delta hedge.** The net call delta of the voucher book is
  computed using the fitted `w`, and 20 % of the resulting delta is
  injected as `extra_inventory` into the underlying (`VELVETFRUIT_EXTRACT`)
  market maker — biases its quotes against the option position so the
  spot leg naturally absorbs option delta over time.

## Implementation Notes

- All persistent EMAs (`past_microprices`, `volatility_ema`, `slope_ema`,
  `total_vol_ema`) are JSON-serialised through `traderData` and reloaded
  on the next tick — making the bot stateful across the 1 M-timestamp
  episode.
- Quote prices are clamped: passive bid never exceeds `best_ask − 1`,
  passive ask never undercuts `best_bid + 1`. If the inside-bid would
  cross the inside-ask after rounding, the bot collapses both quotes
  one tick around `round(fair_adj)`.
- Inventory-stretched quoting (>55 % of limit) disables the same-side
  passive quote and ensures at least one resting quote on the unwind
  side.
