# Round 4 — Techniques Used

Round 4 trades `HYDROGEL_PACK` and the same `VEV_*` voucher chain from
round 3, with `VELVETFRUIT_EXTRACT` priced as the underlying but **not
actively traded** (the delta-hedge cap is set to zero in this version).
The strategy switches from the round-3 implied-vol-fitting approach to
a **conservative low-vol BSM model**, hard-coded volatility-skew biases,
and an explicit time-to-expiry calendar.

## Stochastic / Time-Series Models

- **Black–Scholes–Merton call pricing** with proper time-to-expiry:
  `total_vol = σ·√T`, then standard `S·Φ(d₁) − K·Φ(d₂)`. Here
  `σ = 0.10` (low and constant — a deliberate underpricing of vol so
  the bot leans toward selling premium) and `T = (7 − days_passed) / 252`.
- **Day counter / calendar**: a wraparound detector
  (`state.timestamp < last_timestamp ⇒ day_index += 1`) drives the
  remaining-days-to-expiry, then `T` enters the BSM formula —
  capturing **theta decay** explicitly across the multi-day backtest.
- **AR(1) microstructure correction.** Each non-voucher product applies
  an `acf_adj = ρ₁ · (mid_t − mid_{t−1})` term, with empirically
  measured first-order autocorrelations:
  - HYDROGEL: `ρ₁ = −0.124`
  - VELVET: `ρ₁ = −0.160`
  Since both are negative, the term acts as a **mean-reverting
  one-tick correction** — the natural microstructure bounce-back.
- **Ornstein–Uhlenbeck pull on HYDROGEL**:
  `ou_adj = 0.10 · (μ − ema)` with `μ = 9995.4`. Pulls fair toward the
  long-run mean at rate 10 % per tick — a discrete OU fair-value
  adjustment.
- **EMA mid** per product with product-specific `α`
  (HYDROGEL 0.06, VELVET 0.08, others 0.10) — single-pole IIR low-pass
  on the volume-weighted mid.

## Quantitative / Microstructure Signals

- **VWAP mid**: VWAP across the full visible buy side and full visible
  sell side, then averaged. Robust to top-of-book size noise.
- **Order-book imbalance** across the **whole visible book**:
  `OBI = (Σ bidVol − Σ askVol) / total`. Used as a directional shift
  in the fair: `+ 2.0 · OBI` for VELVET / vouchers, `+ 3.0 · OBI` for
  HYDROGEL.
- **Intrinsic-value floor**: voucher fair clamped to `intrinsic + 0.5`
  for strikes ≤ 5000, enforcing the no-arbitrage lower bound on
  ITM calls.

## Economic / Market-Making Logic

- **Per-product TAKE_EDGE / MAKE_EDGE table** — every product gets a
  hand-tuned threshold for crossing the spread vs quoting passively.
  Far OTM vouchers (`VEV_6000`, `VEV_6500`) are flagged as
  `FLOOR_VOUCHERS` with edge = 999 (effectively never traded).
- **Volatility sell bias** (`VOL_SELL_BIAS`): a per-strike negative
  shift applied to the voucher quote-fair — pushes both sides of the
  passive quote down, so fills net result in net-short volatility.
  This is a hand-coded short-vol skew, not an arbitrage-fitted one.
- **Inventory skew on quote fair**:
  `quote_fair = fair − inv·MAKE_EDGE·skew_mult + vol_bias`,
  with `skew_mult = 2.0` for HYDROGEL (doubled inventory aversion).
- **Asymmetric size scaling on passive quotes**:
  - Buy size scaled by `max(0.25, 1 − inv⁺)` — when long, bid less.
  - Sell size scaled by `max(0.25, 1 + inv⁻)` — when short, ask less.
- **Edge-scaled take quantity**: `qty = min(book_qty, limit, max_take,
  base_size + edge)` — takes more aggressively when the mispricing is
  larger.
- **End-of-day cooldown** for HYDROGEL: in the last 1 000 timestamps
  of the day, aggressive crossing is disabled (passive only) to avoid
  accumulating inventory close to the close.

## Delta-Hedging Framework (Disabled This Round)

- **Net call delta** is computed across the active voucher book using
  BSM delta `Φ(d₁)` with the same low σ.
- The **target VELVET hedge** is `−round(net_call_delta)`, clipped to
  `±VELVET_HEDGE_CAP`. With cap = 0 in this round, no hedge is taken;
  the machinery (`_velvet_hedge_target`, `_trade_velvet_hedge`) is
  dormant code retained for re-enabling.

## Implementation Notes

- State (`ema`, `prev_mid`, `last_mid`, `day_index`, `last_timestamp`)
  is JSON-serialised through `traderData`, so the bot is stateful
  across ticks and across days.
- Working position is tracked on a per-call basis (`working_pos = dict
  (state.position)`) so multiple orders within the same tick respect
  the cumulative limit.
- Quote prices are clipped to `(best_bid+1, best_ask−1)` to avoid
  self-crossing and to ensure passive quotes always rest in the book.
