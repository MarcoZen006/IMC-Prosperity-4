# Round 1 Python Trader — Techniques Used

Round 1 trades two products: `ASH_COATED_OSMIUM` (a stationary, mean-reverting
asset) and `INTARIAN_PEPPER_ROOT` (a slowly drifting trend asset that becomes
unreliable late in the day). The bot is a regime-switching market maker that
combines stochastic mean-reversion modelling, Bayesian state estimation, and
inventory-aware microstructure quoting.

## Stochastic / Time-Series Models

- **Ornstein–Uhlenbeck (OU) process** for Osmium. The asset is modelled as
  `dX_t = θ(μ − X_t)dt + σ dW_t` with calibrated long-run mean `μ ≈ 10000.20`,
  reversion speed `θ = 0.2427`, and tick noise `σ = 3.494`. The long-run
  spread `OU_STD ≈ σ / √(2θ)` is used as the stationary standard deviation.
- **Slow adaptive drift on μ**. The OU mean is updated each tick by an
  exponential filter `μ_t = (1−α)μ_{t−1} + α·mid` with `α = 5e-4`, allowing
  the mean to track slow regime drift without overreacting to noise.
- **Kalman filter** on Pepper. A scalar Kalman filter tracks a detrended
  observation `obs = mid − slope·t` with process variance `Q = 0.10` and
  measurement variance `R = 5.00`. Standard predict / Kalman-gain / update
  recursion produces the smoothed fair, then re-trends with `slope·t`.
- **Linear trend / drift model** for Pepper, `mid_t ≈ slope·t + ε_t`, with a
  fixed per-tick slope of `0.001` used as the prior for the Kalman filter.
- **Local realised volatility** estimator over a rolling window of mid-price
  first differences: `σ̂ = √(Σ Δ² / n)`. Used to dynamically size the OU
  band and to detect erratic regimes.

## Quantitative / Microstructure Signals

- **Microprice** (size-weighted mid):
  `μ̂ = (bid·askVol + ask·bidVol) / (bidVol + askVol)`, a standard
  short-horizon fair-price estimator that beats the arithmetic mid in the
  presence of one-sided pressure.
- **Order-book imbalance** across the top three levels:
  `OBI = (Σ bidVol − Σ askVol) / (Σ bidVol + Σ askVol) ∈ [−1, 1]`.
- **Toxicity / adverse-selection filter**: flow is flagged "toxic" when
  `|OBI| > 0.55` and recent momentum `mid_t − mid_{t−6}` agrees in sign
  with the imbalance — the standard signature of informed flow about to
  push the mid through your quote.
- **Z-score** of mid vs OU mean, `z = (mid − μ) / σ_dyn`, where
  `σ_dyn = clip(0.75·OU_STD + 0.25·σ̂_local, 3, 12)` blends the calibrated
  long-run dispersion with the local realised vol.
- **Residual Kalman z-score** on Pepper, `|mid − KF_x| / σ_resid`, used as
  a **regime-change detector**: three consecutive readings above 3.5σ trip
  the bot into mean-reversion fallback mode.

## Economic / Market-Making Logic

- **Inventory-aware reservation price** (Avellaneda–Stoikov style):
  `fair = μ − k·inv − γ·z + λ·OBI`, where the inventory term `k·inv` skews
  the quote against the current position (encouraging mean-reversion of
  inventory toward zero) and the z-term leans against extreme deviations.
- **State-dependent crossing thresholds**. The minimum edge required to
  cross the spread is a step function of `|z|`: more aggressive (1 tick)
  when far from fair, more conservative (4 ticks) when close. Toxicity
  inflates the threshold; wide spreads (≥ 6) tighten it.
- **Half-spread** scales with realised volatility:
  `h = clip(round(0.35·σ_dyn), 1, 5)`, a discrete approximation of the
  optimal market-maker spread `½γσ²T + (2/γ)log(1 + γ/k)`.
- **Signal-boosted quote sizing**: base size shrinks linearly in `|inv|`
  and grows in `|z|`, with the boost zeroed out when the position already
  leans the wrong way relative to the signal (avoids doubling down).
- **Two-layer quoting**: an inside quote at `bid+1 / ask−1` plus a deeper
  passive quote at `fair ± h` — captures both rebate-style and adverse-
  selection-resistant fills.

## Regime-Switching Logic

- **Pepper trend → mean-reversion switch.** The bot starts in a directional
  "buy and hold the trend" mode (`target = limit` early, ramping down to 0
  near `t = 985 000`), then permanently flips to mean-reversion when any
  of: (i) timestamp ≥ 910 000, (ii) the Kalman residual z-score breaches
  3.5σ for 3 ticks, or (iii) the regime flag was previously latched.
- **Forced end-of-day flatten** at `t = 995 000` to avoid carrying
  inventory past the close.
- **Mean-reversion fallback** uses a short rolling-mean fair
  (window = 18) instead of the Kalman-smoothed trend, with all the same
  microstructure machinery (z-score, imbalance, toxicity, two-layer
  quotes).

## Implementation Notes

- All persistent state (OU mean estimate, Kalman state `(x, P)`, regime
  flag, mid history) is JSON-serialised through `traderData` so the bot is
  Markov across ticks.
- Quote prices are floored / ceiled to the integer grid and clipped to
  `bid+1 / ask−1` to prevent self-crossing or unintentionally improving
  the spread by more than one tick.
