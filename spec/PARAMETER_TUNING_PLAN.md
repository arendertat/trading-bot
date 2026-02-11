# PARAMETER TUNING PLAN (Professional Framework)

## Objective
Optimize strategy parameters without overfitting and while preserving out-of-sample robustness.

---

## 1. Core Principles

- Never tune on the entire dataset.
- Always maintain out-of-sample validation.
- Prefer stability over peak performance.
- Parameter robustness > peak Sharpe.

---

## 2. Data Split Framework

For each symbol:

- 60% In-sample (IS)
- 20% Validation
- 20% Out-of-sample (OOS)

No parameter selection allowed using OOS data.

---

## 3. Parameters Allowed to Tune

- Stop distance (%)
- Target R multiple
- RSI thresholds
- ADX thresholds
- ATR trailing multiplier

Parameters NOT allowed to tune:

- Risk per trade
- Kill switch limits
- Max open positions

Risk parameters are invariant.

---

## 4. Walk-Forward Optimization

Step 1: Optimize on IS.
Step 2: Validate on Validation slice.
Step 3: Roll window forward.
Step 4: Repeat.
Step 5: Final evaluation on untouched OOS.

---

## 5. Robustness Tests

- Parameter sensitivity heatmaps
- Monte Carlo trade order reshuffling
- Slippage stress test (+50% slippage)
- Fee stress test (+50% fees)

Strategy only approved if:
- Expectancy remains positive under stress
- Drawdown < predefined tolerance

---

## 6. Overfitting Guardrails

Reject parameter sets if:
- IS Sharpe >> OOS Sharpe
- Performance collapses with small parameter change
- Trade count < 100

---

## 7. Deployment Rule

Only deploy parameters that:
- Perform acceptably across multiple regimes
- Show stability in rolling 6-month windows
