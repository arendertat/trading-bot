# BACKTEST METHODOLOGY SPEC

## Objective
Ensure realistic, bias-free evaluation of the trading system.

---

## 1. Data Integrity

- Use exchange historical data.
- No future data leakage.
- Candle-close decision logic only.

---

## 2. Execution Simulation

Must simulate:

- Limit order fill probability
- Slippage
- Fees (maker/taker)
- Funding payments
- Partial fills

---

## 3. Metrics Required

- CAGR
- Max Drawdown
- Sharpe ratio
- Sortino ratio
- Expectancy (R multiple)
- Win rate
- Average holding time
- Exposure time

---

## 4. Stress Testing

- Slippage +100%
- Spread widening
- Reduced liquidity simulation
- Random latency injection

---

## 5. Acceptance Criteria

Strategy is valid only if:

- OOS expectancy > 0
- Max DD < 20%
- No catastrophic month (> -15%)

---

## 6. Regime Breakdown Analysis

Backtest must show performance segmented by:

- Trending periods
- Ranging periods
- High volatility periods

No single regime should represent >80% of profits.
