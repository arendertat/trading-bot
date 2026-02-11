# PERFORMANCE_TRACKER_ADDENDUM.md
# Performance Tracking & Strategy Selection Clarifications (Source-of-Truth Addendum)

## Purpose
This addendum clarifies implementation details for performance tracking and strategy selection to reduce ambiguity for code generation.

If any conflict exists, BOT_SPEC_FINAL.md remains the primary source of truth.

---

## 1) What is being tracked?
Track performance **per strategy** (and optionally per strategy+regime), based on **closed trades only**.

Each closed trade must produce a record including:
- pnl_usd (net)
- fees_usd
- funding_usd
- pnl_r_multiple (net R)
- strategy
- regime
- symbol
- timestamps

Net PnL definition:
- net_pnl_usd = gross_pnl_usd - fees_usd - funding_usd
- pnl_r_multiple should be computed using net_pnl_usd.

---

## 2) Rolling window
Default rolling window size:
- PERF_WINDOW_TRADES = 50

Implementation requirement:
- Use the **last 50 closed trades for a strategy** (FIFO rolling window).
- If there are fewer than MIN_TRADES_BEFORE_CONFIDENCE (default 20), confidence must be penalized (low confidence).

Recommended implementation:
- Use an in-memory deque(maxlen=50) per strategy, persisted via logs/DB on restart.
- On startup, rebuild the deques by reading the most recent trade log records.

---

## 3) Core metrics (per strategy)
Given a rolling set of R-multiples:

- win_rate = count(R > 0) / N
- expectancy_R = mean(R)
- avg_win_R = mean(R | R > 0)
- avg_loss_R = mean(R | R < 0)

Drawdown:
- Build a rolling equity curve from cumulative R (or cumulative USD scaled).
- max_drawdown_pct computed over that curve for the rolling window.

---

## 4) Strategy score and confidence
Score definition (default):
- score = expectancy_R - DD_PENALTY_WEIGHT * max_drawdown_pct

Confidence definition:
- confidence = normalize(score) into [0,1] using a stable mapping, e.g.:
  - clamp(0.5 + score / SCORE_SCALE, 0, 1)
Where SCORE_SCALE is a configurable constant to avoid extreme sensitivity.

Hard gate:
- If confidence < CONFIDENCE_THRESHOLD => NO TRADE
- If expectancy_R <= 0 and DISABLE_STRATEGY_IF_EXPECTANCY_NEGATIVE => strategy is not eligible.

---

## 5) Regime-specific tracking (recommended)
Maintain both:
- Global per-strategy window (all regimes)
- Per-strategy+regime window (TREND, RANGE, HIGH_VOL)

Selection should prioritize regime-specific performance when enough samples exist.
Fallback to global when samples are insufficient.

---

## 6) Logging source
PerformanceTracker must use one of:
- JSONL trade log file, OR
- SQLite/Postgres DB

Minimum requirement:
- JSONL is acceptable if it supports filtering last N trades efficiently (load last lines).
- On restart, rebuild state from logs to avoid losing performance memory.

---

## 7) Required outputs
PerformanceTracker must expose methods to return, for any strategy (and optionally regime):
- N (sample count)
- expectancy_R
- win_rate
- max_drawdown_pct
- confidence
- last_updated timestamp

StrategySelector must:
- request these metrics
- apply eligibility gates
- return selected strategy or NO TRADE, with an explanation (for logging).
