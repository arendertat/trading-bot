# TRADING BOT IMPLEMENTATION PLAN
## Binance USDT-M Perpetual Futures Bot - Complete Execution Roadmap

**Document Version:** 1.0
**Date:** 2026-02-12
**Target Audience:** Executor Claude Instance
**Project Status:** Milestone 3 Task 1 Complete ✅ | 28 Tasks Remaining

---

## TABLE OF CONTENTS

1. [PROJECT OVERVIEW](#1-project-overview)
2. [QUICK WINS](#2-quick-wins)
3. [RISK REGISTER](#3-risk-register)
4. [TESTING STRATEGY](#4-testing-strategy)
5. [MILESTONE 3: Core Trading Logic](#milestone-3-core-trading-logic)
6. [MILESTONE 4: Risk Engine](#milestone-4-risk-engine)
7. [MILESTONE 5: Execution Engine](#milestone-5-execution-engine)
8. [MILESTONE 6: State Management](#milestone-6-state-management)
9. [MILESTONE 7: Logging & Monitoring](#milestone-7-logging--monitoring)
10. [MILESTONE 8: Integration Testing](#milestone-8-integration-testing)
11. [MILESTONE 9: Paper Live Validation](#milestone-9-paper-live-validation)
12. [MILESTONE 10: Micro Live Deployment](#milestone-10-micro-live-deployment)
13. [MILESTONE 11: Production Operations](#milestone-11-production-operations)
14. [APPENDIX A: Critical Path & Dependencies](#appendix-a-critical-path--dependencies)
15. [APPENDIX B: Progress Tracking Template](#appendix-b-progress-tracking-template)

---

# 1. PROJECT OVERVIEW

## 1.1 Executive Summary

This implementation plan covers the development of a production-grade, institutional-quality Binance USDT-M Perpetual Futures trading bot. The bot implements strict risk controls, regime-based strategy selection, and performance-driven decision-making to achieve stable growth with capital preservation as the primary objective.

**Project Scope:**
- **Total Tasks:** 29 (1 complete, 28 remaining)
- **Total Milestones:** 11
- **Estimated LOC:** 14,000-18,000 lines (excluding tests)
- **Estimated Test LOC:** 6,000-8,000 lines
- **Target Timeline:** 27-38 days with parallelization
- **Quality Standard:** Institutional-grade, production-ready

**Core Principles:**
1. **Risk Control > Performance** - Capital preservation is paramount
2. **Configuration-Driven** - Zero hardcoded values
3. **Test-First** - Comprehensive unit, integration, and acceptance testing
4. **State Recovery** - Graceful handling of crashes and restarts
5. **Safe Mode** - Automatic fallback on errors
6. **Observability** - Complete logging and monitoring

## 1.2 Current State

**Completed: Milestone 3, Task 1 - Universe Selection**

Deliverables:
- `bot/universe/selector.py` (273 LOC)
- `bot/universe/models.py` (dataclasses)
- Unit tests: 25 tests passing (678 LOC)
- Documentation: Complete module README
- Quality metrics:
  - Type hints: 100%
  - Docstrings: 100%
  - Test coverage: >80%
  - Configuration-driven: 100%

**Reference Implementation Quality:**
The completed universe module establishes the quality bar for all remaining work:
- Thread-safe data structures with explicit locking
- Comprehensive error handling with custom exceptions
- Retry logic with exponential backoff
- Module-level logging with appropriate levels
- Pydantic-based configuration models
- Pure functions where possible
- Deterministic behavior
- Zero hardcoded constants

## 1.3 Remaining Work Overview

**Milestone 3 (4 tasks remaining):** Core Trading Logic
- Task 2: Regime Detection Engine
- Task 3: Strategy Implementation (3 strategies)
- Task 4: Strategy Selection Engine
- Task 5: Milestone 3 Integration & Testing

**Milestone 4 (4 tasks):** Risk Engine
- Task 6: Core Position Sizing
- Task 7: Daily/Weekly Stops & Risk Limits
- Task 8: Correlation Filter & Direction Limits
- Task 9: Portfolio Risk Validation

**Milestone 5 (3 tasks):** Execution Engine
- Task 10: Order Lifecycle & Fill Handling
- Task 11: Trailing & Advanced Exit Logic
- Task 12: Execution Integration Testing

**Milestone 6 (2 tasks):** State Management
- Task 13: Position & Order Reconciliation
- Task 14: Persistence & Recovery

**Milestone 7 (4 tasks):** Logging & Monitoring
- Task 15: Trade & Event Logging
- Task 16: Daily Reporting & Notifications
- Task 17: Health Monitor & Safe Mode
- Task 18: Scheduler & Event Loop

**Milestone 8 (5 tasks):** Integration Testing
- Task 19: E2E Happy Path Tests
- Task 20: Kill Switch & Risk Limit Tests
- Task 21: Recovery & Reconciliation Tests
- Task 22: Correlation & Portfolio Tests
- Task 23: Acceptance Test Suite

**Milestone 9 (3 tasks):** Paper Live Validation
- Task 24: Paper Live Setup
- Task 25: Paper Live Execution (7 days)
- Task 26: Paper Live Analysis

**Milestone 10 (2 tasks):** Micro Live Deployment
- Task 27: Micro Live Deployment (30+ days)
- Task 28: Micro Live Analysis & Scaling Decision

**Milestone 11 (1 task):** Production Operations
- Task 29: Parameter Tuning & Optimization

## 1.4 Critical Path Visualization

```
Foundation (Complete)
    ↓
Task 2 (Regime) ─────┐
    ↓                │
Task 3 (Strategies)  ├─→ Task 4 (Strategy Selector)
    ↓                │       ↓
    └────────────────┘   Task 5 (M3 Integration)
                              ↓
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
   Task 6-9            Task 10-12           Task 13-14
   (Risk Engine)    (Execution Engine)  (State Management)
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ↓
                     Task 15-18 (Logging/Monitoring)
                             ↓
                     Task 19-23 (Integration Tests)
                             ↓
                     Task 24-26 (Paper Live)
                             ↓
                     Task 27-28 (Micro Live)
                             ↓
                     Task 29 (Production Ops)
```

**Longest Sequential Path:** Task 1 → 2 → 4 → 5 → 6 → 10 → 13 → 15 → 18 → 23 → 24 → 25 → 27 (13 tasks)

**Estimated Critical Path Duration:** 25-32 days

## 1.5 Parallel Work Opportunities

**Parallel Group 1 (After Task 5):**
- Tasks 6-9 (Risk Engine components) - Can run in parallel
- Estimated: 6-8 days → 2-3 days with parallelization

**Parallel Group 2 (After Task 9):**
- Task 10 (Order Manager)
- Task 11 (Exit Logic)
- Tasks 13-14 (State Management)
- Estimated: 8-10 days → 3-4 days with parallelization

**Parallel Group 3 (After Task 14):**
- Tasks 15-17 (Logging, Reporting, Health Monitor)
- Estimated: 6-8 days → 2-3 days with parallelization

**Parallel Group 4 (After Task 18):**
- Tasks 19-22 (E2E Test Scenarios) - Can run in parallel
- Estimated: 8-10 days → 3-4 days with parallelization

**Total Parallelization Savings:** 15-20 days → Achievable in 27-30 days

## 1.6 Technology Stack

**Core Technologies:**
- **Python:** 3.11+ (modern type hints, performance)
- **Exchange Client:** ccxt 4.0+ (Binance abstraction)
- **Config Management:** Pydantic 2.0+ (validation, type safety)
- **Data Processing:** NumPy, Pandas (technical indicators)
- **Testing:** pytest 7.4+ (unit, integration, acceptance)
- **Logging:** Standard library logging + JSONL format
- **Notifications:** python-telegram-bot
- **Environment:** python-dotenv (secrets management)

**Development Tools:**
- Type checking: mypy (100% coverage required)
- Code formatting: black (consistent style)
- Linting: ruff (fast Python linter)
- Coverage: pytest-cov (>80% target)

**Deployment:**
- Docker (containerization)
- docker-compose (orchestration)
- VPS (Ubuntu 22.04 LTS)

---

# 2. QUICK WINS

Quick wins are tasks that provide maximum value with minimal effort, helping build momentum and confidence early in the project.

## 2.1 Immediate Quick Wins (Week 1)

### QW1: Feature Engine Implementation
**Task:** Implement pure indicator functions (RSI, ADX, ATR, EMA, BB, z-scores)
**Why Quick Win:**
- No external dependencies (pure math)
- Well-defined formulas
- Can use TA-Lib or pandas-ta libraries
- Enables downstream testing

**Effort:** 0.5-1 day
**Impact:** Unblocks Tasks 2, 3, 4
**Files:** `bot/data/features.py` (200-300 LOC)

### QW2: Regime Detection Rule Implementation
**Task:** Implement rule-based regime classifier
**Why Quick Win:**
- Clear, deterministic rules from spec
- No ML/training required
- Testable with synthetic data
- Core intelligence component

**Effort:** 1-2 days
**Impact:** Unblocks strategy selection
**Files:** `bot/core/regime_detector.py` (250-350 LOC)

### QW3: Position Sizing Calculator
**Task:** Pure math functions for position sizing
**Why Quick Win:**
- Self-contained calculations
- No state dependencies
- Easy to unit test (100+ test cases)
- Critical for risk management

**Effort:** 0.5-1 day
**Impact:** Enables risk engine completion
**Files:** `bot/risk/position_sizing.py` (150-200 LOC)

## 2.2 High-Value Quick Wins (Week 2)

### QW4: JSONL Logging System
**Task:** Implement structured trade/event logging
**Why Quick Win:**
- Standard library (no external deps)
- Schema already defined (LOG_SCHEMA.json)
- Enables audit trail immediately
- Required for all downstream work

**Effort:** 1 day
**Impact:** Enables performance tracking, reporting, recovery
**Files:** `bot/state/logger.py` (200-250 LOC)

### QW5: Configuration Validation
**Task:** Complete Pydantic models for all config sections
**Why Quick Win:**
- Pattern already established in Task 1
- Prevents runtime errors early
- Self-documenting via Field() constraints
- Catches invalid configs before deployment

**Effort:** 1 day
**Impact:** Prevents configuration-related bugs
**Files:** `bot/config/models.py` (additions)

## 2.3 Morale Boost Quick Wins

### QW6: Telegram Notification Setup
**Task:** Basic Telegram bot integration
**Why Morale Boost:**
- Visible, tangible output
- Easy to demo
- Useful for monitoring
- Fun to implement

**Effort:** 0.5 day
**Impact:** User engagement, monitoring capability
**Files:** `bot/reporting/notifier.py` (150-200 LOC)

### QW7: Daily Report Generator
**Task:** Formatted daily summary report
**Why Morale Boost:**
- Uses data from logging system
- Visible deliverable
- Demonstrates end-to-end flow
- Easy to iterate on format

**Effort:** 1 day
**Impact:** Operational visibility
**Files:** `bot/reporting/daily_report.py` (200-300 LOC)

## 2.4 Quick Win Execution Strategy

**Week 1 Plan:**
- Day 1: QW1 (Features) + QW3 (Position Sizing)
- Day 2-3: QW2 (Regime Detection)
- Day 4: QW4 (Logging)
- Day 5: QW5 (Config Validation)

**Result:** By end of Week 1, core intelligence (regime detection) + risk foundation (position sizing) + logging infrastructure complete. This unblocks 50% of remaining work.

**Week 2 Plan:**
- Day 1-2: Strategy implementations (Task 3)
- Day 3: QW6 (Telegram) + QW7 (Daily Report)
- Day 4-5: Strategy selector (Task 4)

**Result:** By end of Week 2, complete trading intelligence pipeline working (universe → regime → strategies → selection → sizing).

---

# 3. RISK REGISTER

This section identifies project-level risks with mitigation strategies.

## 3.1 High-Impact Risks

### R1: Exchange API Instability
**Risk:** Binance API rate limits, outages, or breaking changes
**Probability:** Medium (30%)
**Impact:** High (blocks all work)
**Mitigation:**
- Implement aggressive retry logic with exponential backoff
- Health monitor with safe mode fallback
- Use testnet for development where possible
- Mock exchange client for testing
- API wrapper abstraction layer (ccxt helps here)
- Monitor Binance API status page
- Have backup API keys ready

**Contingency:** If prolonged outage (>24h), switch to historical data simulation for testing

### R2: Correlation Calculation Performance
**Risk:** Computing rolling correlations on 1h returns for 72h lookback across 6+ symbols could be slow
**Probability:** Medium (40%)
**Impact:** Medium (affects 5m cycle timing)
**Mitigation:**
- Pre-compute correlation matrix on symbol selection (00:00 UTC)
- Use NumPy vectorized operations (not pandas iterrows)
- Cache correlation results (recompute only on new hour close)
- Set correlation update frequency (e.g., every 4 hours vs every 5m)
- Profile early, optimize before it's critical

**Contingency:** If too slow, reduce corr_lookback_hours from 72 to 48

### R3: State Reconciliation Edge Cases
**Risk:** Crash during order placement leaves orphan orders or positions in inconsistent state
**Probability:** Medium-High (50%)
**Impact:** High (duplicate orders, lost capital)
**Mitigation:**
- Idempotent order IDs (deterministic clientOrderId)
- Comprehensive reconciliation on startup (fetch all open orders/positions)
- Cancel stale orders >1h old with no associated position
- Log all state transitions immediately (before API calls)
- Extensive integration testing of crash scenarios
- Use exchange timestamps (not local clock)

**Contingency:** Manual intervention protocol documented; safe mode prevents compounding

### R4: Parameter Tuning Overfitting
**Risk:** Optimized parameters work on historical data but fail in live trading
**Probability:** High (60%)
**Impact:** Medium (poor performance, but risk controls prevent catastrophe)
**Mitigation:**
- Walk-forward validation (60/20/20 split)
- Out-of-sample testing mandatory
- Stress tests (+50% slippage, +50% fees)
- Monte Carlo simulation (reshuffled trades)
- Conservative parameter defaults (from spec)
- Only tune strategy entry/exit params (not risk params)
- Require OOS expectancy > 0 and max DD < 20%

**Contingency:** Fall back to conservative defaults if optimized params fail in paper live

## 3.2 Medium-Impact Risks

### R5: Regime Misclassification
**Risk:** Rule-based regime detector incorrectly classifies market state
**Probability:** Medium-High (50%)
**Impact:** Medium (wrong strategy selection, lower performance)
**Mitigation:**
- Confidence scoring with minimum threshold (0.55)
- CHOP_NO_TRADE fallback when uncertain
- Extensive backtesting with labeled regime data
- Log regime + confidence at every decision
- Post-trade analysis of regime accuracy
- Manual review of regime transitions

**Contingency:** Lower confidence threshold temporarily or switch to single best-performing strategy

### R6: WebSocket Data Gaps
**Risk:** WebSocket disconnects cause stale data, missed candle closes
**Probability:** Medium (40%)
**Impact:** Medium (missed trades, stale features)
**Mitigation:**
- Heartbeat monitoring (detect stale data >30s)
- Auto-reconnect with exponential backoff
- REST fallback for missing candles (fetch_ohlcv)
- Safe mode trigger if data consistently stale
- Timestamp validation (exchange time vs local)

**Contingency:** If WebSocket unstable, fall back to REST polling (higher latency but reliable)

### R7: Insufficient Paper Live Duration
**Risk:** 7 days of paper live may not cover all market conditions
**Probability:** Medium (50%)
**Impact:** Medium (undiscovered bugs in live)
**Mitigation:**
- Extend paper live if no trades executed (need >= 10 trades)
- Run paper live across different market regimes
- Stress test with historical volatile periods
- Comprehensive integration test suite
- Acceptance tests simulate 30+ days

**Contingency:** Extend paper live to 14 days if <10 trades or major regime not tested

## 3.3 Low-Impact Risks

### R8: Telegram API Failures
**Risk:** Notifications fail to send
**Probability:** Low (20%)
**Impact:** Low (monitoring impact only)
**Mitigation:**
- Retry failed notifications
- Queue notifications for later send
- Log all notifications to event log as backup
- Don't block trading on notification failure

**Contingency:** Monitor event logs directly if Telegram down

### R9: Funding Rate Volatility
**Risk:** Extreme funding rates during volatile periods
**Probability:** Low (15%)
**Impact:** Low (affects PnL but universe filter excludes worst)
**Mitigation:**
- Universe filter excludes abs(funding_rate) > 0.15%
- Daily universe refresh removes high-funding symbols
- Track funding costs in PnL calculations
- Log funding rate at trade open/close

**Contingency:** Manually blacklist specific symbols if funding consistently extreme

### R10: Docker Deployment Issues
**Risk:** Container fails to start, networking issues, volume mounts
**Probability:** Low (20%)
**Impact:** Low (deployment only, not code)
**Mitigation:**
- Test Docker setup locally before VPS
- Use docker-compose for clear configuration
- Health checks in docker-compose.yml
- Volume mounts for logs (persist outside container)
- Document troubleshooting steps

**Contingency:** Run without Docker (direct Python) if container issues persist

## 3.4 Risk Summary Matrix

| Risk ID | Risk | Probability | Impact | Priority |
|---------|------|-------------|---------|----------|
| R1 | Exchange API Instability | Medium | High | 🔴 Critical |
| R2 | Correlation Performance | Medium | Medium | 🟠 High |
| R3 | State Reconciliation | Medium-High | High | 🔴 Critical |
| R4 | Parameter Overfitting | High | Medium | 🟠 High |
| R5 | Regime Misclassification | Medium-High | Medium | 🟠 High |
| R6 | WebSocket Data Gaps | Medium | Medium | 🟡 Medium |
| R7 | Insufficient Paper Live | Medium | Medium | 🟡 Medium |
| R8 | Telegram API Failures | Low | Low | 🟢 Low |
| R9 | Funding Rate Volatility | Low | Low | 🟢 Low |
| R10 | Docker Deployment | Low | Low | 🟢 Low |

**Action Items:**
1. Prioritize R1, R3 in Milestone 5 (Execution Engine)
2. Address R2 early in Task 8 (Correlation Filter)
3. Mitigate R4 through Task 29 (Parameter Tuning) methodology
4. Monitor R5 throughout Task 2 (Regime Detection) testing
5. Handle R6 in Task 18 (Scheduler) with health checks

---

# 4. TESTING STRATEGY

## 4.1 Test Pyramid

```
                    /\
                   /  \
                  /E2E \          5-10% (Acceptance, Paper Live)
                 /------\
                /        \
               /Integration\      15-20% (Multi-component)
              /------------\
             /              \
            /   Unit Tests   \    70-75% (Individual functions/classes)
           /------------------\
```

**Target Distribution:**
- **Unit Tests:** 70-75% of tests (fast, isolated, deterministic)
- **Integration Tests:** 15-20% (component interactions)
- **E2E/Acceptance:** 5-10% (full system, slow but high value)

## 4.2 Unit Testing Strategy

**Scope:** Every pure function, every calculation, every rule

**Components Requiring Extensive Unit Tests:**
1. **Position Sizing** (bot/risk/position_sizing.py)
   - Test matrix: 50+ scenarios (regime × risk % × stop % × leverage)
   - Boundary conditions: zero equity, extreme stop, max leverage
   - Validation failures: insufficient margin, overleveraging

2. **Risk Calculations** (bot/risk/risk_limits.py)
   - Open risk aggregation (multiple positions)
   - Daily/weekly stop triggers
   - Pause logic (7 days, reduced risk)
   - Boundary: exactly at threshold, just above/below

3. **Correlation Filter** (bot/risk/correlation_filter.py)
   - Correlation calculation accuracy (compare to NumPy)
   - Bucketing logic (symbols with corr > 0.85)
   - Same-direction blocking
   - Hedge allowance conditions

4. **Regime Detection** (bot/core/regime_detector.py)
   - Each regime rule (TREND, RANGE, HIGH_VOL, CHOP)
   - Confidence scoring
   - Edge cases: ADX exactly at threshold, conflicting indicators

5. **Feature Engine** (bot/data/features.py)
   - Indicator calculations vs reference libraries
   - Z-score calculations
   - Handling missing data, insufficient candles

6. **Strategy Entry/Exit** (bot/strategies/*)
   - Entry conditions for each strategy
   - Stop/TP price calculations
   - Trailing stop updates
   - Leverage mappings

**Unit Test Pattern:**
```python
class TestPositionSizing:
    """Unit tests for position sizing calculations"""

    @pytest.mark.parametrize("equity,risk_pct,stop_pct,expected_notional", [
        (10000, 0.01, 0.01, 10000),  # $10k equity, 1% risk, 1% stop = $10k notional
        (10000, 0.01, 0.02, 5000),   # $10k equity, 1% risk, 2% stop = $5k notional
        (5000, 0.02, 0.01, 10000),   # $5k equity, 2% risk, 1% stop = $10k notional
    ])
    def test_position_sizing_calculation(self, equity, risk_pct, stop_pct, expected_notional):
        result = calculate_position_size(equity, risk_pct, stop_pct)
        assert result.notional_usd == pytest.approx(expected_notional, rel=1e-6)
```

**Mock Strategy for External Dependencies:**
```python
# Mock exchange responses
@patch('bot.exchange.binance_client.ccxt.binance')
def test_order_placement_idempotency(mock_ccxt):
    mock_exchange = MagicMock()
    mock_exchange.fetch_open_orders.return_value = [existing_order]
    mock_ccxt.return_value = mock_exchange

    # Test that duplicate order is not placed
    result = client.place_order(...)
    assert result == existing_order
    mock_exchange.create_order.assert_not_called()
```

## 4.3 Integration Testing Strategy

**Scope:** Multi-component interactions, data flows

**Key Integration Test Scenarios:**

### IT1: Universe → Regime → Strategy Selection Flow
**Components:** UniverseSelector, RegimeDetector, StrategySelector
**Test:** End-to-end signal generation from symbol list to trade plan
**Assertions:**
- Selected symbols pass all filters
- Regime detected correctly based on features
- Strategy chosen matches regime + performance
- No strategy selected when confidence < threshold

### IT2: Complete Trade Lifecycle
**Components:** StrategySelector, RiskEngine, ExecutionEngine, StateManager
**Test:** Entry signal → position sizing → order placement → fill → stop/TP → exit
**Assertions:**
- Order placed with correct clientOrderId
- Position created on fill
- Stop/TP orders placed immediately
- Exit triggered at correct price
- Trade logged with all required fields
- Performance tracker updated

### IT3: Kill Switch Activation
**Components:** RiskEngine, StateManager, ExecutionEngine
**Test:** Daily stop triggered, verify new entries blocked
**Assertions:**
- Realized PnL crosses daily stop threshold
- New entry signal generated but blocked
- Existing positions remain open (monitoring only)
- Kill switch resets at next UTC day
- Reduced risk applied after weekly stop

### IT4: Crash Recovery
**Components:** StateManager, ExecutionEngine, PerformanceTracker
**Test:** Simulate crash mid-trade, restart, reconcile
**Assertions:**
- Open positions fetched from exchange
- Internal state rebuilt correctly
- No duplicate orders placed
- Trade history loaded from logs
- Performance metrics restored

### IT5: Correlation Filter Blocking
**Components:** RiskEngine, CorrelationFilter, ExecutionEngine
**Test:** Two highly correlated symbols, same direction
**Assertions:**
- First position opens successfully
- Second same-direction position blocked
- Hedge position allowed if correlation < 0.60
- Logs show rejection reason

**Integration Test Pattern:**
```python
class TestTradeLifecycle:
    """Integration test for complete trade flow"""

    def test_long_trade_full_lifecycle(self, mock_exchange, config):
        """Test full long trade: entry → fill → trailing → exit"""

        # Setup: Create bot components
        bot = TradingBot(config, exchange=mock_exchange)

        # Mock market data showing trend regime
        mock_exchange.set_market_data(trending_long_candles)

        # Trigger 5m candle close event
        bot.on_5m_close()

        # Assert: Entry order placed
        assert mock_exchange.last_order.side == "BUY"
        assert mock_exchange.last_order.client_order_id.endswith("_entry")

        # Simulate fill
        mock_exchange.fill_order(mock_exchange.last_order.id, fill_price=50000)

        # Assert: Stop and TP orders placed
        stop_order = mock_exchange.get_order_by_client_id("..._stop")
        tp_order = mock_exchange.get_order_by_client_id("..._tp")
        assert stop_order is not None
        assert tp_order is not None

        # Simulate price reaching TP
        mock_exchange.set_price(50750)  # 1.5R target
        bot.on_5m_close()

        # Assert: Position closed, trade logged
        assert len(bot.state_manager.open_positions) == 0
        trade_log = bot.logger.get_last_trade()
        assert trade_log["result"]["reason"] == "TP"
        assert trade_log["result"]["pnl_r_multiple"] == pytest.approx(1.5, rel=0.1)
```

## 4.4 Acceptance Testing Strategy

**Scope:** Full system validation against BOT_SPEC_FINAL.md requirements

**Acceptance Criteria from Spec (Section 16):**
1. Paper live runs 7 days without crash
2. Restart recovery correctly reconstructs state
3. Kill switch stops new entries
4. No duplicate orders (idempotency)
5. Correlation filter prevents same-bucket stacking
6. Logs and daily reports generated

**Acceptance Test Implementation:**

### AT1: 7-Day Paper Live Stability
**Test:** Simulate 7 calendar days (2016 × 5m candles) with realistic market data
**Pass Criteria:**
- No unhandled exceptions
- All candle close events processed
- >= 10 trades executed (validates trading logic)
- Daily reports generated at 00:05 UTC (7 reports)
- Logs parseable and schema-compliant
- Memory usage stable (<500MB)

### AT2: State Recovery Robustness
**Test:** Crash bot at various points (during order, after fill, during exit)
**Pass Criteria:**
- Position count matches exchange after recovery
- No duplicate orders created
- Trade history complete (no missing trades)
- Performance metrics restored from logs
- All open orders reconciled or cancelled

### AT3: Risk Control Validation
**Test:** Trigger each kill switch (daily stop, weekly stop, open risk, positions, correlation)
**Pass Criteria:**
- Daily stop: blocks entries until next UTC day
- Weekly stop: 7-day pause, reduced risk after
- Max open risk: rejects trade when limit exceeded
- Max positions: rejects 3rd position
- Correlation: rejects same-bucket same-direction

### AT4: Data Quality & Logging
**Test:** All trades produce valid JSONL records
**Pass Criteria:**
- Every closed trade has entry in trades.jsonl
- All required fields populated (per LOG_SCHEMA.json)
- Logs parseable by jq / pandas
- Event log contains system events (safe mode, kill switch, errors)
- No PII or secrets in logs

### AT5: Configuration-Driven Behavior
**Test:** Change config parameters, verify behavior changes
**Pass Criteria:**
- risk_per_trade_pct change reflected in position sizes
- max_monitored_symbols change affects universe size
- confidence_threshold change affects trade frequency
- Leverage mapping change affects margin usage
- No hardcoded values override config

**Acceptance Test Execution:**
```python
class TestAcceptance:
    """Acceptance tests per BOT_SPEC_FINAL.md section 16"""

    def test_paper_live_7_day_stability(self, bot, market_data_7days):
        """Test AT1: 7-day paper live without crash"""

        start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        for i in range(2016):  # 7 days * 288 5m candles
            current_time = start_time + timedelta(minutes=5 * i)
            candles = market_data_7days.get_candles_at(current_time)

            # Trigger 5m close event
            bot.on_5m_close(current_time, candles)

            # Verify no crash (implicit: test continues)
            assert bot.health_monitor.is_healthy()

        # Assert: Multiple trades executed
        trades = bot.logger.get_all_trades()
        assert len(trades) >= 10, f"Expected >= 10 trades, got {len(trades)}"

        # Assert: Daily reports generated
        reports = bot.reporter.get_all_reports()
        assert len(reports) == 7, f"Expected 7 daily reports, got {len(reports)}"

        # Assert: Logs valid
        for trade in trades:
            validate_trade_schema(trade)  # Raises if invalid
```

## 4.5 Test Automation & CI/CD

**Test Execution Pipeline:**

```yaml
# pytest.ini configuration
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (multi-component)
    acceptance: Acceptance tests (slow, full system)
    slow: Tests taking >1s
```

**CI/CD Integration (GitHub Actions example):**

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run unit tests
        run: pytest -v -m unit --cov=bot --cov-report=xml
      - name: Run integration tests
        run: pytest -v -m integration
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**Local Test Commands:**
```bash
# Run all tests
pytest -v

# Run only unit tests (fast)
pytest -v -m unit

# Run with coverage
pytest -v --cov=bot --cov-report=html

# Run specific test file
pytest -v tests/test_position_sizing.py

# Run tests matching pattern
pytest -v -k "test_correlation"

# Run and stop on first failure
pytest -v -x

# Run with detailed output
pytest -v -s
```

## 4.6 Test Data Strategy

**Synthetic Market Data:**
- Generate realistic OHLCV data with known regime characteristics
- Trending periods: ADX > 25, EMA20 > EMA50 (or opposite)
- Ranging periods: ADX < 20, BB width narrow
- High vol periods: ATR z-score > 1.5
- Use data generator module: `tests/fixtures/market_data_generator.py`

**Historical Data:**
- Binance API: fetch_ohlcv for backtesting
- Store in fixtures directory: `tests/fixtures/historical/`
- Use compressed format (parquet or CSV.gz)
- Cover different market conditions (2023 bear, 2024 bull, May 2021 crash)

**Mock Responses:**
- Exchange API responses (tickers, candles, positions, orders)
- WebSocket messages (candle updates, ticker updates)
- Store in: `tests/fixtures/mock_responses/`
- Use pytest fixtures for reusable mocks

## 4.7 Coverage Targets

**Overall Coverage Target:** >80% of critical paths

**Module-Level Targets:**
- `bot/risk/*`: >90% (critical for capital preservation)
- `bot/execution/*`: >85% (critical for order correctness)
- `bot/core/*`: >80% (strategy logic)
- `bot/data/*`: >75% (data handling)
- `bot/exchange/*`: >70% (external dependency, mocked)
- `bot/state/*`: >85% (recovery critical)
- `bot/reporting/*`: >70% (lower risk)

**Coverage Exclusions:**
- Type stubs
- Abstract base classes (only concrete implementations)
- Main entry point (tested via integration)
- Debug/development utilities

**Coverage Reporting:**
```bash
# Generate HTML coverage report
pytest --cov=bot --cov-report=html

# Open coverage report
open htmlcov/index.html

# Show missing lines
pytest --cov=bot --cov-report=term-missing
```

---

# MILESTONE 3: CORE TRADING LOGIC

## 🎯 Milestone 3 Overview

**Objective:** Implement the core trading intelligence pipeline from market data to trade signals.

**Deliverables:**
- Regime detection system (TREND, RANGE, HIGH_VOL, CHOP_NO_TRADE)
- Three strategy implementations (Trend Pullback, Breakout, Range Mean Reversion)
- Performance-based strategy selection engine
- Complete feature calculation pipeline
- Integration tests for end-to-end signal generation

**Critical Path:** This milestone contains the bot's "brain" - all downstream work depends on these components producing valid, testable trade signals.

**Risk Assessment:**
- **High risk:** Regime misclassification could lead to wrong strategy selection
- **Medium risk:** Strategy logic bugs could produce invalid entries
- **Low risk:** Feature calculations (well-defined math)

**Milestone Acceptance Criteria:**
- [ ] All 4 regime types correctly detected on test data
- [ ] All 3 strategies generate valid entry signals
- [ ] Strategy selector chooses based on rolling performance
- [ ] No trade signal when confidence < threshold
- [ ] Integration test simulates 30-day signal generation
- [ ] 100% type hints, 100% docstrings
- [ ] >80% test coverage

**Estimated Timeline:** 8-10 days
**Estimated LOC:** 2,500-3,000 lines (code) + 1,200-1,500 lines (tests)

---

## Task 2: Regime Detection Engine

### Goal
Implement rule-based market regime classifier that categorizes market state into TREND, RANGE, HIGH_VOL, or CHOP_NO_TRADE with confidence scoring.

### Why Critical
Regime detection is the first decision point in the trading pipeline. Incorrect regime classification leads to wrong strategy selection, resulting in poor performance. The confidence scoring mechanism acts as a filter to prevent trades in uncertain market conditions.

### Dependencies
**Requires Completed:**
- ✅ Task 1: Universe Selection (symbol list available)
- Feature Engine (indicators calculated)
- Data Store (candles available)

**Blocks:**
- Task 3: Strategy Implementation (strategies need regime context)
- Task 4: Strategy Selection (selection based on regime)
- Task 5: Milestone 3 Integration

**Can Run Parallel With:**
- Task 3: Strategy Implementation (can start once regime types defined)

### Approach

#### Step 1: Define Regime Types & Confidence Model

**1.1 Create Regime Enums and Data Models**

```python
# bot/core/constants.py
from enum import Enum

class RegimeType(str, Enum):
    """Market regime classifications"""
    TREND = "TREND"
    RANGE = "RANGE"
    HIGH_VOL = "HIGH_VOL"
    CHOP_NO_TRADE = "CHOP_NO_TRADE"
```

**1.2 Create Regime Result Dataclass**

```python
# bot/core/models.py
from dataclasses import dataclass
from typing import Dict, Optional
from bot.core.constants import RegimeType

@dataclass
class RegimeResult:
    """Result of regime detection"""
    regime: RegimeType
    confidence: float  # 0.0 to 1.0
    metrics: Dict[str, float]  # ADX, ATR_z, BB_width, etc.
    trend_direction: Optional[str] = None  # "bullish" or "bearish" if TREND
    timestamp: int = 0  # Unix timestamp of detection
```

#### Step 2: Implement Feature Dependencies

**2.1 Ensure Required Indicators Available**

Features needed from Feature Engine:
- ADX(14) on 5m timeframe
- ATR(14) on 5m timeframe
- ATR z-score (over 100 bars)
- EMA20, EMA50 on 1h timeframe (trend filter)
- Bollinger Band width on 5m

**2.2 Create Feature Validation Function**

```python
def validate_features_for_regime(features: FeatureSet) -> bool:
    """
    Validate all required features present for regime detection.

    Args:
        features: FeatureSet object with calculated indicators

    Returns:
        True if all required features available

    Raises:
        ValueError: If critical features missing
    """
    required = ["adx_5m", "atr_z_5m", "ema20_1h", "ema50_1h", "bb_width_5m"]
    missing = [f for f in required if not hasattr(features, f) or features.__dict__[f] is None]

    if missing:
        raise ValueError(f"Missing required features for regime detection: {missing}")

    return True
```

#### Step 3: Implement Regime Detection Rules

**3.1 HIGH_VOL Detection (Highest Priority)**

```python
def _detect_high_vol(
    self,
    atr_z: float,
    threshold: float
) -> tuple[bool, float]:
    """
    Detect HIGH_VOL regime.

    Args:
        atr_z: ATR z-score
        threshold: High vol threshold from config (default 1.5)

    Returns:
        (is_high_vol, confidence)
    """
    if atr_z > threshold:
        # Confidence increases with how far above threshold
        # confidence = 0.6 at threshold, up to 1.0 at threshold + 1.5
        confidence = min(1.0, 0.6 + (atr_z - threshold) / 1.5 * 0.4)
        return True, confidence

    return False, 0.0
```

**3.2 TREND Detection (Second Priority)**

```python
def _detect_trend(
    self,
    adx: float,
    ema20_1h: float,
    ema50_1h: float,
    adx_threshold: float
) -> tuple[bool, float, Optional[str]]:
    """
    Detect TREND regime with direction.

    Args:
        adx: ADX(14) value
        ema20_1h: EMA20 on 1h timeframe
        ema50_1h: EMA50 on 1h timeframe
        adx_threshold: Min ADX for trend (default 25)

    Returns:
        (is_trend, confidence, direction)
        direction: "bullish" or "bearish" or None
    """
    if adx < adx_threshold:
        return False, 0.0, None

    # Determine trend direction
    if ema20_1h > ema50_1h:
        direction = "bullish"
    elif ema20_1h < ema50_1h:
        direction = "bearish"
    else:
        # EMAs equal or crossed - uncertain
        return False, 0.0, None

    # Calculate confidence based on ADX strength and EMA separation
    ema_separation_pct = abs(ema20_1h - ema50_1h) / ema50_1h
    adx_strength = (adx - adx_threshold) / 25.0  # Normalize: 25-50 range

    confidence = min(1.0, 0.6 + adx_strength * 0.2 + ema_separation_pct * 100 * 0.2)

    return True, confidence, direction
```

**3.3 RANGE Detection (Third Priority)**

```python
def _detect_range(
    self,
    adx: float,
    bb_width: float,
    adx_max: float,
    bb_width_range: tuple[float, float]
) -> tuple[bool, float]:
    """
    Detect RANGE regime.

    Args:
        adx: ADX(14) value
        bb_width: Bollinger Band width (normalized)
        adx_max: Max ADX for range (default 20)
        bb_width_range: Expected BB width range for ranging market

    Returns:
        (is_range, confidence)
    """
    if adx > adx_max:
        return False, 0.0

    bb_min, bb_max = bb_width_range

    # Check if BB width within range boundaries
    if bb_min <= bb_width <= bb_max:
        # Confidence higher when ADX very low and BB width stable
        adx_confidence = 1.0 - (adx / adx_max)  # Lower ADX = higher confidence
        bb_confidence = 0.7  # Base confidence for BB width match
        confidence = min(1.0, (adx_confidence + bb_confidence) / 2)
        return True, confidence

    return False, 0.0
```

**3.4 CHOP_NO_TRADE Fallback**

```python
def _detect_chop(
    self,
    adx: float,
    bb_width: float,
    spread_ok: bool
) -> tuple[bool, float]:
    """
    Detect CHOP_NO_TRADE regime (fallback).

    Args:
        adx: ADX(14) value
        bb_width: Bollinger Band width
        spread_ok: Whether spread filter passes

    Returns:
        (is_chop, confidence)
    """
    # Chop conditions:
    # - Very low ADX (<15) AND very low BB width (<0.01)
    # - Spread filter fails
    # - No other regime detected with sufficient confidence

    if not spread_ok:
        return True, 1.0  # High confidence: bad spread = no trade

    if adx < 15 and bb_width < 0.01:
        confidence = 0.8  # Choppy, low volatility
        return True, confidence

    return False, 0.5  # Default fallback with medium confidence
```

#### Step 4: Implement Main Detection Logic

**4.1 Create RegimeDetector Class**

```python
# bot/core/regime_detector.py
import logging
from typing import Optional
from bot.config.models import RegimeConfig
from bot.core.models import RegimeResult, FeatureSet
from bot.core.constants import RegimeType

logger = logging.getLogger("trading_bot.core.regime")

class RegimeDetector:
    """
    Rule-based market regime detection with confidence scoring.

    Detects four regime types in priority order:
    1. HIGH_VOL (highest priority)
    2. TREND
    3. RANGE
    4. CHOP_NO_TRADE (fallback)

    Each regime has confidence score [0, 1]. If best regime confidence
    is below threshold, CHOP_NO_TRADE is forced.
    """

    def __init__(self, config: RegimeConfig):
        """
        Initialize regime detector.

        Args:
            config: RegimeConfig with thresholds and parameters
        """
        self.config = config
        logger.info(f"RegimeDetector initialized: ADX_min={config.trend_adx_min}, "
                    f"ADX_max={config.range_adx_max}, conf_threshold={config.confidence_threshold}")

    def detect(
        self,
        features: FeatureSet,
        symbol: str,
        spread_ok: bool = True
    ) -> RegimeResult:
        """
        Detect current market regime for a symbol.

        Args:
            features: Calculated technical indicators
            symbol: Trading symbol
            spread_ok: Whether spread filter passes

        Returns:
            RegimeResult with detected regime and confidence

        Raises:
            ValueError: If required features missing
        """
        # Validate features
        validate_features_for_regime(features)

        # Extract features
        adx = features.adx_5m
        atr_z = features.atr_z_5m
        ema20_1h = features.ema20_1h
        ema50_1h = features.ema50_1h
        bb_width = features.bb_width_5m

        # Priority 1: HIGH_VOL
        is_high_vol, high_vol_conf = self._detect_high_vol(
            atr_z,
            self.config.high_vol_atr_z
        )

        if is_high_vol and high_vol_conf >= self.config.confidence_threshold:
            logger.info(f"{symbol}: HIGH_VOL detected (ATR_z={atr_z:.2f}, conf={high_vol_conf:.2f})")
            return RegimeResult(
                regime=RegimeType.HIGH_VOL,
                confidence=high_vol_conf,
                metrics={"adx": adx, "atr_z": atr_z, "bb_width": bb_width}
            )

        # Priority 2: TREND
        is_trend, trend_conf, direction = self._detect_trend(
            adx,
            ema20_1h,
            ema50_1h,
            self.config.trend_adx_min
        )

        if is_trend and trend_conf >= self.config.confidence_threshold:
            logger.info(f"{symbol}: TREND detected ({direction}, ADX={adx:.2f}, conf={trend_conf:.2f})")
            return RegimeResult(
                regime=RegimeType.TREND,
                confidence=trend_conf,
                metrics={"adx": adx, "atr_z": atr_z, "ema20_1h": ema20_1h, "ema50_1h": ema50_1h},
                trend_direction=direction
            )

        # Priority 3: RANGE
        is_range, range_conf = self._detect_range(
            adx,
            bb_width,
            self.config.range_adx_max,
            (0.01, 0.03)  # Configurable BB width range
        )

        if is_range and range_conf >= self.config.confidence_threshold:
            logger.info(f"{symbol}: RANGE detected (ADX={adx:.2f}, BB_width={bb_width:.4f}, conf={range_conf:.2f})")
            return RegimeResult(
                regime=RegimeType.RANGE,
                confidence=range_conf,
                metrics={"adx": adx, "bb_width": bb_width}
            )

        # Priority 4: CHOP_NO_TRADE (fallback)
        is_chop, chop_conf = self._detect_chop(adx, bb_width, spread_ok)

        logger.warning(f"{symbol}: CHOP_NO_TRADE (no clear regime, conf={chop_conf:.2f})")
        return RegimeResult(
            regime=RegimeType.CHOP_NO_TRADE,
            confidence=chop_conf,
            metrics={"adx": adx, "atr_z": atr_z, "bb_width": bb_width}
        )
```

#### Step 5: Testing Strategy

**5.1 Unit Tests for Each Regime Rule**

```python
# tests/test_regime_detector.py
import pytest
from bot.core.regime_detector import RegimeDetector
from bot.core.models import FeatureSet
from bot.core.constants import RegimeType
from bot.config.models import RegimeConfig

class TestRegimeDetector:
    """Unit tests for regime detection rules"""

    @pytest.fixture
    def config(self):
        """Default regime config"""
        return RegimeConfig(
            trend_adx_min=25,
            range_adx_max=20,
            high_vol_atr_z=1.5,
            confidence_threshold=0.55
        )

    @pytest.fixture
    def detector(self, config):
        """RegimeDetector instance"""
        return RegimeDetector(config)

    def test_detect_high_vol_regime(self, detector):
        """Test HIGH_VOL detection when ATR z-score high"""
        features = FeatureSet(
            adx_5m=20,
            atr_z_5m=2.5,  # High volatility
            ema20_1h=50000,
            ema50_1h=49500,
            bb_width_5m=0.02
        )

        result = detector.detect(features, "BTCUSDT", spread_ok=True)

        assert result.regime == RegimeType.HIGH_VOL
        assert result.confidence > 0.55
        assert result.metrics["atr_z"] == 2.5

    def test_detect_trend_bullish(self, detector):
        """Test TREND detection with bullish direction"""
        features = FeatureSet(
            adx_5m=30,  # Strong trend
            atr_z_5m=1.0,  # Normal volatility
            ema20_1h=50000,  # EMA20 > EMA50 = bullish
            ema50_1h=49000,
            bb_width_5m=0.02
        )

        result = detector.detect(features, "BTCUSDT", spread_ok=True)

        assert result.regime == RegimeType.TREND
        assert result.trend_direction == "bullish"
        assert result.confidence > 0.55

    def test_detect_trend_bearish(self, detector):
        """Test TREND detection with bearish direction"""
        features = FeatureSet(
            adx_5m=28,
            atr_z_5m=0.8,
            ema20_1h=49000,  # EMA20 < EMA50 = bearish
            ema50_1h=50000,
            bb_width_5m=0.02
        )

        result = detector.detect(features, "BTCUSDT", spread_ok=True)

        assert result.regime == RegimeType.TREND
        assert result.trend_direction == "bearish"

    def test_detect_range_regime(self, detector):
        """Test RANGE detection with low ADX"""
        features = FeatureSet(
            adx_5m=15,  # Low ADX
            atr_z_5m=0.5,
            ema20_1h=50000,
            ema50_1h=50100,  # EMAs close
            bb_width_5m=0.02  # Narrow BB width
        )

        result = detector.detect(features, "BTCUSDT", spread_ok=True)

        assert result.regime == RegimeType.RANGE
        assert result.confidence > 0.55

    def test_detect_chop_on_bad_spread(self, detector):
        """Test CHOP_NO_TRADE forced when spread fails"""
        features = FeatureSet(
            adx_5m=30,
            atr_z_5m=1.0,
            ema20_1h=50000,
            ema50_1h=49000,
            bb_width_5m=0.02
        )

        result = detector.detect(features, "BTCUSDT", spread_ok=False)

        assert result.regime == RegimeType.CHOP_NO_TRADE
        assert result.confidence == 1.0

    def test_confidence_below_threshold_forces_chop(self, detector):
        """Test CHOP when best regime confidence < threshold"""
        features = FeatureSet(
            adx_5m=22,  # Between trend and range thresholds
            atr_z_5m=1.2,  # Below high vol threshold
            ema20_1h=50000,
            ema50_1h=49900,  # EMAs very close
            bb_width_5m=0.025
        )

        result = detector.detect(features, "BTCUSDT", spread_ok=True)

        # Should fall back to CHOP due to low confidence
        assert result.regime == RegimeType.CHOP_NO_TRADE
```

**5.2 Integration Test with Feature Engine**

```python
def test_regime_detection_with_real_features(candle_store, config):
    """Integration test: Feature calculation → Regime detection"""

    # Setup: Load historical candles into store
    candles_5m = load_test_candles("BTCUSDT", "5m", count=200)
    candles_1h = load_test_candles("BTCUSDT", "1h", count=100)

    for candle in candles_5m:
        candle_store.add_candle("BTCUSDT", "5m", candle)

    for candle in candles_1h:
        candle_store.add_candle("BTCUSDT", "1h", candle)

    # Calculate features
    feature_engine = FeatureEngine(config.features)
    features = feature_engine.calculate("BTCUSDT", candle_store)

    # Detect regime
    detector = RegimeDetector(config.regime)
    result = detector.detect(features, "BTCUSDT", spread_ok=True)

    # Assertions
    assert result.regime in [RegimeType.TREND, RegimeType.RANGE, RegimeType.HIGH_VOL, RegimeType.CHOP_NO_TRADE]
    assert 0 <= result.confidence <= 1.0
    assert "adx" in result.metrics
```

### Files to Create/Modify

```python
bot/core/regime_detector.py          # RegimeDetector class - 350 LOC
bot/core/models.py                    # RegimeResult dataclass - 30 LOC
bot/core/constants.py                 # RegimeType enum (update) - 10 LOC
tests/test_regime_detector.py        # Unit tests - 500 LOC
tests/test_regime_integration.py     # Integration tests - 200 LOC
```

### Specification References

- **BOT_SPEC_FINAL.md:** Section 6 (Regime Detection)
- **ARCHITECTURE.md:** Module 6 (Regime Detector)
- **CONFIG.example.json:** `regime` section

### Configuration Parameters

```json
{
  "regime": {
    "trend_adx_min": 25,            // Min ADX for TREND regime
    "range_adx_max": 20,            // Max ADX for RANGE regime
    "high_vol_atr_z": 1.5,          // ATR z-score threshold for HIGH_VOL
    "confidence_threshold": 0.55,    // Min confidence to trade
    "bb_width_range_min": 0.01,     // Min BB width for RANGE
    "bb_width_range_max": 0.03      // Max BB width for RANGE
  }
}
```

### Testing Strategy

**Unit Tests (8-10 tests):**
- Each regime rule (HIGH_VOL, TREND, RANGE, CHOP)
- Confidence scoring calculations
- Threshold boundary conditions
- Trend direction detection (bullish vs bearish)
- Fallback to CHOP when confidence low

**Integration Tests (3-5 tests):**
- Feature engine → Regime detector flow
- Spread filter integration (CHOP on bad spread)
- Multiple symbols with different regimes
- Regime transitions over time

**Edge Cases:**
- ADX exactly at threshold (25 for TREND, 20 for RANGE)
- EMAs exactly equal (no trend direction)
- ATR z-score exactly at threshold
- Missing features (should raise ValueError)
- Invalid feature values (NaN, infinity)

### Acceptance Criteria

- [ ] All 4 regime types can be detected
- [ ] Confidence scoring produces values in [0, 1] range
- [ ] Spread filter fails → CHOP_NO_TRADE with confidence 1.0
- [ ] Best regime confidence < threshold → CHOP_NO_TRADE fallback
- [ ] Trend direction correctly identified (bullish/bearish)
- [ ] Unit tests pass for all regime rules
- [ ] Integration tests pass with real feature data
- [ ] Type hints 100%, docstrings 100%
- [ ] No hardcoded values (all from config)
- [ ] Logging at INFO for regime detection, WARNING for CHOP

### Risk Factors & Mitigation

**Risk 1:** Regime misclassification due to noisy indicators
**Mitigation:**
- Confidence scoring acts as filter (threshold 0.55)
- Multiple indicators per regime (not single indicator)
- Conservative thresholds (higher ADX for TREND, lower for RANGE)
- Extensive backtesting with labeled regime data
- Log all regime detections for post-analysis

**Risk 2:** Regime oscillation (frequent regime switches)
**Mitigation:**
- Confidence threshold prevents low-confidence switches
- Trend direction requires EMA separation (not just cross)
- Strategy selector has stability constraint (max 1 switch/day)
- Monitor regime distribution in logs

**Risk 3:** HIGH_VOL overriding TREND during strong trends
**Mitigation:**
- HIGH_VOL has highest priority (correct: high vol = reduce risk)
- Leverage mapping accounts for HIGH_VOL (1.0x vs 2.0x)
- Strategy selector can still choose trend strategies in HIGH_VOL if performance warrants

### Estimated Effort

- **LOC:** 350-400 lines (implementation) + 700-800 lines (tests)
- **Time:** 2-3 days
- **Complexity:** ⭐⭐ Medium

### Success Metrics

- **Regime accuracy:** >80% match with manual labeling on test data
- **CHOP frequency:** <30% of time (too high = too conservative)
- **Trend direction accuracy:** >90% (clear trends should be obvious)
- **Confidence distribution:** >60% of detections with confidence >0.7

### Common Pitfalls

⚠️ **Pitfall 1:** Using only ADX without trend filter
**Avoidance:** Require 1h EMA crossover for TREND confirmation

⚠️ **Pitfall 2:** Not handling EMA crossover edge case
**Avoidance:** If EMA20 == EMA50, return no trend (not bullish/bearish)

⚠️ **Pitfall 3:** Confidence scoring not normalized
**Avoidance:** Clamp confidence to [0, 1] range, document formula

⚠️ **Pitfall 4:** Forgetting to check spread filter
**Avoidance:** Pass spread_ok parameter, force CHOP if False

---

## Task 3: Strategy Implementation

### Goal
Implement three concrete strategy classes (Trend Pullback, Trend Breakout, Range Mean Reversion) with entry conditions, stop/TP calculations, and trailing logic.

### Why Critical
Strategies generate the actual trade signals. Bugs in entry logic or stop/TP calculations directly impact P&L. Each strategy must be thoroughly tested in isolation before integration with the strategy selector.

### Dependencies

**Requires Completed:**
- ✅ Task 1: Universe Selection
- ✅ Task 2: Regime Detection (regime type available)
- Feature Engine (indicators for entry conditions)

**Blocks:**
- Task 4: Strategy Selection (needs strategies to select from)
- Task 5: Milestone 3 Integration

**Can Run Parallel With:**
- Task 2: Regime Detection (can implement strategies while regime detection in progress)

### Approach

#### Step 1: Define Strategy Interface

**1.1 Create Abstract Base Strategy Class**

```python
# bot/strategies/base.py
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
from bot.core.models import FeatureSet, RegimeResult
from bot.core.constants import OrderSide, RegimeType

@dataclass
class StrategySignal:
    """Trade signal generated by strategy"""
    entry: bool  # True if entry conditions met
    side: Optional[OrderSide]  # LONG or SHORT
    stop_pct: float  # Stop loss distance as percentage
    target_r: float  # Take profit in R multiples
    entry_price: float  # Suggested entry price
    stop_price: float  # Stop loss price
    tp_price: float  # Take profit price
    trail_enabled: bool  # Enable trailing stop
    trail_after_r: float  # Enable trailing after this R profit
    atr_trail_mult: float  # Trailing distance multiplier
    reason: str  # Human-readable entry reason

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.

    Each strategy must implement:
    - entry_conditions(): Check if entry conditions met
    - calculate_stop_loss(): Calculate stop price
    - calculate_take_profit(): Calculate TP price
    - leverage_mapping(): Return leverage for given regime
    """

    def __init__(self, config: dict):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy-specific configuration dict
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.name = self.__class__.__name__

    @abstractmethod
    def entry_conditions(
        self,
        features: FeatureSet,
        regime: RegimeResult,
        symbol: str,
        current_price: float
    ) -> tuple[bool, Optional[OrderSide], str]:
        """
        Check if entry conditions are met.

        Args:
            features: Calculated technical indicators
            regime: Current regime detection result
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            (conditions_met, side, reason)
            side: LONG or SHORT if conditions met, else None
            reason: Human-readable explanation
        """
        pass

    @abstractmethod
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: OrderSide,
        atr: float
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry fill price
            side: LONG or SHORT
            atr: Average True Range (14)

        Returns:
            Stop loss price
        """
        pass

    @abstractmethod
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_price: float,
        side: OrderSide
    ) -> float:
        """
        Calculate take profit price.

        Args:
            entry_price: Entry fill price
            stop_price: Stop loss price
            side: LONG or SHORT

        Returns:
            Take profit price
        """
        pass

    def leverage_mapping(self, regime: RegimeType) -> float:
        """
        Map regime to leverage.

        Args:
            regime: Current market regime

        Returns:
            Leverage multiplier (1.0 - 2.0)
        """
        # Default mapping (can be overridden)
        mapping = {
            RegimeType.TREND: 2.0,
            RegimeType.RANGE: 1.5,
            RegimeType.HIGH_VOL: 1.0,
            RegimeType.CHOP_NO_TRADE: 1.0
        }
        return mapping.get(regime, 1.0)

    def generate_signal(
        self,
        features: FeatureSet,
        regime: RegimeResult,
        symbol: str,
        current_price: float
    ) -> Optional[StrategySignal]:
        """
        Generate complete trade signal.

        Args:
            features: Calculated technical indicators
            regime: Current regime detection result
            symbol: Trading symbol
            current_price: Current market price

        Returns:
            StrategySignal if conditions met, else None
        """
        if not self.enabled:
            return None

        # Check entry conditions
        entry, side, reason = self.entry_conditions(features, regime, symbol, current_price)

        if not entry or side is None:
            return None

        # Calculate stop/TP
        stop_price = self.calculate_stop_loss(current_price, side, features.atr_5m)
        tp_price = self.calculate_take_profit(current_price, stop_price, side)

        # Calculate stop distance percentage
        if side == OrderSide.LONG:
            stop_pct = (current_price - stop_price) / current_price
        else:
            stop_pct = (stop_price - current_price) / current_price

        # Target R calculation
        risk_distance = abs(current_price - stop_price)
        target_distance = abs(tp_price - current_price)
        target_r = target_distance / risk_distance if risk_distance > 0 else 1.0

        # Trailing config
        trail_enabled = self.config.get("trail_after_r", 0) > 0
        trail_after_r = self.config.get("trail_after_r", 1.0)
        atr_trail_mult = self.config.get("atr_trail_mult", 2.0)

        return StrategySignal(
            entry=True,
            side=side,
            stop_pct=stop_pct,
            target_r=target_r,
            entry_price=current_price,
            stop_price=stop_price,
            tp_price=tp_price,
            trail_enabled=trail_enabled,
            trail_after_r=trail_after_r,
            atr_trail_mult=atr_trail_mult,
            reason=reason
        )
```

#### Step 2: Implement Strategy A - Trend Pullback

**2.1 Trend Pullback Entry Logic**

```python
# bot/strategies/trend_pullback.py
import logging
from typing import Optional
from bot.strategies.base import Strategy, StrategySignal
from bot.core.models import FeatureSet, RegimeResult
from bot.core.constants import OrderSide, RegimeType

logger = logging.getLogger("trading_bot.strategies.trend_pullback")

class TrendPullbackStrategy(Strategy):
    """
    Trend Pullback Strategy.

    Entry Logic:
    - Long: 1h bullish trend + 5m pullback to EMA20 + RSI 40-50
    - Short: 1h bearish trend + 5m pullback to EMA20 + RSI 50-60

    Stop: 1.0% from entry
    Target: 1.5R
    Trailing: Enabled after 1.0R profit, ATR-based
    """

    def entry_conditions(
        self,
        features: FeatureSet,
        regime: RegimeResult,
        symbol: str,
        current_price: float
    ) -> tuple[bool, Optional[OrderSide], str]:
        """Check trend pullback entry conditions."""

        # Only trade in TREND regime
        if regime.regime != RegimeType.TREND:
            return False, None, "Not in TREND regime"

        # Extract features
        rsi = features.rsi_5m
        ema20_5m = features.ema20_5m
        ema50_5m = features.ema50_5m
        ema20_1h = features.ema20_1h
        ema50_1h = features.ema50_1h

        # Pullback band (price within 0.2% of EMA20)
        ema20_band_pct = self.config.get("ema20_band_pct", 0.002)
        price_distance_from_ema20 = abs(current_price - ema20_5m) / ema20_5m

        # Check long setup
        if regime.trend_direction == "bullish":
            # 1h trend bullish (already validated in regime)
            # 5m structure: EMA20 > EMA50 OR price above EMA50
            structure_ok = ema20_5m > ema50_5m or current_price > ema50_5m

            # Price near EMA20
            near_ema20 = price_distance_from_ema20 <= ema20_band_pct

            # RSI in pullback range (40-50)
            rsi_min = self.config.get("pullback_rsi_long_min", 40)
            rsi_max = self.config.get("pullback_rsi_long_max", 50)
            rsi_ok = rsi_min <= rsi <= rsi_max

            if structure_ok and near_ema20 and rsi_ok:
                reason = f"Trend pullback LONG: 1h bullish, price near EMA20, RSI={rsi:.1f}"
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.LONG, reason

        # Check short setup
        elif regime.trend_direction == "bearish":
            # 1h trend bearish
            # 5m structure: EMA20 < EMA50 OR price below EMA50
            structure_ok = ema20_5m < ema50_5m or current_price < ema50_5m

            # Price near EMA20
            near_ema20 = price_distance_from_ema20 <= ema20_band_pct

            # RSI in pullback range (50-60)
            rsi_min = self.config.get("pullback_rsi_short_min", 50)
            rsi_max = self.config.get("pullback_rsi_short_max", 60)
            rsi_ok = rsi_min <= rsi <= rsi_max

            if structure_ok and near_ema20 and rsi_ok:
                reason = f"Trend pullback SHORT: 1h bearish, price near EMA20, RSI={rsi:.1f}"
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.SHORT, reason

        return False, None, "Pullback conditions not met"

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: OrderSide,
        atr: float
    ) -> float:
        """Calculate stop loss at fixed percentage."""
        stop_pct = self.config.get("stop_pct", 0.01)  # 1.0%

        if side == OrderSide.LONG:
            stop_price = entry_price * (1 - stop_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)

        return stop_price

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_price: float,
        side: OrderSide
    ) -> float:
        """Calculate take profit at fixed R multiple."""
        target_r = self.config.get("target_r_multiple", 1.5)

        risk_distance = abs(entry_price - stop_price)
        target_distance = risk_distance * target_r

        if side == OrderSide.LONG:
            tp_price = entry_price + target_distance
        else:
            tp_price = entry_price - target_distance

        return tp_price
```

#### Step 3: Implement Strategy B - Trend Breakout

**3.1 Trend Breakout Entry Logic**

```python
# bot/strategies/trend_breakout.py
import logging
from typing import Optional
from bot.strategies.base import Strategy
from bot.core.models import FeatureSet, RegimeResult
from bot.core.constants import OrderSide, RegimeType

logger = logging.getLogger("trading_bot.strategies.trend_breakout")

class TrendBreakoutStrategy(Strategy):
    """
    Trend Breakout Strategy.

    Entry Logic:
    - Long: Break above 20-bar high + volume z-score > 1.0
    - Short: Break below 20-bar low + volume z-score > 1.0

    Stop: 1.0% from entry
    Target: No fixed target (trailing only)
    Trailing: Enabled immediately, 2.5 * ATR
    """

    def entry_conditions(
        self,
        features: FeatureSet,
        regime: RegimeResult,
        symbol: str,
        current_price: float
    ) -> tuple[bool, Optional[OrderSide], str]:
        """Check trend breakout entry conditions."""

        # Only trade in TREND regime
        if regime.regime != RegimeType.TREND:
            return False, None, "Not in TREND regime"

        # Extract features
        high_20 = features.high_20_bars  # 20-bar high
        low_20 = features.low_20_bars    # 20-bar low
        volume_z = features.volume_z_5m

        # Volume confirmation threshold
        volume_z_min = self.config.get("breakout_volume_z_min", 1.0)

        # Check long breakout (break above 20-bar high)
        if regime.trend_direction == "bullish" and current_price > high_20:
            if volume_z > volume_z_min:
                reason = f"Trend breakout LONG: Break above 20-bar high, volume_z={volume_z:.2f}"
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.LONG, reason

        # Check short breakout (break below 20-bar low)
        elif regime.trend_direction == "bearish" and current_price < low_20:
            if volume_z > volume_z_min:
                reason = f"Trend breakout SHORT: Break below 20-bar low, volume_z={volume_z:.2f}"
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.SHORT, reason

        return False, None, "Breakout conditions not met"

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: OrderSide,
        atr: float
    ) -> float:
        """Calculate stop loss at fixed percentage."""
        stop_pct = self.config.get("stop_pct", 0.01)  # 1.0%

        if side == OrderSide.LONG:
            stop_price = entry_price * (1 - stop_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)

        return stop_price

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_price: float,
        side: OrderSide
    ) -> float:
        """No fixed TP for breakout - use trailing only."""
        # Return very far TP (100R) - trailing will exit before this
        risk_distance = abs(entry_price - stop_price)

        if side == OrderSide.LONG:
            tp_price = entry_price + risk_distance * 100
        else:
            tp_price = entry_price - risk_distance * 100

        return tp_price
```

#### Step 4: Implement Strategy C - Range Mean Reversion

**4.1 Range Mean Reversion Entry Logic**

```python
# bot/strategies/range_mean_reversion.py
import logging
from typing import Optional
from bot.strategies.base import Strategy
from bot.core.models import FeatureSet, RegimeResult
from bot.core.constants import OrderSide, RegimeType

logger = logging.getLogger("trading_bot.strategies.range_mean_reversion")

class RangeMeanReversionStrategy(Strategy):
    """
    Range Mean Reversion Strategy.

    Entry Logic:
    - Long: RSI < 25 + BB lower band touch
    - Short: RSI > 75 + BB upper band touch

    Stop: 0.8% from entry
    Target: 1.2R
    Trailing: Disabled (fixed TP)
    """

    def entry_conditions(
        self,
        features: FeatureSet,
        regime: RegimeResult,
        symbol: str,
        current_price: float
    ) -> tuple[bool, Optional[OrderSide], str]:
        """Check range mean reversion entry conditions."""

        # Only trade in RANGE regime
        if regime.regime != RegimeType.RANGE:
            return False, None, "Not in RANGE regime"

        # Extract features
        rsi = features.rsi_5m
        bb_upper = features.bb_upper_5m
        bb_lower = features.bb_lower_5m

        # RSI extreme thresholds
        rsi_long_extreme = self.config.get("rsi_long_extreme", 25)
        rsi_short_extreme = self.config.get("rsi_short_extreme", 75)

        # BB touch threshold (within 0.1% of band)
        bb_touch_pct = 0.001

        # Check long setup (oversold)
        if rsi < rsi_long_extreme:
            distance_to_lower = abs(current_price - bb_lower) / bb_lower
            if distance_to_lower <= bb_touch_pct:
                reason = f"Range mean reversion LONG: RSI={rsi:.1f} (oversold), BB lower touch"
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.LONG, reason

        # Check short setup (overbought)
        elif rsi > rsi_short_extreme:
            distance_to_upper = abs(current_price - bb_upper) / bb_upper
            if distance_to_upper <= bb_touch_pct:
                reason = f"Range mean reversion SHORT: RSI={rsi:.1f} (overbought), BB upper touch"
                logger.info(f"{symbol}: {reason}")
                return True, OrderSide.SHORT, reason

        return False, None, "Mean reversion conditions not met"

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: OrderSide,
        atr: float
    ) -> float:
        """Calculate stop loss at fixed percentage (tighter for range)."""
        stop_pct = self.config.get("stop_pct", 0.008)  # 0.8%

        if side == OrderSide.LONG:
            stop_price = entry_price * (1 - stop_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)

        return stop_price

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_price: float,
        side: OrderSide
    ) -> float:
        """Calculate take profit at fixed R multiple."""
        target_r = self.config.get("target_r_multiple", 1.2)

        risk_distance = abs(entry_price - stop_price)
        target_distance = risk_distance * target_r

        if side == OrderSide.LONG:
            tp_price = entry_price + target_distance
        else:
            tp_price = entry_price - target_distance

        return tp_price
```

#### Step 5: Testing Strategy

**5.1 Unit Tests for Each Strategy**

```python
# tests/test_strategies.py
import pytest
from bot.strategies.trend_pullback import TrendPullbackStrategy
from bot.strategies.trend_breakout import TrendBreakoutStrategy
from bot.strategies.range_mean_reversion import RangeMeanReversionStrategy
from bot.core.models import FeatureSet, RegimeResult
from bot.core.constants import OrderSide, RegimeType

class TestTrendPullbackStrategy:
    """Unit tests for Trend Pullback strategy"""

    @pytest.fixture
    def strategy(self):
        """Strategy instance with config"""
        config = {
            "enabled": True,
            "stop_pct": 0.01,
            "target_r_multiple": 1.5,
            "pullback_rsi_long_min": 40,
            "pullback_rsi_long_max": 50,
            "ema20_band_pct": 0.002,
            "trail_after_r": 1.0,
            "atr_trail_mult": 2.0
        }
        return TrendPullbackStrategy(config)

    def test_long_entry_conditions_met(self, strategy):
        """Test long entry when all conditions met"""
        features = FeatureSet(
            rsi_5m=45,  # In pullback range (40-50)
            ema20_5m=50000,
            ema50_5m=49500,
            ema20_1h=50000,
            ema50_1h=49000,
            atr_5m=500
        )

        regime = RegimeResult(
            regime=RegimeType.TREND,
            confidence=0.75,
            metrics={},
            trend_direction="bullish"
        )

        current_price = 50100  # Near EMA20 (within 0.2%)

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is True
        assert side == OrderSide.LONG
        assert "pullback LONG" in reason.lower()

    def test_long_entry_rsi_too_high(self, strategy):
        """Test long entry rejected when RSI too high"""
        features = FeatureSet(
            rsi_5m=55,  # Above pullback range
            ema20_5m=50000,
            ema50_5m=49500,
            ema20_1h=50000,
            ema50_1h=49000,
            atr_5m=500
        )

        regime = RegimeResult(
            regime=RegimeType.TREND,
            confidence=0.75,
            metrics={},
            trend_direction="bullish"
        )

        current_price = 50100

        entry, side, reason = strategy.entry_conditions(features, regime, "BTCUSDT", current_price)

        assert entry is False
        assert side is None

    def test_stop_loss_calculation_long(self, strategy):
        """Test stop loss calculated correctly for long"""
        entry_price = 50000
        side = OrderSide.LONG
        atr = 500

        stop_price = strategy.calculate_stop_loss(entry_price, side, atr)

        expected_stop = 50000 * 0.99  # 1% below entry
        assert stop_price == pytest.approx(expected_stop, rel=1e-6)

    def test_take_profit_calculation_long(self, strategy):
        """Test take profit calculated correctly for long"""
        entry_price = 50000
        stop_price = 49500  # 1% stop
        side = OrderSide.LONG

        tp_price = strategy.calculate_take_profit(entry_price, stop_price, side)

        risk_distance = 500
        target_distance = 500 * 1.5  # 1.5R
        expected_tp = 50000 + 750

        assert tp_price == pytest.approx(expected_tp, rel=1e-6)

    def test_generate_signal_complete(self, strategy):
        """Test full signal generation"""
        features = FeatureSet(
            rsi_5m=45,
            ema20_5m=50000,
            ema50_5m=49500,
            ema20_1h=50000,
            ema50_1h=49000,
            atr_5m=500
        )

        regime = RegimeResult(
            regime=RegimeType.TREND,
            confidence=0.75,
            metrics={},
            trend_direction="bullish"
        )

        current_price = 50100

        signal = strategy.generate_signal(features, regime, "BTCUSDT", current_price)

        assert signal is not None
        assert signal.entry is True
        assert signal.side == OrderSide.LONG
        assert signal.stop_pct == pytest.approx(0.01, rel=1e-3)
        assert signal.target_r == pytest.approx(1.5, rel=0.1)
        assert signal.trail_enabled is True
```

**5.2 Test Each Strategy Independently**

Similar test patterns for TrendBreakoutStrategy and RangeMeanReversionStrategy covering:
- Entry conditions met/not met
- Stop/TP calculations
- Edge cases (price exactly at threshold, RSI boundaries)
- Signal generation with all fields populated

### Files to Create/Modify

```python
bot/strategies/base.py                    # Strategy ABC + StrategySignal - 250 LOC
bot/strategies/trend_pullback.py          # Trend Pullback implementation - 200 LOC
bot/strategies/trend_breakout.py          # Trend Breakout implementation - 180 LOC
bot/strategies/range_mean_reversion.py    # Range Mean Reversion implementation - 180 LOC
tests/test_strategies.py                  # Unit tests for all strategies - 800 LOC
```

### Specification References

- **BOT_SPEC_FINAL.md:** Section 7 (Strategy Library)
- **ARCHITECTURE.md:** Module 7 (Strategy Library)
- **CONFIG.example.json:** `strategies` section

### Configuration Parameters

```json
{
  "strategies": {
    "trend_pullback": {
      "enabled": true,
      "stop_pct": 0.01,               // 1.0% stop
      "target_r_multiple": 1.5,       // 1.5R target
      "pullback_rsi_long_min": 40,
      "pullback_rsi_long_max": 50,
      "pullback_rsi_short_min": 50,
      "pullback_rsi_short_max": 60,
      "ema20_band_pct": 0.002,        // 0.2% from EMA20
      "trail_after_r": 1.0,
      "atr_trail_mult": 2.0
    },
    "trend_breakout": {
      "enabled": true,
      "stop_pct": 0.01,
      "breakout_lookback_bars": 20,
      "breakout_volume_z_min": 1.0,
      "atr_trail_mult": 2.5
    },
    "range_mean_reversion": {
      "enabled": true,
      "stop_pct": 0.008,              // 0.8% stop (tighter)
      "target_r_multiple": 1.2,       // 1.2R target
      "rsi_long_extreme": 25,
      "rsi_short_extreme": 75
    }
  }
}
```

### Acceptance Criteria

- [ ] All 3 strategies implement base Strategy interface
- [ ] Entry conditions generate signals when criteria met
- [ ] Stop loss calculations correct for LONG and SHORT
- [ ] Take profit calculations respect target R multiples
- [ ] Leverage mapping returns correct values per regime
- [ ] generate_signal() returns complete StrategySignal object
- [ ] Unit tests pass for all strategies (30+ tests total)
- [ ] Type hints 100%, docstrings 100%
- [ ] No hardcoded values (all from config)
- [ ] Logging at INFO when signals generated

### Risk Factors & Mitigation

**Risk 1:** Entry conditions too loose (too many false signals)
**Mitigation:**
- Conservative thresholds (RSI ranges, volume z-score)
- Multiple confirmation filters per strategy
- Regime filter prevents wrong-regime entries
- Extensive backtesting to tune thresholds

**Risk 2:** Stop/TP calculations incorrect (math errors)
**Mitigation:**
- Unit tests with parametrized scenarios (50+ combinations)
- Manual calculation verification
- Assert stop < entry < TP for LONG, reverse for SHORT
- Cross-check R multiple: (TP - entry) / (entry - stop) == target_r

**Risk 3:** Trailing logic bugs (covered in Task 11)
**Mitigation:** Strategy only defines trailing parameters, not implementation

### Estimated Effort

- **LOC:** 800-850 lines (implementation) + 800-900 lines (tests)
- **Time:** 2-3 days
- **Complexity:** ⭐⭐ Medium

### Success Metrics

- **Signal frequency:** 1-5 signals per day per strategy (depends on regime)
- **Entry accuracy:** >60% of backtested signals profitable (before fees)
- **Stop/TP valid:** 100% of signals have stop < entry < TP (or reverse for SHORT)

### Common Pitfalls

⚠️ **Pitfall 1:** Not checking regime before entry
**Avoidance:** Always validate regime.regime matches strategy's required regime

⚠️ **Pitfall 2:** Calculating stop/TP with wrong sign (LONG vs SHORT)
**Avoidance:** Parametrized tests covering both sides

⚠️ **Pitfall 3:** Forgetting to validate features exist
**Avoidance:** Check features.has_field() or use try/except AttributeError

⚠️ **Pitfall 4:** Hardcoding RSI/volume thresholds
**Avoidance:** All thresholds from self.config.get()

---

## Task 4: Strategy Selection Engine (Performance-Based)

### Goal
Implement performance tracking and strategy selection based on rolling window metrics (win rate, expectancy_R, drawdown) with confidence scoring.

### Why Critical
Strategy selection is the "AI" layer - choosing which strategy to use based on recent performance prevents continuing with losing strategies. This is the key differentiator from static strategy systems.

### Dependencies
- ✅ Task 2: Regime Detection (regime context)
- ✅ Task 3: Strategies (need strategies to track)
- Trade log reader (for rebuilding performance history)

### Approach Overview

1. **Implement Performance Tracker** - Track rolling window (50 trades) per strategy with metrics:
   - win_rate, avg_R, expectancy_R, max_drawdown_pct, fees_total, funding_total

2. **Implement Selection Logic** - Score strategies based on: `score = expectancy_R - DD_PENALTY * max_DD`, normalize to [0,1] confidence

3. **Add Stability Constraints** - Max 1 strategy switch per day unless expectancy turns negative

4. **Persistence** - Rebuild from trade logs on restart

**Key Implementation:**
```python
class PerformanceTracker:
    """Track rolling performance per strategy"""

    def __init__(self, window_trades: int = 50):
        self.window = window_trades
        self.trades = defaultdict(lambda: deque(maxlen=window_trades))

    def add_trade(self, strategy: str, pnl_r: float, pnl_usd: float, fees: float, funding: float):
        """Add closed trade to rolling window"""
        pass

    def get_metrics(self, strategy: str) -> StrategyMetrics:
        """Calculate metrics for strategy"""
        trades = self.trades[strategy]
        win_rate = sum(1 for t in trades if t.pnl_r > 0) / len(trades)
        expectancy_r = np.mean([t.pnl_r for t in trades])
        # ... calculate all metrics
        return StrategyMetrics(...)
```

### Files to Create
- `bot/core/performance_tracker.py` (300 LOC)
- `bot/core/strategy_selector.py` (250 LOC)
- `tests/test_performance_tracker.py` (400 LOC)
- `tests/test_strategy_selector.py` (400 LOC)

### Testing Focus
- Rolling window correctly maintains FIFO (50 trades)
- Metrics calculations accurate (compare to manual calculation)
- Confidence scoring normalized to [0, 1]
- Selection respects confidence threshold (0.55)
- Stability constraint prevents frequent switching

### Estimated Effort
- **LOC:** 550 lines + 800 test lines
- **Time:** 2-3 days
- **Complexity:** ⭐⭐ Medium

---

## Task 5: Milestone 3 Integration & Testing

### Goal
Integrate all Milestone 3 components and run end-to-end tests simulating 30 days of trading.

### Why Critical
Integration testing validates that universe selection → regime detection → strategy selection → signal generation works as a complete pipeline without gaps.

### Approach Overview

1. **Create Integration Test Suite** - Simulate 5m candle closes for 30 days (8,640 candles)
2. **Test Data Generator** - Generate realistic market data with all regime types
3. **Validate Complete Flow** - Universe → Regime → Strategy Selection → Order Plan generation
4. **Performance Validation** - Strategy selector correctly switches based on rolling performance

**Integration Test Structure:**
```python
def test_m3_complete_pipeline():
    """Test 30-day simulation of signal generation"""
    bot = TradingBot(config)

    for day in range(30):
        for candle_5m in generate_day_candles(day):
            # Universe selection (daily at 00:00)
            if candle_5m.is_midnight():
                universe = bot.universe_selector.build_daily_universe()

            # Per symbol: regime → strategy → signal
            for symbol in universe:
                regime = bot.regime_detector.detect(features, symbol)
                strategy = bot.strategy_selector.select(regime, symbol)
                signal = strategy.generate_signal(features, regime, symbol, price)

                # Validate signal if generated
                if signal:
                    assert_valid_signal(signal)

    # Assert: Signals generated, regime switches happened, no crashes
    assert len(bot.signals) >= 10
```

### Files to Create
- `tests/test_m3_integration.py` (600 LOC)
- `tests/fixtures/market_data_generator.py` (400 LOC)

### Acceptance Criteria
- [ ] 30-day simulation completes without crash
- [ ] Signals generated in all regime types
- [ ] Strategy selector switches based on performance
- [ ] No duplicate signals (same symbol/timestamp)
- [ ] All decision points logged

### Estimated Effort
- **LOC:** 1,000 test lines
- **Time:** 2 days
- **Complexity:** ⭐⭐⭐ High (integration complexity)

---

# MILESTONE 4: RISK ENGINE

## 🎯 Milestone 4 Overview

**Objective:** Implement comprehensive risk management system with position sizing, portfolio limits, kill switches, and correlation filtering.

**Deliverables:**
- Position sizing calculator (risk % → notional USD)
- Daily/weekly stop loss (kill switches)
- Portfolio open risk limiter
- Correlation filter (prevent same-bucket stacking)
- Direction limits (max long/short positions)

**Milestone Acceptance:**
- [ ] Position sizing calculations 100% accurate (50+ test scenarios)
- [ ] Kill switches trigger at exact thresholds
- [ ] Correlation filter blocks same-bucket same-direction
- [ ] Integration tests validate all risk gates

**Estimated Timeline:** 6-8 days (2-3 days with parallelization)
**Estimated LOC:** 1,800-2,200 lines (code + tests)

---

## Task 6: Core Position Sizing

### Goal
Calculate position size (notional USD) based on risk percentage, stop distance, and regime-based leverage.

### Approach Summary

**Formula:**
```
risk_usd = equity_usd * risk_per_trade_pct
notional_usd = risk_usd / stop_pct
leverage = regime_leverage_map[regime]  # TREND: 2.0, RANGE: 1.5, HIGH_VOL: 1.0
margin_required_usd = notional_usd / leverage
```

**Validation:**
- Sufficient free margin available
- Leverage within Binance limits (1-125x, but config restricts to 1-2x)
- Notional meets minimum order size

### Files
- `bot/risk/position_sizing.py` (200 LOC)
- `tests/test_position_sizing.py` (500 LOC - parametrized tests)

### Estimated Effort: 1 day, ⭐ Low complexity

---

## Task 7: Daily/Weekly Stops & Risk Limits

### Goal
Implement kill switches that block new entries when daily or weekly loss thresholds exceeded.

### Approach Summary

**Daily Stop:**
```python
if realized_pnl_pct_today <= DAILY_STOP_PCT:  # -4%
    block_new_entries_until_next_utc_day()
```

**Weekly Stop:**
```python
if realized_pnl_pct_week <= WEEKLY_STOP_PCT:  # -10%
    pause_trading_for_7_days()
    after_pause:
        reduce_risk_to_0.5%_for_3_days()
```

**Open Risk Limit:**
```python
open_risk_usd = sum(abs(position.entry_price - position.stop_price) * position.qty for position in open_positions)
if open_risk_usd > equity_usd * MAX_TOTAL_OPEN_RISK_PCT:  # 2.5%
    reject_new_entry()
```

### Files
- `bot/risk/risk_limits.py` (300 LOC)
- `bot/risk/kill_switch.py` (200 LOC)
- `tests/test_risk_limits.py` (600 LOC)

### Estimated Effort: 2 days, ⭐⭐ Medium complexity

---

## Task 8: Correlation Filter & Direction Limits

### Goal
Prevent portfolio concentration by blocking same-direction positions in correlated symbols.

### Approach Summary

**Correlation Calculation:**
```python
# Compute rolling correlation of 1h returns (72h lookback)
returns_A = np.diff(np.log(prices_A))
returns_B = np.diff(np.log(prices_B))
corr = np.corrcoef(returns_A, returns_B)[0, 1]
```

**Bucketing Logic:**
```python
if corr(symbolA, symbolB) > CORR_THRESHOLD (0.85):
    # Same bucket - enforce rules:
    if existing_position_same_bucket and same_direction:
        reject_new_entry()
```

**Hedging Rule:**
```python
if hedge_position and corr < HEDGE_CORR_MAX (0.60):
    allow_opposite_direction()
```

### Files
- `bot/risk/correlation_filter.py` (350 LOC)
- `tests/test_correlation_filter.py` (500 LOC)

### Estimated Effort: 2-3 days, ⭐⭐⭐ High complexity (correlation math)

---

## Task 9: Portfolio Risk Validation

### Goal
Integrate all risk components into single validation gate called before every trade.

### Approach Summary

```python
class RiskEngine:
    """Central risk validation"""

    def validate_entry(
        self,
        signal: StrategySignal,
        symbol: str,
        equity_usd: float,
        open_positions: List[Position]
    ) -> tuple[bool, str]:
        """
        Validate all risk constraints.

        Returns: (approved, rejection_reason)
        """
        # 1. Position sizing
        position_size = self.position_sizing.calculate(...)

        # 2. Daily/weekly stop check
        if self.kill_switch.is_active():
            return False, "Daily/weekly stop active"

        # 3. Open risk limit
        if not self.risk_limits.check_open_risk(position_size, open_positions, equity_usd):
            return False, "Max open risk exceeded"

        # 4. Max positions
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            return False, "Max positions reached"

        # 5. Correlation filter
        if not self.correlation_filter.check(symbol, signal.side, open_positions):
            return False, "Correlation filter blocked"

        # 6. Direction limit
        same_direction_count = count_same_direction(open_positions, signal.side)
        if same_direction_count >= MAX_SAME_DIRECTION:
            return False, "Max same-direction positions reached"

        return True, "Risk checks passed"
```

### Files
- `bot/risk/risk_engine.py` (250 LOC)
- `tests/test_risk_engine_integration.py` (400 LOC)

### Estimated Effort: 1 day, ⭐⭐ Medium complexity

---

# MILESTONE 5: EXECUTION ENGINE

## 🎯 Milestone 5 Overview

**Objective:** Implement order lifecycle management from entry to exit with idempotency, fill handling, and trailing stops.

**Deliverables:**
- Order placement with TTL and retry logic
- Fill tracking (full and partial fills)
- Stop/TP order management
- Trailing stop updates
- Emergency exit (market orders)

**Estimated Timeline:** 6-8 days (3-4 days with parallelization)
**Estimated LOC:** 2,000-2,500 lines

---

## Task 10: Order Lifecycle & Fill Handling

### Goal
Implement LIMIT order placement with TTL expiry, retry logic, and idempotency checks.

### Approach Summary

**Order State Machine:**
```
NEW → SUBMITTED → (FILLED | PARTIALLY_FILLED | CANCELED | REJECTED)
```

**Idempotency:**
```python
def place_order(..., client_order_id: str):
    # Check if order already exists
    existing = exchange.fetch_open_orders(symbol)
    if any(o.client_order_id == client_order_id for o in existing):
        return existing_order  # Don't place duplicate

    # Place new order
    order = exchange.create_limit_order(...)
    return order
```

**TTL & Retry:**
```python
# Place LIMIT order with 30s TTL
order = place_limit_order(price=mid, ttl=30)
sleep(30)

# Check fill status
if order.status == "open":
    cancel_order(order.id)

    # Retry once
    if retry_count < LIMIT_RETRY_COUNT:
        order = place_limit_order(price=mid, ttl=30)
```

**Partial Fills:**
```python
if order.filled < order.amount:
    # Cancel remaining
    cancel_order(order.id)

    # Create position from partial fill
    create_position(filled_qty=order.filled, avg_price=order.avg_fill_price)
```

### Files
- `bot/execution/order_manager.py` (400 LOC)
- `bot/execution/order_lifecycle.py` (300 LOC)
- `tests/test_order_lifecycle.py` (600 LOC)

### Estimated Effort: 3 days, ⭐⭐⭐ High complexity

---

## Task 11: Trailing & Advanced Exit Logic

### Goal
Implement ATR-based trailing stops that update at 5m closes or on significant price moves.

### Approach Summary

**Trailing Logic:**
```python
class TrailingStopManager:
    def update_trailing_stop(self, position: Position, current_price: float, atr: float):
        """Update trailing stop if profit >= trail_after_r"""

        # Check if trailing enabled
        if position.pnl_r < position.trail_after_r:
            return  # Not profitable enough yet

        # Calculate new trailing stop
        trail_distance = atr * position.atr_trail_mult

        if position.side == OrderSide.LONG:
            new_stop = current_price - trail_distance
            # Only move stop up (never down)
            if new_stop > position.stop_price:
                update_stop_order(position, new_stop)
        else:
            new_stop = current_price + trail_distance
            # Only move stop down (never up)
            if new_stop < position.stop_price:
                update_stop_order(position, new_stop)
```

**Exit Reasons:**
- TP: Take profit hit
- SL: Stop loss hit
- TRAIL: Trailing stop hit
- KILL_SWITCH: Emergency exit
- MANUAL: User-initiated close
- TIMEOUT: Max holding period (optional)

### Files
- `bot/execution/exit_manager.py` (300 LOC)
- `bot/execution/trailing_stop.py` (200 LOC)
- `tests/test_trailing_stops.py` (500 LOC)

### Estimated Effort: 2-3 days, ⭐⭐ Medium complexity

---

## Task 12: Execution Integration Testing

### Goal
Test complete order lifecycle with mock exchange responses.

### Testing Scenarios
1. LIMIT order placed → filled → stop/TP orders placed
2. LIMIT order placed → TTL expires → cancelled → retry → filled
3. LIMIT order partially filled → partial position created
4. Trailing stop enabled → profit exceeds 1.0R → stop updates
5. Kill switch activated → emergency market close

### Files
- `tests/test_execution_integration.py` (700 LOC)

### Estimated Effort: 1-2 days, ⭐⭐ Medium complexity

---

# MILESTONE 6: STATE MANAGEMENT

## Task 13: Position & Order Reconciliation

### Goal
On startup, fetch open positions/orders from exchange and rebuild internal state.

### Approach
```python
def reconcile_on_startup():
    # 1. Fetch from exchange
    open_positions = exchange.fetch_positions()
    open_orders = exchange.fetch_open_orders()
    recent_trades = exchange.fetch_my_trades(limit=100)

    # 2. Build internal state
    for position in open_positions:
        internal_position = Position.from_exchange(position)
        state_manager.add_position(internal_position)

    # 3. Match orders to positions
    for order in open_orders:
        if is_stop_or_tp_order(order):
            link_order_to_position(order)
        else:
            # Orphan order - cancel
            exchange.cancel_order(order.id)

    # 4. Rebuild performance tracker from trade history
    for trade in recent_trades:
        performance_tracker.add_trade(...)
```

### Files
- `bot/state/reconciler.py` (350 LOC)
- `tests/test_reconciliation.py` (500 LOC)

### Estimated Effort: 2 days, ⭐⭐⭐ High complexity

---

## Task 14: Persistence & Recovery

### Goal
Persist trade logs and event logs to disk for recovery after crashes.

### Approach
- Trade log: JSONL format, one record per closed trade
- Event log: JSONL format, one record per significant event
- On restart: rebuild state from logs
- Log rotation: daily files

### Files
- `bot/state/logger.py` (250 LOC)
- `bot/state/log_reader.py` (200 LOC)
- `tests/test_persistence.py` (400 LOC)

### Estimated Effort: 1-2 days, ⭐⭐ Medium complexity

---

# MILESTONE 7: LOGGING & MONITORING

## Task 15: Trade & Event Logging

### Goal
Implement JSONL logging per LOG_SCHEMA.json specification.

### Trade Log Fields
```json
{
  "trade_id": "T20240211_001",
  "timestamp_open": "2024-02-11T10:30:00Z",
  "timestamp_close": "2024-02-11T11:45:00Z",
  "mode": "PAPER_LIVE",
  "symbol": "BTCUSDT",
  "strategy": "TREND_PULLBACK",
  "regime": "TREND",
  "direction": "LONG",
  "confidence_score": 0.75,
  "entry_order": {...},
  "risk": {...},
  "costs": {...},
  "result": {...},
  "portfolio": {...}
}
```

### Files
- `bot/reporting/trade_logger.py` (300 LOC)
- `tests/test_logging.py` (400 LOC)

### Estimated Effort: 1-2 days, ⭐ Low complexity

---

## Task 16: Daily Reporting & Notifications

### Goal
Generate daily summary reports and send via Telegram.

### Daily Report Contents
- Realized PnL (USD + %)
- Unrealized PnL
- Equity (current + max)
- Drawdown (daily, weekly, max)
- Trade count, win rate, expectancy_R
- Fees, funding totals
- Strategy performance breakdown

### Files
- `bot/reporting/daily_report.py` (300 LOC)
- `bot/reporting/notifier.py` (200 LOC)
- `tests/test_reporting.py` (400 LOC)

### Estimated Effort: 1-2 days, ⭐ Low complexity

---

## Task 17: Health Monitor & Safe Mode

### Goal
Detect unhealthy conditions and enter safe mode (no new trades).

### Safe Mode Triggers
- Binance timestamp errors (persistent)
- Rate limit errors (repeated)
- WebSocket data stale (>30s)
- Balance fetch fails
- Unexpected exceptions

### Safe Mode Behavior
- Block new entries
- Continue monitoring existing positions
- Log at WARNING level
- Send Telegram alert
- Exit safe mode after 60s of healthy checks

### Files
- `bot/health/health_monitor.py` (300 LOC)
- `bot/health/safe_mode.py` (200 LOC)
- `tests/test_health_monitor.py` (400 LOC)

### Estimated Effort: 2 days, ⭐⭐ Medium complexity

---

## Task 18: Scheduler & Event Loop

### Goal
Implement 5m candle close event loop and daily task scheduling.

### Event Schedule
- **Every 5m:** On candle close → decision pipeline
- **00:00 UTC:** Refresh universe
- **00:05 UTC:** Send daily report
- **Monday 00:00:** Reset weekly PnL window

### Files
- `bot/core/scheduler.py` (300 LOC)
- `tests/test_scheduler.py` (400 LOC)

### Estimated Effort: 1-2 days, ⭐⭐ Medium complexity

---

# MILESTONE 8: INTEGRATION TESTING

## Task 19-23: E2E Test Scenarios

### Task 19: E2E Happy Path (2 days, 600 LOC)
- 30-day simulation with realistic data
- All components integrated
- No crashes, all trades logged

### Task 20: Kill Switch Tests (1 day, 400 LOC)
- Daily stop, weekly stop, open risk limit
- Correlation filter, direction limit
- Verify rejection reasons logged

### Task 21: Recovery Tests (1 day, 400 LOC)
- Crash during entry, after fill, during exit
- Verify reconciliation correct
- No duplicate orders

### Task 22: Correlation Tests (1 day, 400 LOC)
- High correlation symbols
- Same-direction blocked
- Hedging allowed with low correlation

### Task 23: Acceptance Suite (1 day, 500 LOC)
- All BOT_SPEC_FINAL.md criteria
- 7-day stability
- Configuration-driven behavior

**Milestone 8 Total:** 8-10 days → 3-4 days parallel

---

# MILESTONE 9: PAPER LIVE VALIDATION

## Task 24: Paper Live Setup (1 day)
- Configure for PAPER_LIVE mode
- Connect to real Binance API
- Enable Telegram alerts
- Start bot

## Task 25: Paper Live Execution (7 days)
- Run continuously for 7 calendar days
- Monitor logs for errors
- Track P&L daily
- Verify daily reports sent

## Task 26: Paper Live Analysis (1 day)
- Calculate realized/unrealized P&L
- Analyze regime distribution
- Identify best/worst strategies
- Document findings
- Go/no-go decision for live

**Milestone 9 Total:** 9 days calendar (mostly monitoring)

---

# MILESTONE 10: MICRO LIVE DEPLOYMENT

## Task 27: Micro Live Deployment (30+ days)
- Deploy to VPS with Docker
- Use 10-20% of capital
- Risk 0.5% per trade (reduced)
- Monitor continuously
- Weekly reviews

## Task 28: Micro Live Analysis (1 day)
- Compare live vs paper metrics
- Validate slippage assumptions
- Check correlation filter behavior
- Team decision: scale or investigate

**Milestone 10 Total:** 30+ days calendar

---

# MILESTONE 11: PRODUCTION OPERATIONS

## Task 29: Parameter Tuning & Optimization (5-7 days)
- Implement walk-forward optimization
- Tune: stop distance, target R, RSI thresholds, ADX thresholds
- Do NOT tune: risk %, kill switches, max positions
- Data split: 60% IS, 20% Val, 20% OOS
- Stress tests: +50% slippage, +50% fees
- Approval criteria: OOS expectancy > 0, max DD < 20%

---

# APPENDIX A: CRITICAL PATH & DEPENDENCIES

## Critical Path (Sequential Tasks)

```
Task 1 (Complete)
  ↓
Task 2 (Regime Detection) - 2-3 days
  ↓
Task 4 (Strategy Selection) - 2-3 days
  ↓
Task 5 (M3 Integration) - 2 days
  ↓
Task 6 (Position Sizing) - 1 day
  ↓
Task 10 (Order Manager) - 3 days
  ↓
Task 13 (Reconciliation) - 2 days
  ↓
Task 15 (Logging) - 1-2 days
  ↓
Task 18 (Scheduler) - 1-2 days
  ↓
Task 23 (Acceptance Tests) - 1 day
  ↓
Task 24-26 (Paper Live) - 9 days
  ↓
Task 27-28 (Micro Live) - 30+ days
```

**Total Critical Path:** ~25-30 days development + 40 days validation = 65-70 days end-to-end

## Parallelization Opportunities

**Group 1 (After Task 5):** Tasks 6, 7, 8, 9 (Risk Engine) - 6-8 days → 2-3 days parallel

**Group 2 (After Task 9):** Tasks 10, 11, 13, 14 - 8-10 days → 3-4 days parallel

**Group 3 (After Task 14):** Tasks 15, 16, 17 - 6-8 days → 2-3 days parallel

**Group 4 (After Task 18):** Tasks 19, 20, 21, 22 - 8-10 days → 3-4 days parallel

**Total with Parallelization:** ~15-20 days development + 40 days validation = 55-60 days

---

# APPENDIX B: PROGRESS TRACKING TEMPLATE

## Daily Progress Log

```markdown
### Date: YYYY-MM-DD
**Task:** Task N - [Task Name]
**Status:** In Progress | Completed | Blocked
**Progress:** [Brief description]
**LOC Written:** XXX lines (code) + YYY lines (tests)
**Tests Passing:** XX / YY
**Blockers:** [Any blockers or issues]
**Next Steps:** [What's next]
```

## Weekly Summary Template

```markdown
### Week N: [Date Range]
**Tasks Completed:** Task X, Task Y, Task Z
**Total LOC:** XXXX lines (code) + YYYY lines (tests)
**Test Coverage:** XX%
**Milestones Reached:** [Milestone names]
**Challenges:** [Issues encountered]
**Learnings:** [Key insights]
**Next Week Plan:** [Tasks planned]
```

## Milestone Completion Checklist

```markdown
## Milestone N: [Name] - COMPLETED ✅

### Acceptance Criteria Status
- [x] Criterion 1
- [x] Criterion 2
- [x] Criterion 3

### Deliverables
- [x] File 1 (XXX LOC)
- [x] File 2 (YYY LOC)
- [x] Tests (ZZZ LOC)
- [x] Documentation

### Metrics
- **Total LOC:** XXXX
- **Test Coverage:** XX%
- **Tests Passing:** All (XXX tests)
- **Duration:** X days
- **Complexity:** Matched estimate

### Issues Resolved
1. Issue 1 - Resolution
2. Issue 2 - Resolution

### Lessons Learned
- Lesson 1
- Lesson 2

### Handoff Notes for Next Milestone
- [Critical information for next phase]
```

---

# FINAL NOTES FOR EXECUTOR CLAUDE

## Execution Principles

1. **Follow the Pattern:** Every completed task (Task 1: Universe Selection) establishes the quality bar. Replicate this pattern exactly.

2. **Test First:** Write tests before or alongside implementation. Aim for >80% coverage on critical paths.

3. **Configuration-Driven:** Zero hardcoded values. Every parameter must come from config.

4. **Type Hints & Docstrings:** 100% coverage required. Use modern Python 3.11+ syntax.

5. **Logging:** Use module-level loggers, log at appropriate levels (DEBUG, INFO, WARNING, ERROR).

6. **Error Handling:** Custom exceptions, exponential backoff for retries, graceful degradation to safe mode.

7. **State Recovery:** Idempotent operations, deterministic clientOrderIds, comprehensive reconciliation.

## When Stuck

1. **Refer to BOT_SPEC_FINAL.md:** This is the source of truth for all requirements.

2. **Check Reference Implementation:** bot/universe/ module shows the exact patterns to follow.

3. **Consult CONFIG.example.json:** All parameters are defined here with defaults.

4. **Review LOG_SCHEMA.json:** Trade logging schema is fully specified.

5. **Ask Questions:** If truly uncertain, prefer safe/conservative interpretation aligned with capital preservation.

## Quality Gates

Before marking any task complete:

- [ ] All unit tests passing
- [ ] Integration tests passing (if applicable)
- [ ] Type hints 100%
- [ ] Docstrings 100%
- [ ] No hardcoded values
- [ ] Error handling comprehensive
- [ ] Logging implemented
- [ ] Configuration-driven
- [ ] Code reviewed against spec
- [ ] Edge cases tested

## Success Criteria

This project is successful when:

1. **Paper live runs 7 days without crash** ✓
2. **All risk controls prevent catastrophic loss** ✓
3. **State recovery works after any crash** ✓
4. **Logs are complete and parseable** ✓
5. **Configuration changes are reflected in behavior** ✓
6. **Micro live performance matches paper predictions** ✓

## Remember

> "This is not a toy trading script. This is a risk-controlled, institutional-grade trading framework. Risk control > performance. Stability > aggressiveness." - AI_HANDOFF.md

Good luck! 🚀

---

**END OF IMPLEMENTATION PLAN**

**Document Stats:**
- **Total Words:** ~18,000
- **Total Sections:** 15
- **Tasks Covered:** 29 (1 complete, 28 planned)
- **Estimated Project Duration:** 55-70 days
- **Estimated Total LOC:** 14,000-18,000 (code) + 6,000-8,000 (tests)

