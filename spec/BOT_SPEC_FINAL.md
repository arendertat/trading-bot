# BOT SPEC FINAL — Binance USDT-M Perpetual Futures Bot (Stabil Growth)

## 0) Goal
Build a 24/7 automated crypto futures trading bot on Binance USDT-M Perpetual that targets stable growth with strict risk controls. The bot uses regime detection and a multi-strategy library. It does NOT invent strategies; it selects among predefined strategies based on recent performance.

## 1) Scope
- Exchange: Binance USDT-M Perpetual
- Account currency: USDT
- Leverage: 1x–2x
- Margin mode: ISOLATED (mandatory)
- Modes: PAPER_LIVE, LIVE
- Primary timeframe: 5m (signals & decisions happen on candle close)
- Trend filter timeframe: 1h
- Default max open positions: 2
- Default hedge behavior: allow long+short only if correlation and cost filters pass

## 2) Non-Goals
- No prediction “magic” or internet-learned strategies
- No martingale, no position averaging down
- No high leverage (>2x)
- No unrestricted multi-coin exposure without correlation & risk gates

## 3) Universe Selection (Symbols)
### 3.1 Daily Universe Filter (runs at 00:00 UTC)
From all USDT-M perpetual symbols, build candidates that pass:
- 24h quote volume >= MIN_24H_VOLUME_USDT
- average top-of-book spread <= MAX_SPREAD_PCT
- abs(funding_rate) <= MAX_ABS_FUNDING_RATE
- ATR(14)/price >= MIN_ATR_RATIO
- symbol is not in blacklist (optional)

### 3.2 Actively Monitored Symbols
- Take top MAX_MONITORED_SYMBOLS by a liquidity/quality score (e.g., volume/spread).
- Trading eligibility is evaluated per-symbol at each decision point.

## 4) Data & Timing
### 4.1 Data sources
- Klines (5m and 1h)
- Best bid/ask (for spread filter)
- Funding rate (for filter + PnL)
- Account endpoints: positions, open orders, balances

### 4.2 Decision timing
- All signals and decisions happen on **5m candle close**.
- The bot must not “peek” inside the forming candle.

## 5) Feature Engine
Compute per symbol:
- RSI(14) on 5m
- ADX(14) on 5m
- ATR(14) on 5m
- EMA20, EMA50 on 5m
- EMA20, EMA50 on 1h (trend filter)
- Bollinger Bands(20, 2) on 5m + BB width
- Volume z-score over last 100 5m candles
- ATR z-score over last 100 5m candles

## 6) Regime Detection (Rule-based, v1)
Regimes:
- TREND
- RANGE
- HIGH_VOL
- CHOP_NO_TRADE

Rules (evaluated at 5m close):
- HIGH_VOL if ATR_Z > HIGH_VOL_ATR_Z
- TREND if ADX > TREND_ADX_MIN and 1h trend filter is directional:
  - bullish if EMA20_1H > EMA50_1H
  - bearish if EMA20_1H < EMA50_1H
- RANGE if ADX < RANGE_ADX_MAX and BB_WIDTH within range thresholds
- CHOP_NO_TRADE if (very low ADX AND very low BB width) OR spread filter fails OR confidence < CONF_THRESHOLD

Regime confidence:
- A simple confidence score is computed (0..1) based on how strongly rules match.
- If max confidence < CONF_THRESHOLD => CHOP_NO_TRADE.

## 7) Strategy Library (v1)
All strategies must define:
- entry_conditions()
- stop_loss_price()
- take_profit_price() or trailing rules
- leverage_mapping(regime)
- default stop distance (%)
- position sizing based on risk engine

### 7.1 Strategy A: Trend Pullback (Primary)
Regime: TREND
Long setup:
- 1h bullish trend
- 5m EMA20 > EMA50 OR price above EMA50
- RSI in PULLBACK_RSI_LONG_RANGE (e.g., 40-50)
- price near EMA20 (within PULLBACK_EMA20_BAND_PCT)
Short setup: symmetric
Stop: STOP_PCT_TREND (default 1.0%)
Take profit: TARGET_R_MULTIPLE_TREND_PULLBACK (default 1.5R)
Trailing: enabled after profit >= 1.0R, trailing distance based on ATR_MULT_TRAIL

### 7.2 Strategy B: Trend Breakout (Secondary)
Regime: TREND
- Break of 20-bar high/low (5m)
- Volume_z > BREAKOUT_VOLUME_Z_MIN
Stop: STOP_PCT_TREND (default 1.0%)
Trailing: enabled immediately with ATR-based trailing

### 7.3 Strategy C: Range Mean Reversion
Regime: RANGE
- RSI < RSI_LONG_EXTREME and BB lower band touch => long
- RSI > RSI_SHORT_EXTREME and BB upper band touch => short
Stop: STOP_PCT_RANGE (0.8–1.0%)
Take profit: TARGET_R_MULTIPLE_RANGE (1.0–1.2R)
No averaging down.

## 8) Strategy Selection (“AI” layer)
The bot maintains rolling performance stats per strategy (and optionally per symbol):
- rolling window size = PERF_WINDOW_TRADES (default 50)
Metrics:
- win_rate
- avg_R
- expectancy_R (avg_R)
- max_drawdown_pct
- fees_total, funding_total
- net_expectancy_R (after fees+funding)

Selection:
- Given current regime, consider only eligible strategies.
- Prefer strategies with net_expectancy_R > 0.
- Score = net_expectancy_R - DD_PENALTY_WEIGHT * max_drawdown_pct
- confidence_score = normalized(score) and must be >= CONF_THRESHOLD
- If no strategy qualifies => NO TRADE.

Stability constraint:
- Do not switch the “preferred strategy per regime” more than once per day unless a kill condition triggers (e.g., expectancy turns negative with sufficient samples).

## 9) Risk Engine (Core)
### 9.1 Per-trade risk
- risk_per_trade_pct default: 1%
- risk_usd = equity_usd * risk_per_trade_pct

### 9.2 Total open risk limit
- max_total_open_risk_pct default: 2.5%
- open_risk_usd is sum of worst-case losses to stop for all open positions
- enforce open_risk_usd <= equity_usd * max_total_open_risk_pct

### 9.3 Daily & weekly stops (Kill switches)
- Daily stop: realized PnL <= DAILY_STOP_PCT => disable new entries until next UTC day
- Weekly stop: realized PnL <= WEEKLY_STOP_PCT => pause trading for PAUSE_DAYS
  - During pause: keep running in PAPER_LIVE monitoring mode, no live orders
  - After pause: risk_per_trade_pct reduced to REDUCED_RISK_AFTER_PAUSE_PCT for REDUCED_RISK_DAYS, then return to normal

### 9.4 Max positions & direction limits
- max_open_positions = 2
- max_same_direction_positions = 2
- Hedging allowed only if correlation filter and cost filter pass.

### 9.5 Correlation filter (Portfolio exposure control)
Compute rolling correlation of 1h returns for last CORR_LOOKBACK_HOURS (default 72):
- If corr(symbolA, symbolB) > CORR_THRESHOLD (default 0.85), treat them as same bucket.
Rules:
- Do not open a new position in same direction within the same bucket if already have one.
- Hedge allowed only if correlations are below HEDGE_CORR_MAX (default 0.60) or the hedge reduces net beta exposure.

## 10) Position Sizing
Given:
- risk_usd (max loss)
- stop_pct distance
Compute:
- notional_usd = risk_usd / stop_pct
- leverage chosen via mapping:
  - TREND: 2.0x
  - RANGE: 1.5x
  - HIGH_VOL: 1.0–1.5x (prefer 1.0x)
- margin_required_usd = notional_usd / leverage
Enforce:
- sufficient free margin
- total open risk limit
- max positions
- direction limits
- correlation filter

## 11) Execution Policy (Order Lifecycle)
### 11.1 Entry
- Default: LIMIT order at a price close to mid (configurable)
- TTL: LIMIT_TTL_SECONDS (default 30)
- Retry: LIMIT_RETRY_COUNT (default 1)
- If still not filled after retries => cancel and skip trade (no market chasing)

### 11.2 Stop loss & take profit
- Immediately after entry fill, place:
  - STOP-MARKET stop order
  - TAKE-PROFIT (limit) or trailing mechanism
- For trailing:
  - Update stop periodically (at 5m closes) or as price moves.
- Emergency exits:
  - On kill switch or system faults: close position with MARKET (or reduce-only market).

### 11.3 Partial fills
- If partially filled at TTL:
  - Cancel remaining
  - Manage open partial as a position (place stop/tp accordingly)

## 12) Fees, Funding, and Slippage
- LIVE: use actual fills, actual fees, funding cashflows
- PAPER:
  - Apply slippage assumptions:
    - limit: PAPER_SLIPPAGE_LIMIT_PCT
    - market: PAPER_SLIPPAGE_MARKET_PCT
    - stop: PAPER_SLIPPAGE_STOP_PCT
  - Apply fee assumptions:
    - maker_fee_pct / taker_fee_pct from config
  - Funding can be approximated by funding_rate * notional over funding periods or ignored in paper if not available

## 13) State, Recovery, and Safety
### 13.1 Reconciliation on startup
On boot / restart:
- Fetch open positions
- Fetch open orders
- Build internal state from exchange as source of truth
- Cancel stale orders if needed
- Ensure no duplicate orders (idempotency)

### 13.2 Idempotency
- Every order must have unique clientOrderId = deterministic (trade_id + role)
- Never submit the same order twice if already exists.

### 13.3 Safe mode triggers
Enter SAFE_MODE (no new trades) if:
- Binance timestamp / recvWindow errors persist
- repeated rate limit errors
- websocket data stale
- equity/balance fetch fails
- unexpected exceptions in strategy/risk engine
SAFE_MODE exits only after health check passes for HEALTHY_STREAK seconds.

## 14) Monitoring & Reporting
- Telegram (or similar) notifications:
  - Daily summary at 00:05 UTC
  - Alerts on: kill switch triggers, errors, safe mode, new position opened/closed
- Daily report includes: realized, unrealized, equity, DD, trade count, win rate, expectancy_R, fees, funding, open risk

## 15) Configuration
All parameters must be configurable via a JSON file:
- risk settings
- universe filters
- strategy params
- execution params
- fee/slippage assumptions
- symbol whitelist/blacklist

## 16) Acceptance tests (must pass)
- Paper live runs for 7 days without crashing
- Restart recovery correctly reconstructs state
- Kill switch stops new entries
- No duplicate orders
- Correlation filter prevents same-bucket stacking
- Logs and daily reports generated
