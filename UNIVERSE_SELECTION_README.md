# Universe Selection Module - Milestone 3 Task 1

## Overview

The universe selection module (`bot/universe/`) implements daily symbol filtering and ranking for the trading bot.

## Components

### 1. Data Models (`bot/universe/models.py`)

**SymbolEligibility**
- Tracks eligibility check results for a symbol
- Fields:
  - `symbol`: Symbol name (e.g., "BTC/USDT")
  - `pass_volume`: Boolean - passed 24h volume filter
  - `pass_spread`: Boolean - passed spread filter
  - `pass_funding`: Boolean - passed funding rate filter
  - `pass_atr_ratio`: Boolean - passed ATR ratio filter
  - `score`: Float - liquidity/quality score (higher = better)
  - `reasons`: List of failure reasons (empty if eligible)
- Property `is_eligible`: True if all filters pass

### 2. Universe Selector (`bot/universe/selector.py`)

**UniverseSelector**

Main class for building the daily universe.

**Constructor:**
```python
UniverseSelector(
    exchange_client: BinanceFuturesClient,
    candle_store: CandleStore,
    config: UniverseConfig,
)
```

**Key Method:**
```python
def build_daily_universe(now_utc: datetime) -> List[str]:
    """
    Build daily universe of eligible symbols.

    Returns:
        List of top N eligible symbol names (ranked by score)
    """
```

**Filtering Process:**
1. Fetch all USDT-M perpetual symbols from exchange
2. Apply whitelist/blacklist:
   - If whitelist non-empty → only use whitelisted symbols
   - Always exclude blacklisted symbols
3. Fetch market data (tickers, funding rates)
4. Warm up candles for ATR calculation (5m, 20 bars)
5. Apply eligibility filters:
   - 24h quote volume >= `config.universe.min_24h_volume_usdt`
   - Spread <= `config.universe.max_spread_pct`
   - |funding_rate| <= `config.universe.max_abs_funding_rate`
   - ATR(14)/price >= `config.universe.min_atr_ratio`
6. Calculate liquidity/quality score for eligible symbols
7. Rank by score (descending) and return top N

**Scoring Formula:**
```
volume_ratio = quote_volume / min_24h_volume_usdt
spread_penalty = max(spread_pct, 1e-9)
funding_penalty = 1.0 - (abs_funding / max_abs_funding_rate)

score = (volume_ratio / spread_penalty) * funding_penalty
```

Higher volume + lower spread + lower funding rate → higher score

### 3. Exchange Client Extensions (`bot/exchange/binance_client.py`)

Added methods to support universe selection:

- `list_usdtm_perp_symbols() -> List[str]`
  - Lists all active USDT-M perpetual symbols

- `fetch_24h_tickers(symbols: Optional[List[str]]) -> Dict[str, Dict]`
  - Fetches 24h ticker data (volume, bid, ask)

- `fetch_funding_rates(symbols: List[str]) -> Dict[str, float]`
  - Fetches current funding rates for symbols

## Configuration

From `config/config.json` (universe section):

```json
{
  "universe": {
    "min_24h_volume_usdt": 100000000,      // 100M USDT minimum
    "max_spread_pct": 0.0005,              // 0.05% max spread
    "max_abs_funding_rate": 0.0015,        // 0.15% max funding rate
    "min_atr_ratio": 0.005,                // 0.5% min ATR/price
    "max_monitored_symbols": 6,            // Top N symbols to select
    "whitelist": [],                       // Optional whitelist
    "blacklist": [],                       // Optional blacklist
    "hedge_max_combined_funding": 0.0015
  }
}
```

## Usage Example

```python
from datetime import datetime
from bot.exchange.binance_client import BinanceFuturesClient
from bot.data.candle_store import CandleStore
from bot.config.loader import load_config
from bot.universe.selector import UniverseSelector

# Load configuration
config = load_config("config/config.json")

# Initialize components
exchange_client = BinanceFuturesClient.from_config(config.exchange)
candle_store = CandleStore()

# Create universe selector
selector = UniverseSelector(
    exchange_client=exchange_client,
    candle_store=candle_store,
    config=config.universe,
)

# Build daily universe (call this at 00:00 UTC daily)
now_utc = datetime.utcnow()
selected_symbols = selector.build_daily_universe(now_utc)

print(f"Selected {len(selected_symbols)} symbols: {selected_symbols}")
# Example output: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'MATIC/USDT', 'AVAX/USDT', 'LINK/USDT']
```

## Testing

### Running Tests

**Prerequisites:**
```bash
pip install -r requirements.txt
```

**Run all universe tests:**
```bash
./run_universe_tests.sh
```

Or manually:
```bash
pytest tests/test_universe.py -v
```

**Run specific test class:**
```bash
pytest tests/test_universe.py::TestWhitelistBlacklist -v
```

**Run with coverage:**
```bash
pytest tests/test_universe.py --cov=bot.universe --cov-report=term-missing
```

### Test Coverage

The test suite (`tests/test_universe.py`) covers:

#### 1. Data Model Tests
- SymbolEligibility.is_eligible property
- All pass/fail combinations

#### 2. Whitelist/Blacklist Tests
- No whitelist, no blacklist (all symbols pass)
- Whitelist only (only whitelisted symbols)
- Blacklist only (exclude blacklisted symbols)
- Whitelist + blacklist (blacklist takes precedence)
- Empty input handling

#### 3. Volume Filter Tests
- Pass: volume >= threshold
- Fail: volume < threshold
- Edge case: volume exactly at threshold

#### 4. Spread Filter Tests
- Pass: spread <= max
- Fail: spread > max
- Missing bid/ask data

#### 5. Funding Rate Filter Tests
- Pass: positive funding within limit
- Pass: negative funding within limit
- Fail: positive funding exceeds limit
- Fail: negative funding exceeds limit

#### 6. ATR Ratio Filter Tests
- Pass: sufficient volatility
- Fail: insufficient candle data

#### 7. Scoring Tests
- Score determinism (same inputs → same score)
- Higher volume → higher score
- Lower spread → higher score

#### 8. Ranking Tests
- Returns exactly top N symbols
- Ranking is deterministic
- Handles no eligible symbols (returns empty list)

#### 9. Edge Cases
- No ticker data
- Zero bid price
- Empty candidate list after filters

All tests use mocked exchange clients (no network calls).

## Design Decisions

### 1. Daily Refresh
- Universe is built once per day at 00:00 UTC
- Not recalculated on every 5m candle to avoid churn
- Provides stable symbol list for intraday trading

### 2. Deterministic Scoring
- Score formula is fully deterministic (no randomness)
- Same market conditions → same symbol selection
- Reproducible for backtesting

### 3. Strict Filtering
- Symbols must pass ALL filters to be eligible
- Any single failure disqualifies the symbol
- Reasons tracked for debugging

### 4. Whitelist Priority
- If whitelist is non-empty, only those symbols are considered
- Blacklist always applied (even to whitelisted symbols)
- Allows manual override of universe

### 5. ATR Calculation
- Uses 5m timeframe for ATR (not 1h)
- Requires minimum 15 candles (14 + 1 for calculation)
- Warmup fetches 20 candles for safety margin

### 6. Error Handling
- Missing ticker data → symbol ineligible
- Failed ATR calculation → symbol ineligible
- Funding rate fetch errors → default to 0.0 and log warning
- Partial failures don't stop entire universe build

## Integration Notes

### Daily Schedule
The universe selector should be called once per day at 00:00 UTC:

```python
import schedule
import time

def daily_universe_refresh():
    now_utc = datetime.utcnow()
    selected_symbols = selector.build_daily_universe(now_utc)
    # Store selected_symbols for use by other modules

# Schedule daily at 00:00 UTC
schedule.every().day.at("00:00").do(daily_universe_refresh)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Downstream Usage
Other modules (regime detection, strategy selection) will receive the daily symbol list and operate only on those symbols during the trading day.

## Logging

The module logs to `trading_bot.universe` logger:

- **INFO**: Universe build start/completion, symbol counts at each filter stage
- **DEBUG**: Whitelist/blacklist application, per-symbol candle warmup, top symbol scores
- **WARNING**: No candidates after filters, failed candle fetches, missing data

## Future Enhancements (Post-Milestone 3)

- Multi-factor scoring (volatility, trend strength, etc.)
- Adaptive thresholds based on market regime
- Symbol rotation limits (max changes per day)
- Historical eligibility tracking
- Performance-based reweighting
