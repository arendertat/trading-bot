# Binance USDT-M Futures Trading Bot

Production-grade automated trading bot for Binance USDT-M Perpetual Futures with strict risk controls.

**⚠️ RISK DISCLAIMER**: This bot trades cryptocurrency futures with leverage. Trading futures carries substantial risk of loss. Past performance does not guarantee future results. Only use this bot with capital you can afford to lose. Always test thoroughly in PAPER_LIVE mode before going live. The authors are not responsible for any financial losses.

---

## Milestone 1: Configuration Foundation

This is Milestone 1 of the implementation. At this stage, the bot includes:

- ✅ Complete project skeleton
- ✅ Typed configuration system with Pydantic validation
- ✅ Environment variable substitution
- ✅ Cross-field validation
- ✅ Basic logging setup
- ✅ Comprehensive unit tests

**Note**: Trading logic (exchange connectivity, strategies, risk engine, execution) will be implemented in subsequent milestones.

---

## Installation

### Requirements
- Python 3.11+
- pip

### Setup

1. **Clone the repository** (or navigate to the project directory)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create configuration files**:
   ```bash
   # Copy environment template
   cp .env.example .env

   # Edit .env with your credentials
   nano .env

   # Copy config template
   cp config/config.example.json config/config.json

   # Edit config with your parameters
   nano config/config.json
   ```

---

## Configuration

### Environment Variables (.env)

```bash
# Binance API Credentials
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

### Configuration File (config/config.json)

The configuration file uses **decimal notation** for percentages:
- `0.01` = 1%
- `0.025` = 2.5%
- `0.0005` = 0.05%

Key parameters:

#### Mode
- `"mode": "PAPER_LIVE"` - Paper trading (simulated)
- `"mode": "LIVE"` - Live trading (real money)

#### Risk Settings
- `risk_per_trade_pct`: 0.01 = 1% risk per trade
- `max_total_open_risk_pct`: 0.025 = 2.5% total open risk
- `max_open_positions`: Maximum concurrent positions (default: 2)
- `daily_stop_pct`: -0.04 = -4% daily stop loss
- `weekly_stop_pct`: -0.1 = -10% weekly stop loss

#### Leverage
- `trend`: 2.0x max for trend strategies
- `range`: 1.5x max for range strategies
- `high_vol`: 1.0x max for high volatility

See [config/config.example.json](config/config.example.json) for full parameter documentation.

---

## Usage

### Validate Configuration

Before running the bot, validate your configuration:

```bash
python -m bot.config.loader config/config.json
```

Expected output:
```
✓ Config validation successful
  Mode: PAPER_LIVE
  Exchange: binance
  Max positions: 2
  Risk per trade: 1.00%
```

### Run the Bot (Milestone 1)

```bash
python -m bot.main
```

Expected output:
```
Loading config from config/config.json...
2024-01-01 00:00:00 - trading_bot - INFO - ============================================================
2024-01-01 00:00:00 - trading_bot - INFO - Binance USDT-M Futures Trading Bot - Milestone 1
2024-01-01 00:00:00 - trading_bot - INFO - ============================================================
2024-01-01 00:00:00 - trading_bot - INFO - Mode: PAPER_LIVE
2024-01-01 00:00:00 - trading_bot - INFO - Exchange: binance
2024-01-01 00:00:00 - trading_bot - INFO - Margin Mode: ISOLATED
2024-01-01 00:00:00 - trading_bot - INFO - Max Open Positions: 2
2024-01-01 00:00:00 - trading_bot - INFO - Risk Per Trade: 1.00%
2024-01-01 00:00:00 - trading_bot - INFO - ✓ Config validation successful
2024-01-01 00:00:00 - trading_bot - INFO - Note: Trading logic not yet implemented (Milestone 2+)
2024-01-01 00:00:00 - trading_bot - INFO - Exiting gracefully.
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=bot --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Run Specific Test File

```bash
pytest tests/test_config.py -v
```

---

## Project Structure

```
trading-bot/
├── bot/                        # Main bot package
│   ├── config/                 # Configuration management
│   │   ├── models.py           # Pydantic config models
│   │   ├── loader.py           # Config loading + env substitution
│   │   └── validator.py        # Additional validation
│   ├── core/                   # Core types and constants
│   │   ├── constants.py        # Enums (BotMode, RegimeType, etc.)
│   │   └── types.py            # Dataclasses (Position, Order, etc.)
│   ├── utils/                  # Utility functions
│   │   └── logger.py           # Structured logging
│   ├── data/                   # Data & features (Milestone 3)
│   ├── exchange/               # Exchange layer (Milestone 2)
│   ├── regime/                 # Regime detection (Milestone 4)
│   ├── strategies/             # Trading strategies (Milestone 5)
│   ├── risk/                   # Risk engine (Milestone 6)
│   ├── execution/              # Order execution (Milestone 7)
│   ├── state/                  # State management (Milestone 8)
│   ├── reporting/              # Reporting (Milestone 9)
│   └── main.py                 # Entry point
├── config/                     # Configuration files
│   └── config.example.json
├── tests/                      # Test suite
│   └── test_config.py
├── spec/                       # Specification documents
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Configuration Validation Rules

The config system enforces these constraints:

### Risk Constraints
- `risk_per_trade_pct` must be > 0 and <= 10%
- `max_total_open_risk_pct` must be >= `risk_per_trade_pct`
- `max_same_direction_positions` <= `max_open_positions`
- `reduced_risk_after_pause_pct` < `risk_per_trade_pct`
- `daily_stop_pct` must be less severe than `weekly_stop_pct`
- `hedge_corr_max` < `correlation_threshold`

### Regime Constraints
- `range_adx_max` < `trend_adx_min` (no overlap)
- `bb_width_range_min` < `bb_width_range_max`

### Strategy Constraints
- At least one strategy must be enabled
- RSI ranges must be valid (min < max)
- Leverage cannot exceed 2.0x (per spec)

### Exchange Constraints
- Only `ISOLATED` margin mode supported
- Only `binance` exchange supported

---

## Next Steps

Upcoming milestones:

- **Milestone 2**: Exchange layer (Binance API, WebSocket)
- **Milestone 3**: Data & feature engine (indicators, universe filter)
- **Milestone 4**: Regime detection
- **Milestone 5**: Strategy classes (Trend Pullback, Breakout, Range Reversion)
- **Milestone 6**: Risk engine (position sizing, correlation filter, kill switch)
- **Milestone 7**: Execution engine (order lifecycle, trailing stops)
- **Milestone 8**: State management (reconciliation, persistence)
- **Milestone 9**: Reporting & notifications (Telegram, daily reports)
- **Milestone 10**: Safe mode & health monitor
- **Milestone 11**: Paper mode simulation
- **Milestone 12**: Main event loop integration
- **Milestone 13**: Full test suite
- **Milestone 14**: Docker deployment

---

## Troubleshooting

### Config Validation Fails

**Error**: `FileNotFoundError: Config file not found`
- Ensure `config/config.json` exists
- Copy from `config/config.example.json` if needed

**Error**: `Environment variable BINANCE_API_KEY not found`
- Ensure `.env` file exists with valid API credentials
- Check environment variables are correctly formatted

**Error**: `max_same_direction_positions cannot exceed max_open_positions`
- Review risk constraints in config
- Ensure `max_same_direction_positions` <= `max_open_positions`

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'pydantic'`
- Run `pip install -r requirements.txt`
- Ensure Python 3.11+ is installed

---

## License

This project is for educational and research purposes. Use at your own risk.

---

## Contact

For questions about the bot specification, refer to:
- [spec/BOT_SPEC_FINAL.md](spec/BOT_SPEC_FINAL.md)
- [spec/AI_HANDOFF.md](spec/AI_HANDOFF.md)
