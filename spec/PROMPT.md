# PROMPT — Generate Production-Grade Futures Trading Bot

You are a senior quantitative engineer and production-level Python developer.

Your task is to implement a complete Binance USDT-M Perpetual Futures trading bot according to the attached specification files:

- BOT_SPEC_FINAL.md
- ARCHITECTURE.md
- CONFIG.example.json
- LOG_SCHEMA.json
- ACCEPTANCE_TESTS.md

You MUST strictly follow these specifications.
Do NOT invent new strategies.
Do NOT simplify risk logic.
Do NOT remove safety mechanisms.

---

# Technical Requirements

## Language
- Python 3.11+

## Architecture
- Modular, production-grade structure
- Clear separation of concerns
- Typed functions where possible
- Use dataclasses or pydantic models for structured state

## Suggested Folder Structure

bot/
  config/
  core/
  exchange/
  strategies/
  risk/
  execution/
  state/
  reporting/
  utils/
  main.py

---

# Exchange Layer

- Use ccxt OR official Binance futures API
- Support REST + WebSocket
- Implement:
  - fetch klines
  - fetch funding
  - fetch account balance
  - fetch open positions
  - place order
  - cancel order
  - fetch open orders

All orders must use unique clientOrderId.

---

# Data Handling

- Maintain in-memory candle history per symbol
- Support rolling window for:
  - indicators
  - performance tracking
  - correlation

---

# Regime Engine

- Implement rule-based regime detection exactly as defined
- Output regime + confidence score
- If confidence < threshold → NO TRADE

---

# Strategy Engine

- Each strategy must be a class with:
  - generate_signal()
  - calculate_stop()
  - calculate_target()
  - required_leverage()

- Strategies must NOT know about portfolio risk.
- Risk engine handles sizing.

---

# Risk Engine

Must enforce:

- risk_per_trade_pct
- max_total_open_risk_pct
- max_open_positions
- same-direction limit
- correlation filter
- daily stop
- weekly stop
- pause logic
- reduced risk after pause

If any rule fails → trade rejected.

---

# Execution Engine

- Limit entry with TTL
- Retry exactly once
- Stop-market for stop loss
- Trailing logic when enabled
- Partial fill handling
- Emergency market exit for kill switch

---

# State Management

On startup:
- Fetch positions
- Fetch open orders
- Rebuild internal state
- Ensure idempotency

---

# Logging

- Every trade logged in JSONL format
- Must match LOG_SCHEMA.json
- Daily summary must be generated

---

# Error Handling

Implement:

- Retry with exponential backoff
- Safe mode for repeated API errors
- Health monitor
- Graceful shutdown

---

# Testing

Write basic unit tests for:

- risk calculations
- position sizing
- correlation filter
- regime detection

---

# Constraints

- Do NOT use hardcoded magic numbers.
- All parameters must come from config.
- Code must be readable and production-ready.
- No monolithic scripts.
- No global state.

---

# Deliverables

1. Complete Python project
2. requirements.txt
3. README.md explaining how to:
   - configure
   - run paper mode
   - switch to live mode
4. Example .env file
5. Dockerfile
6. docker-compose.yml

---

# Important

This is not a toy bot.
This must be written like production trading infrastructure.

Follow the specification strictly.
If any part of spec is ambiguous, make the safest possible implementation aligned with risk control.

Do NOT simplify.
