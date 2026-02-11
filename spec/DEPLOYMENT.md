# Deployment Guide (VPS + Docker)

## Requirements
- Ubuntu VPS (1 vCPU, 1GB RAM is enough)
- Docker + Docker Compose
- Python app inside container

## Security
- Binance API key: trade-only, withdrawals disabled
- IP whitelist to VPS IP
- Secrets in env vars, not in repo

## Run Modes
- PAPER_LIVE first (recommended 2–4 weeks)
- Then LIVE with small capital allocation (e.g., 200–300 USDT)

## Ops
- Systemd or Docker restart policy: always
- Log rotation enabled
- Telegram alerts configured

## Health
- Bot exposes a /health endpoint or writes heartbeat file
- External watchdog can restart container if heartbeat stops
