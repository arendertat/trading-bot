# Architecture — Futures Bot (Stabil Growth)

## Modules
1. Config Loader
2. Exchange Client (Binance REST + WebSocket)
3. Market Data Store (candles, book, funding)
4. Feature Engine
5. Universe Selector
6. Regime Detector
7. Strategy Library
8. Strategy Selector (performance-based)
9. Risk Engine
10. Execution Engine (order lifecycle)
11. State Manager (positions/orders reconciliation)
12. Logger + Metrics
13. Notifier (Telegram)
14. Scheduler (5m close events, daily tasks)
15. Health Monitor + Safe Mode Controller

## Main Loop (5m candle close)
1) Pull latest completed 5m candle for each monitored symbol
2) Update features
3) Run universe filter gates (spread/funding/volume)
4) Regime detection
5) For each eligible symbol:
   - Strategy selection (based on regime + rolling performance)
   - Risk checks (open risk, positions, correlation, direction limits)
   - If pass => generate order plan (entry/stop/tp)
6) Execution engine places/cancels orders, handles fills, partial fills
7) Update logs and performance stats on trade close

## Daily Tasks
- 00:00 UTC: refresh universe
- 00:05 UTC: send daily report
- Weekly: reset weekly PnL window

## Error Handling
- API errors -> retry with backoff -> safe mode
- Data stale -> safe mode
- Unexpected exception -> safe mode + alert
