# Acceptance Tests

## Paper Live Stability
- Run PAPER_LIVE 7 days without crash.
- Verify daily report arrives at configured time.

## Recovery
- Start bot, open a simulated position, restart process.
- Bot must reconcile positions/orders from exchange (live) or from simulator state (paper) without duplicating.

## Risk Rules
- Simulate losses to hit daily stop -> bot stops new entries.
- Simulate losses to hit weekly stop -> pause 7 days (no live entries).

## Order Lifecycle
- Entry limit order TTL expiry -> cancel + retry exactly once.
- Partial fill -> cancel rest and manage partial as open position.

## Correlation Gate
- Create two symbols with corr > threshold -> second same-direction entry must be rejected.

## Logging
- Every trade produces one JSONL log record matching schema.
