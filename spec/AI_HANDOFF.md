# AI_HANDOFF.md
# Production Trading Bot – AI Implementation Instructions

## PURPOSE
This document explains how an AI coding system must interpret and use the provided specification files.

The goal is to ensure:
- No ambiguity
- No simplification of risk logic
- No architectural shortcuts
- Strict adherence to the core specification

---

# 1️⃣ DOCUMENT PRIORITY ORDER (SOURCE OF TRUTH)

If any conflicts arise, follow this hierarchy:

1. BOT_SPEC_FINAL.md  ← Absolute source of truth
2. CONFIG.example.json
3. LOG_SCHEMA.json
4. ARCHITECTURE.md
5. ACCEPTANCE_TESTS.md
6. DEPLOYMENT.md
7. Advanced framework documents (methodology only)

If Advanced docs conflict with BOT_SPEC_FINAL.md,
BOT_SPEC_FINAL.md ALWAYS wins.

---

# 2️⃣ IMPLEMENTATION RULES

The AI must:

- Follow modular architecture exactly as described.
- Implement full risk engine before strategy execution.
- Never bypass correlation or risk filters.
- Never hardcode numeric constants.
- Never simplify stop or kill-switch logic.
- Implement state reconciliation on startup.
- Ensure idempotent order submission.
- Support PAPER_LIVE and LIVE modes.

---

# 3️⃣ DEVELOPMENT SEQUENCE (MANDATORY ORDER)

The AI must build the system in this order:

1. Config loader
2. Exchange layer
3. Data + feature engine
4. Regime detection
5. Strategy classes
6. Risk engine
7. Execution engine
8. State management
9. Logging system
10. Reporting
11. Health monitor & safe mode

Risk engine must be implemented BEFORE execution logic.

---

# 4️⃣ NON-NEGOTIABLE SAFETY CONSTRAINTS

The following must NEVER be removed:

- Daily stop
- Weekly stop
- Pause logic
- Reduced risk after pause
- Correlation filter
- Max total open risk cap
- Max same-direction cap
- Slippage modeling in paper mode

---

# 5️⃣ CODE QUALITY REQUIREMENTS

- Python 3.11+
- Type hints required
- No global state
- No monolithic scripts
- Clear module boundaries
- Use dataclasses or pydantic models
- Proper exception handling
- Retry with exponential backoff
- Health monitoring mechanism

---

# 6️⃣ TESTING REQUIREMENTS

The AI must produce:

- Unit tests for risk calculations
- Unit tests for position sizing
- Unit tests for correlation filter
- Unit tests for regime detection
- Simulation test for kill switch logic

---

# 7️⃣ DEPLOYMENT REQUIREMENTS

Deliver:

- Dockerfile
- docker-compose.yml
- requirements.txt
- README.md
- Example .env file

---

# 8️⃣ FINAL NOTE TO AI

This is not a toy trading script.

This is a risk-controlled, institutional-grade trading framework.

If uncertain about implementation detail,
choose the safest possible interpretation aligned with capital preservation.

Risk control > performance.
Stability > aggressiveness.
