#!/usr/bin/env bash
# ============================================================
# run_paper.sh — Start the bot in PAPER_LIVE mode (Testnet)
# ============================================================
# Usage:
#   chmod +x run_paper.sh
#   ./run_paper.sh
#
# Requirements:
#   - Python 3.9+
#   - .env file with BINANCE_API_KEY and BINANCE_API_SECRET
#     (get testnet keys from https://testnet.binancefuture.com)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check Python ─────────────────────────────────────────────
PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" &>/dev/null; then
    echo "[ERROR] Python not found. Install Python 3.9+"
    exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PY_VERSION"

# ── Check .env ───────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo "[ERROR] .env file not found."
    echo "  → Copy .env.example to .env and fill in your testnet API keys."
    exit 1
fi

# Check API keys are not placeholders
API_KEY=$(grep "^BINANCE_API_KEY=" .env | cut -d= -f2 | tr -d ' ')
if [ -z "$API_KEY" ] || [ "$API_KEY" = "YOUR_TESTNET_API_KEY_HERE" ]; then
    echo "[ERROR] BINANCE_API_KEY is not set in .env"
    echo "  → Get testnet API keys from https://testnet.binancefuture.com"
    exit 1
fi

# ── Check config ──────────────────────────────────────────────
if [ ! -f "config/config.json" ]; then
    echo "[ERROR] config/config.json not found."
    exit 1
fi

# ── Check dependencies ────────────────────────────────────────
echo "Checking dependencies..."
"$PYTHON" -c "import ccxt, pydantic, dotenv" 2>/dev/null || {
    echo "Installing dependencies..."
    "$PYTHON" -m pip install -r requirements.txt -q
}

# ── Create log directory ──────────────────────────────────────
mkdir -p logs

# ── Start bot ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Starting Bot — PAPER_LIVE mode (Binance Testnet)"
echo "  Press Ctrl-C to stop"
echo "============================================================"
echo ""

exec "$PYTHON" -m bot.main "$@"
