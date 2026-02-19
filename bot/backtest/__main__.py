"""
CLI entry point for the backtest engine.

Usage:
    python -m bot.backtest \\
        --config config/config.json \\
        --symbols BTCUSDT ETHUSDT \\
        --start 2024-01-01 \\
        --end   2024-06-01 \\
        --equity 10000

Options:
    --config   Path to config JSON file (default: config/config.json)
    --symbols  Space-separated list of symbols (default: from config whitelist or BTCUSDT)
    --start    Start date YYYY-MM-DD (default: 90 days ago)
    --end      End date YYYY-MM-DD   (default: today)
    --equity   Starting equity in USD (default: 10000)
    --output   Optional path to write JSON report
    --verbose  Enable DEBUG logging
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m bot.backtest",
        description="Run backtest against historical Binance data",
    )
    parser.add_argument(
        "--config",
        default="config/config.json",
        help="Path to config JSON (default: config/config.json)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to backtest e.g. BTCUSDT ETHUSDT",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date YYYY-MM-DD (default: 90 days ago)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=10_000.0,
        help="Starting equity in USD (default: 10000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional file path to save JSON report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("trading_bot.backtest.cli")

    # ── Load config ────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        print("  Hint: cp config/config.example.json config/config.json", file=sys.stderr)
        sys.exit(1)

    try:
        from bot.config.loader import load_config
        config = load_config(str(config_path))
    except Exception as e:
        print(f"[ERROR] Config load failed: {e}", file=sys.stderr)
        sys.exit(1)

    logger.info(f"Config loaded: {config_path}")

    # ── Resolve symbols ────────────────────────────────────────────────
    symbols = args.symbols
    if not symbols:
        symbols = list(config.universe.whitelist) if config.universe.whitelist else ["BTCUSDT"]
    logger.info(f"Symbols: {symbols}")

    # ── Resolve dates ──────────────────────────────────────────────────
    now_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if args.start:
        try:
            start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"[ERROR] Invalid start date: {args.start} (expected YYYY-MM-DD)", file=sys.stderr)
            sys.exit(1)
    else:
        start = now_utc - timedelta(days=90)

    if args.end:
        try:
            end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"[ERROR] Invalid end date: {args.end} (expected YYYY-MM-DD)", file=sys.stderr)
            sys.exit(1)
    else:
        end = now_utc

    if end <= start:
        print("[ERROR] --end must be after --start", file=sys.stderr)
        sys.exit(1)

    days = (end - start).days
    logger.info(
        f"Date range: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} ({days} days)"
    )
    logger.info(f"Starting equity: {args.equity:,.0f} USD")

    # ── Build exchange client ──────────────────────────────────────────
    try:
        import os
        from bot.exchange.binance_client import BinanceFuturesClient
        api_key = os.environ.get(config.exchange.api_key_env, "")
        api_secret = os.environ.get(config.exchange.api_secret_env, "")
        if not api_key or not api_secret:
            raise ValueError(
                f"API credentials not found in environment. "
                f"Set {config.exchange.api_key_env} and {config.exchange.api_secret_env} in .env"
            )
        client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=config.exchange.testnet,
            recv_window_ms=config.exchange.recv_window_ms,
        )
        exchange = client.exchange
        logger.info("Exchange connected")
    except Exception as e:
        print(f"[ERROR] Exchange connection failed: {e}", file=sys.stderr)
        print(
            "  Hint: Check BINANCE_API_KEY / BINANCE_API_SECRET in .env, "
            "or set exchange.testnet=true in config.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Run backtest ───────────────────────────────────────────────────
    try:
        from bot.backtest.engine import BacktestEngine
        engine = BacktestEngine(config, exchange, initial_equity=args.equity)
        run_started_at = datetime.now(timezone.utc)
        result = engine.run(symbols=symbols, start=start, end=end)
        run_ended_at = datetime.now(timezone.utc)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Backtest aborted by user.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        sys.exit(1)

    # ── Build and print report ─────────────────────────────────────────
    from bot.backtest.reporter import build_report
    report = build_report(result)
    report.print_summary()

    # ── Optional JSON output ───────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _save_report_json(
                report=report,
                result=result,
                path=output_path,
                config=config,
                run_started_at=run_started_at,
                run_ended_at=run_ended_at,
                config_path=str(config_path),
            )
            logger.info(f"Report saved to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save report: {e}")


def _save_report_json(
    report,
    result,
    path: Path,
    config,
    run_started_at: datetime,
    run_ended_at: datetime,
    config_path: str,
) -> None:
    """Serialise report + raw trades to JSON."""
    import dataclasses
    import subprocess

    def _to_serialisable(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            d = dataclasses.asdict(obj)
            return _to_serialisable(d)
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serialisable(v) for v in obj]
        if isinstance(obj, tuple):
            return [_to_serialisable(v) for v in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return obj

    def _get_git_commit() -> str | None:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out or None
        except Exception:
            return None

    run_meta = {
        "run_id": f"bt-{run_started_at.strftime('%Y%m%d-%H%M%S')}",
        "started_at": run_started_at.isoformat(),
        "ended_at": run_ended_at.isoformat(),
        "symbols": result.symbols,
        "start": result.start.isoformat(),
        "end": result.end.isoformat(),
        "timeframes": config.timeframes.model_dump(),
        "initial_equity": result.initial_equity,
        "exchange": {
            "name": config.exchange.name,
            "testnet": config.exchange.testnet,
            "margin_mode": config.exchange.margin_mode,
            "recv_window_ms": config.exchange.recv_window_ms,
        },
        "fee_model": {
            "maker_fee_pct": config.execution.maker_fee_pct,
            "taker_fee_pct": config.execution.taker_fee_pct,
        },
        "slippage_model": {
            "paper_slippage_limit_pct": config.execution.paper_slippage_limit_pct,
            "paper_slippage_market_pct": config.execution.paper_slippage_market_pct,
            "paper_slippage_stop_pct": config.execution.paper_slippage_stop_pct,
        },
        "git_commit": _get_git_commit(),
        "config_path": config_path,
    }

    config_snapshot = {
        "timeframes": config.timeframes.model_dump(),
        "risk": config.risk.model_dump(),
        "strategies": config.strategies.model_dump(),
        "universe": config.universe.model_dump(),
    }

    payload = {
        "run": run_meta,
        "config_snapshot": config_snapshot,
        "diagnostics": result.data_quality,
        "report": _to_serialisable(report),
        "trades": _to_serialisable(result.trades),
        "equity_curve": result.equity_curve,
        "features_at_entry": result.features_at_entry,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


if __name__ == "__main__":
    main()
