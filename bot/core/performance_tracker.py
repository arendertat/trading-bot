"""Performance tracking for strategy selection"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from bot.core.types import Trade

logger = logging.getLogger("trading_bot.performance")


@dataclass
class StrategyMetrics:
    """
    Rolling performance metrics for a strategy.

    Calculated over a rolling window of recent trades.
    """
    strategy_name: str
    total_trades: int
    win_rate: float  # 0.0 to 1.0
    avg_r: float  # Average R multiple per trade
    expectancy_r: float  # Expected R per trade (win_rate * avg_win_r - loss_rate * avg_loss_r)
    max_drawdown_pct: float  # Max drawdown as percentage
    fees_total: float  # Total fees paid
    funding_total: float  # Total funding paid
    confidence: float = 0.0  # Confidence score for selection (0.0 to 1.0)

    def __post_init__(self):
        """Clamp confidence to [0, 1]"""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class TradeRecord:
    """
    Simplified trade record for performance tracking.

    Stores only essential data needed for metric calculations.
    """
    strategy: str
    pnl_r: float  # P&L in R multiples
    pnl_usd: float  # P&L in USD
    fees: float  # Fees paid in USD
    funding: float  # Funding paid in USD
    timestamp: int  # Unix timestamp


class PerformanceTracker:
    """
    Track rolling performance metrics per strategy.

    Maintains a rolling window of recent trades (default 50) for each strategy
    and calculates performance metrics used for strategy selection.

    Thread-safe: Uses simple data structures, external locking required if
    accessed from multiple threads.
    """

    def __init__(self, window_trades: int = 50):
        """
        Initialize performance tracker.

        Args:
            window_trades: Number of recent trades to track per strategy
        """
        self.window = window_trades
        # Rolling window of trades per strategy (FIFO deque)
        self.trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_trades))
        logger.info(f"PerformanceTracker initialized (window={window_trades} trades)")

    def add_trade(
        self,
        strategy: str,
        pnl_r: float,
        pnl_usd: float,
        fees: float = 0.0,
        funding: float = 0.0,
        timestamp: int = 0
    ) -> None:
        """
        Add a closed trade to the rolling window.

        Automatically evicts oldest trade when window is full (FIFO).

        Args:
            strategy: Strategy name that generated the trade
            pnl_r: P&L in R multiples (risk-adjusted return)
            pnl_usd: P&L in USD
            fees: Trading fees paid in USD
            funding: Funding fees paid in USD
            timestamp: Unix timestamp of trade close
        """
        trade = TradeRecord(
            strategy=strategy,
            pnl_r=pnl_r,
            pnl_usd=pnl_usd,
            fees=fees,
            funding=funding,
            timestamp=timestamp
        )

        self.trades[strategy].append(trade)
        logger.debug(
            f"Trade added: {strategy} | PnL: {pnl_r:.2f}R (${pnl_usd:.2f}) | "
            f"Window: {len(self.trades[strategy])}/{self.window}"
        )

    def add_trade_from_record(self, trade: Trade) -> None:
        """
        Add trade from Trade dataclass.

        Args:
            trade: Complete trade record
        """
        self.add_trade(
            strategy=trade.strategy,
            pnl_r=trade.r_multiple,
            pnl_usd=trade.realized_pnl,
            fees=trade.fees,
            funding=trade.funding,
            timestamp=int(trade.exit_time.timestamp()) if trade.exit_time else 0
        )

    def get_metrics(self, strategy: str) -> Optional[StrategyMetrics]:
        """
        Calculate performance metrics for a strategy.

        Returns None if insufficient trades (< 10) in the window.

        Args:
            strategy: Strategy name

        Returns:
            StrategyMetrics if sufficient data, else None
        """
        trades = list(self.trades[strategy])

        # Minimum trades required for meaningful metrics
        MIN_TRADES = 10
        if len(trades) < MIN_TRADES:
            logger.debug(f"{strategy}: Insufficient trades ({len(trades)} < {MIN_TRADES})")
            return None

        # Extract arrays for calculations
        pnl_r_values = np.array([t.pnl_r for t in trades])
        pnl_usd_values = np.array([t.pnl_usd for t in trades])

        # Win rate calculation
        wins = np.sum(pnl_r_values > 0)
        losses = np.sum(pnl_r_values < 0)
        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Average R calculation
        avg_r = np.mean(pnl_r_values)

        # Expectancy R calculation (more robust than simple avg)
        if wins > 0 and losses > 0:
            avg_win_r = np.mean(pnl_r_values[pnl_r_values > 0])
            avg_loss_r = np.mean(pnl_r_values[pnl_r_values < 0])
            loss_rate = 1.0 - win_rate
            expectancy_r = (win_rate * avg_win_r) + (loss_rate * avg_loss_r)
        else:
            # Not enough data for proper expectancy calculation
            expectancy_r = avg_r

        # Max drawdown calculation (from cumulative P&L)
        cumulative_pnl = np.cumsum(pnl_usd_values)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = cumulative_pnl - running_max
        max_drawdown_usd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

        # Convert to percentage (assuming initial balance reference)
        # Use total positive P&L as reference for percentage calculation
        total_positive_pnl = np.sum(pnl_usd_values[pnl_usd_values > 0])
        max_drawdown_pct = abs(max_drawdown_usd) / total_positive_pnl if total_positive_pnl > 0 else 0.0

        # Fee and funding totals
        fees_total = sum(t.fees for t in trades)
        funding_total = sum(t.funding for t in trades)

        metrics = StrategyMetrics(
            strategy_name=strategy,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_r=avg_r,
            expectancy_r=expectancy_r,
            max_drawdown_pct=max_drawdown_pct,
            fees_total=fees_total,
            funding_total=funding_total
        )

        logger.debug(
            f"{strategy} metrics: WR={win_rate:.2%}, Exp_R={expectancy_r:.3f}, "
            f"DD={max_drawdown_pct:.2%}, Trades={total_trades}"
        )

        return metrics

    def get_all_strategy_names(self) -> List[str]:
        """
        Get list of all strategy names being tracked.

        Returns:
            List of strategy names
        """
        return list(self.trades.keys())

    def get_trade_count(self, strategy: str) -> int:
        """
        Get number of trades in window for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Number of trades in rolling window
        """
        return len(self.trades[strategy])

    def clear_strategy(self, strategy: str) -> None:
        """
        Clear all trades for a strategy.

        Args:
            strategy: Strategy name to clear
        """
        if strategy in self.trades:
            self.trades[strategy].clear()
            logger.info(f"Cleared performance history for {strategy}")

    def rebuild_from_trades(self, trades: List[Trade]) -> None:
        """
        Rebuild performance tracker from historical trades.

        Used on bot startup to restore performance state from trade logs.

        Args:
            trades: List of completed trades (sorted by exit time)
        """
        # Clear existing data
        self.trades.clear()

        # Add trades in chronological order
        for trade in trades:
            if trade.exit_time is not None:
                self.add_trade_from_record(trade)

        logger.info(
            f"Rebuilt performance tracker from {len(trades)} historical trades. "
            f"Strategies: {len(self.trades)}"
        )
