"""
Simulated account state for backtesting.

Tracks equity, open positions, closed trades, and PnL.
Mirrors the real ExecPosition interface closely so the same
risk engine and strategy code works unmodified.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from bot.core.constants import OrderSide
from bot.core.types import Position as CorePosition, PositionStatus as CorePositionStatus


@dataclass
class BacktestTrade:
    """A completed trade record."""
    trade_id: str
    symbol: str
    side: str                  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    quantity: float
    notional_usd: float
    leverage_used: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str           # "SL", "TP", "TRAIL", "EOD" (end of data)
    pnl_usd: float
    pnl_r: float               # PnL in R-multiples
    risk_usd: float
    fees_usd: float
    slippage_usd: float
    funding_usd: float
    gross_pnl_usd: float
    net_pnl_usd: float
    strategy: str
    regime: str
    stop_price: float
    tp_price: Optional[float]
    regime_confidence: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    estimated_cost_r: Optional[float] = None
    expected_edge_r: Optional[float] = None
    setup_quality_score: Optional[float] = None
    max_adverse_excursion: float = 0.0   # Worst unrealised loss during trade
    max_favourable_excursion: float = 0.0  # Best unrealised profit during trade
    # Early window stats (first N bars after entry)
    early_window_bars: int = 0
    early_mae_usd: float = 0.0
    early_mfe_usd: float = 0.0
    early_mae_r: float = 0.0
    early_mfe_r: float = 0.0


@dataclass
class OpenBacktestPosition:
    """A position that is currently open in the backtest."""
    trade_id: str
    symbol: str
    side: OrderSide
    entry_price: float
    quantity: float
    notional_usd: float
    risk_usd: float
    leverage_used: float
    stop_price: float
    tp_price: Optional[float]
    entry_time: datetime
    strategy: str
    regime: str
    margin_usd: float = 0.0          # Actual margin locked (notional / leverage)
    trail_enabled: bool = False
    trailing_stop: Optional[float] = None  # Current trailing stop price
    trail_after_r: float = 1.0
    atr_trail_mult: float = 2.0
    initial_stop_price: float = 0.0
    fees_usd: float = 0.0
    entry_price_raw: float = 0.0
    entry_slippage_usd: float = 0.0
    regime_confidence: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    estimated_cost_r: Optional[float] = None
    expected_edge_r: Optional[float] = None
    setup_quality_score: Optional[float] = None
    max_adverse_excursion: float = 0.0
    max_favourable_excursion: float = 0.0
    # Early window tracking (first N bars after entry)
    early_window_bars: int = 3
    early_bars_count: int = 0
    early_mae_usd: float = 0.0
    early_mfe_usd: float = 0.0
    last_bar_ts: Optional[int] = None

    def unrealised_pnl(self, current_price: float) -> float:
        if self.side == OrderSide.LONG:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def r_multiple(self, current_price: float) -> float:
        if self.risk_usd <= 0:
            return 0.0
        return self.unrealised_pnl(current_price) / self.risk_usd

    def to_core_position(self, current_price: float) -> CorePosition:
        """Convert to CorePosition for risk engine compatibility."""
        return CorePosition(
            symbol=self.symbol,
            side=self.side,
            entry_price=self.entry_price,
            quantity=self.quantity,
            notional=self.notional_usd,
            leverage=1.0,
            margin=self.notional_usd,
            stop_price=self.stop_price,
            tp_price=self.tp_price,
            unrealized_pnl=self.unrealised_pnl(current_price),
            realized_pnl=0.0,
            entry_time=self.entry_time,
            trade_id=self.trade_id,
            status=CorePositionStatus.OPEN,
        )


class BacktestAccount:
    """
    Simulated trading account for backtesting.

    Tracks:
    - Equity (starting capital ± realized PnL)
    - Open positions
    - Closed trades history
    - Daily PnL for kill switch simulation
    """

    def __init__(
        self,
        initial_equity: float = 10_000.0,
        maker_fee_pct: float = 0.0002,
        taker_fee_pct: float = 0.0004,
        slippage_pct: float = 0.0002,
    ) -> None:
        self.initial_equity = initial_equity
        self._equity = initial_equity
        self._maker_fee_pct = maker_fee_pct
        self._taker_fee_pct = taker_fee_pct
        self._slippage_pct = slippage_pct

        self._open: Dict[str, OpenBacktestPosition] = {}  # trade_id → position
        self._closed: List[BacktestTrade] = []

        # Daily PnL tracking (for kill switch)
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._current_date: Optional[str] = None  # "YYYY-MM-DD"

        # Equity curve: list of (timestamp_ms, equity)
        self.equity_curve: List[tuple] = [(0, initial_equity)]

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def free_margin(self) -> float:
        """Equity minus margin used by open positions."""
        used = sum(p.margin_usd if p.margin_usd > 0 else p.notional_usd for p in self._open.values())
        return max(0.0, self._equity - used)

    @property
    def open_positions(self) -> List[OpenBacktestPosition]:
        return list(self._open.values())

    @property
    def closed_trades(self) -> List[BacktestTrade]:
        return list(self._closed)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def weekly_pnl(self) -> float:
        return self._weekly_pnl

    # ── Day management ─────────────────────────────────────────────────

    def tick_day(self, now: datetime) -> None:
        """Call at the start of each new candle day to reset daily PnL."""
        date_str = now.strftime("%Y-%m-%d")
        # Monday = reset weekly PnL
        if now.weekday() == 0 and self._current_date != date_str:
            self._weekly_pnl = 0.0
        if self._current_date != date_str:
            self._daily_pnl = 0.0
            self._current_date = date_str

    # ── Position management ────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        side: OrderSide,
        entry_price_raw: float,  # mid price before slippage
        quantity: float,
        notional_usd: float,
        margin_usd: float,       # actual margin to block (notional / leverage)
        risk_usd: float,
        leverage_used: float,
        stop_price: float,
        tp_price: Optional[float],
        entry_time: datetime,
        strategy: str,
        regime: str,
        regime_confidence: Optional[float] = None,
        estimated_cost_usd: Optional[float] = None,
        estimated_cost_r: Optional[float] = None,
        expected_edge_r: Optional[float] = None,
        setup_quality_score: Optional[float] = None,
        trail_after_r: float = 1.0,
        atr_trail_mult: float = 2.0,
    ) -> Optional[OpenBacktestPosition]:
        """
        Simulate opening a position with slippage and entry fee.

        Returns None if insufficient equity.
        """
        # Apply entry slippage
        if side == OrderSide.LONG:
            fill_price = entry_price_raw * (1.0 + self._slippage_pct)
            entry_slippage_usd = (fill_price - entry_price_raw) * quantity
        else:
            fill_price = entry_price_raw * (1.0 - self._slippage_pct)
            entry_slippage_usd = (entry_price_raw - fill_price) * quantity

        entry_fee = notional_usd * self._maker_fee_pct

        if self.free_margin < margin_usd + entry_fee:
            return None

        trade_id = f"BT-{uuid.uuid4().hex[:8].upper()}"
        pos = OpenBacktestPosition(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            quantity=quantity,
            notional_usd=notional_usd,
            margin_usd=margin_usd,
            risk_usd=risk_usd,
            leverage_used=leverage_used,
            stop_price=stop_price,
            tp_price=tp_price,
            entry_time=entry_time,
            strategy=strategy,
            regime=regime,
            trail_after_r=trail_after_r,
            atr_trail_mult=atr_trail_mult,
            initial_stop_price=stop_price,
            fees_usd=entry_fee,
            entry_price_raw=entry_price_raw,
            entry_slippage_usd=entry_slippage_usd,
            regime_confidence=regime_confidence,
            estimated_cost_usd=estimated_cost_usd,
            estimated_cost_r=estimated_cost_r,
            expected_edge_r=expected_edge_r,
            setup_quality_score=setup_quality_score,
        )
        self._open[trade_id] = pos
        self._equity -= entry_fee
        return pos

    def close_position(
        self,
        trade_id: str,
        exit_price_raw: float,
        exit_time: datetime,
        exit_reason: str,
        is_market: bool = True,
    ) -> Optional[BacktestTrade]:
        """
        Simulate closing a position with slippage and exit fee.

        Returns the completed BacktestTrade or None if not found.
        """
        pos = self._open.pop(trade_id, None)
        if pos is None:
            return None

        fee_pct = self._taker_fee_pct if is_market else self._maker_fee_pct

        # Apply exit slippage (adverse for market exits)
        if is_market:
            if pos.side == OrderSide.LONG:
                fill_price = exit_price_raw * (1.0 - self._slippage_pct)
                exit_slippage_usd = (exit_price_raw - fill_price) * pos.quantity
            else:
                fill_price = exit_price_raw * (1.0 + self._slippage_pct)
                exit_slippage_usd = (fill_price - exit_price_raw) * pos.quantity
        else:
            fill_price = exit_price_raw  # limit/stop fills at trigger price
            exit_slippage_usd = 0.0

        exit_fee = pos.notional_usd * fee_pct
        total_fees = pos.fees_usd + exit_fee

        # PnL calculation (gross from raw prices)
        if pos.side == OrderSide.LONG:
            gross_pnl = (exit_price_raw - pos.entry_price_raw) * pos.quantity
        else:
            gross_pnl = (pos.entry_price_raw - exit_price_raw) * pos.quantity

        slippage_usd = pos.entry_slippage_usd + exit_slippage_usd
        funding_usd = 0.0
        net_pnl = gross_pnl - total_fees - slippage_usd - funding_usd
        pnl_r = net_pnl / pos.risk_usd if pos.risk_usd > 0 else 0.0
        early_mae_r = pos.early_mae_usd / pos.risk_usd if pos.risk_usd > 0 else 0.0
        early_mfe_r = pos.early_mfe_usd / pos.risk_usd if pos.risk_usd > 0 else 0.0

        # Update equity (entry fee already applied at entry)
        equity_delta = gross_pnl - slippage_usd - exit_fee - funding_usd
        self._equity += equity_delta
        self._daily_pnl += net_pnl
        self._weekly_pnl += net_pnl

        trade = BacktestTrade(
            trade_id=trade_id,
            symbol=pos.symbol,
            side=pos.side.value,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            quantity=pos.quantity,
            notional_usd=pos.notional_usd,
            leverage_used=pos.leverage_used,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            exit_reason=exit_reason,
            pnl_usd=net_pnl,
            pnl_r=pnl_r,
            risk_usd=pos.risk_usd,
            fees_usd=total_fees,
            slippage_usd=slippage_usd,
            funding_usd=funding_usd,
            gross_pnl_usd=gross_pnl,
            net_pnl_usd=net_pnl,
            strategy=pos.strategy,
            regime=pos.regime,
            regime_confidence=pos.regime_confidence,
            estimated_cost_usd=pos.estimated_cost_usd,
            estimated_cost_r=pos.estimated_cost_r,
            expected_edge_r=pos.expected_edge_r,
            setup_quality_score=pos.setup_quality_score,
            stop_price=pos.stop_price,
            tp_price=pos.tp_price,
            max_adverse_excursion=pos.max_adverse_excursion,
            max_favourable_excursion=pos.max_favourable_excursion,
            early_window_bars=pos.early_window_bars,
            early_mae_usd=pos.early_mae_usd,
            early_mfe_usd=pos.early_mfe_usd,
            early_mae_r=early_mae_r,
            early_mfe_r=early_mfe_r,
        )
        self._closed.append(trade)
        return trade

    def update_mfe_mae(
        self,
        trade_id: str,
        current_price: float,
        timestamp_ms: Optional[int] = None,
    ) -> None:
        """Update max favourable/adverse excursion for an open position."""
        pos = self._open.get(trade_id)
        if pos is None:
            return
        upnl = pos.unrealised_pnl(current_price)
        if upnl > pos.max_favourable_excursion:
            pos.max_favourable_excursion = upnl
        if upnl < pos.max_adverse_excursion:
            pos.max_adverse_excursion = upnl

        # Early window MAE/MFE tracking (first N bars after entry)
        if timestamp_ms is not None:
            if pos.last_bar_ts != timestamp_ms:
                pos.last_bar_ts = timestamp_ms
                pos.early_bars_count += 1
            if pos.early_bars_count <= pos.early_window_bars:
                if upnl > pos.early_mfe_usd:
                    pos.early_mfe_usd = upnl
                if upnl < pos.early_mae_usd:
                    pos.early_mae_usd = upnl

    def snapshot_equity(self, timestamp_ms: int) -> None:
        """Record current equity on the equity curve."""
        # Add unrealised PnL of open positions (mark-to-market)
        # Note: we don't have current prices here, so we track realised equity only.
        # The engine calls this each bar.
        self.equity_curve.append((timestamp_ms, self._equity))

    def get_core_positions(self, current_price_map: Dict[str, float]) -> List[CorePosition]:
        """Return open positions as CorePosition list for risk engine."""
        result = []
        for pos in self._open.values():
            price = current_price_map.get(pos.symbol, pos.entry_price)
            result.append(pos.to_core_position(price))
        return result
