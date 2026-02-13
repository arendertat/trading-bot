"""
Telegram Notification System

Sends trading alerts and daily reports via Telegram Bot API.
Non-blocking: notification failures never interrupt trading logic.

Requires environment variables:
    TELEGRAM_BOT_TOKEN  — Telegram bot token from @BotFather
    TELEGRAM_CHAT_ID    — Chat ID to send messages to

Usage::

    from bot.reporting.notifier import TelegramNotifier

    notifier = TelegramNotifier(token="...", chat_id="...")
    notifier.send("Bot started successfully.")
    notifier.send_daily_report(summary)
    notifier.send_kill_switch_alert("Daily loss limit reached")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import json

from bot.reporting.daily_report import DailySummary, DailyReportGenerator


module_logger = logging.getLogger("trading_bot.reporting.notifier")

# Telegram API base URL
_TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"

# Max Telegram message length (4096 chars); truncate if exceeded
_TELEGRAM_MAX_LEN = 4096

# Retry settings for failed sends
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2.0


class TelegramNotifier:
    """
    Sends Telegram messages for trading events and daily reports.

    All send methods are fail-safe: exceptions are caught and logged,
    never propagated to the caller. Trading must never stop due to a
    notification failure.

    Attributes:
        _token: Telegram bot token
        _chat_id: Target chat ID
        _enabled: Whether Telegram is enabled
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True,
        token_env: str = "TELEGRAM_BOT_TOKEN",
        chat_id_env: str = "TELEGRAM_CHAT_ID",
    ) -> None:
        """
        Initialise TelegramNotifier.

        Token and chat_id can be passed directly or read from environment
        variables. If either is missing, notifications are silently disabled.

        Args:
            token: Telegram bot token (overrides env var if provided)
            chat_id: Target Telegram chat ID (overrides env var if provided)
            enabled: Master enable/disable switch
            token_env: Environment variable name for bot token
            chat_id_env: Environment variable name for chat ID
        """
        self._token = token or os.environ.get(token_env, "")
        self._chat_id = chat_id or os.environ.get(chat_id_env, "")
        self._enabled = enabled and bool(self._token) and bool(self._chat_id)

        if enabled and not self._enabled:
            module_logger.warning(
                "TelegramNotifier: token or chat_id missing — notifications disabled"
            )
        else:
            module_logger.info(
                f"TelegramNotifier initialised "
                f"(enabled={self._enabled}, chat_id={self._chat_id!r})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        """Whether Telegram notifications are active."""
        return self._enabled

    def send(self, text: str) -> bool:
        """
        Send a plain text message to the configured chat.

        Silently logs and returns False on failure; never raises.

        Args:
            text: Message text (truncated to Telegram's 4096-char limit)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._enabled:
            module_logger.debug(f"Notifier disabled — skipping: {text[:80]!r}")
            return False

        truncated = self._truncate(text)
        return self._send_with_retry(truncated)

    def send_daily_report(self, summary: DailySummary) -> bool:
        """
        Send formatted daily report to Telegram.

        Args:
            summary: DailySummary from DailyReportGenerator.build()

        Returns:
            True if sent successfully, False otherwise
        """
        gen = DailyReportGenerator.__new__(DailyReportGenerator)
        text = gen.format_text(summary)
        module_logger.info(f"Sending daily report for {summary.report_date.isoformat()}")
        return self.send(text)

    def send_trade_opened(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_price: float,
        risk_usd: float,
        strategy: str,
    ) -> bool:
        """
        Send trade-opened notification.

        Args:
            symbol: Trading pair
            direction: LONG or SHORT
            entry_price: Entry price
            stop_price: Stop loss price
            risk_usd: Risk amount in USD
            strategy: Strategy name

        Returns:
            True if sent successfully, False otherwise
        """
        stop_pct = abs(entry_price - stop_price) / entry_price * 100.0
        text = (
            f"TRADE OPENED\n"
            f"Symbol:   {symbol}\n"
            f"Side:     {direction}\n"
            f"Entry:    {entry_price:.4f}\n"
            f"Stop:     {stop_price:.4f} ({stop_pct:.2f}%)\n"
            f"Risk:     {risk_usd:.2f} USD\n"
            f"Strategy: {strategy}"
        )
        return self.send(text)

    def send_trade_closed(
        self,
        symbol: str,
        direction: str,
        pnl_usd: float,
        pnl_r: float,
        exit_reason: str,
    ) -> bool:
        """
        Send trade-closed notification.

        Args:
            symbol: Trading pair
            direction: LONG or SHORT
            pnl_usd: Realized PnL in USD
            pnl_r: PnL in R-multiples
            exit_reason: Exit reason string

        Returns:
            True if sent successfully, False otherwise
        """
        sign = "+" if pnl_usd >= 0 else ""
        emoji = "✓" if pnl_usd >= 0 else "✗"
        text = (
            f"{emoji} TRADE CLOSED\n"
            f"Symbol:  {symbol}\n"
            f"Side:    {direction}\n"
            f"PnL:     {sign}{pnl_usd:.2f} USD ({sign}{pnl_r:.2f} R)\n"
            f"Reason:  {exit_reason}"
        )
        return self.send(text)

    def send_kill_switch_alert(self, reason: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Send kill switch activation alert (high priority).

        Args:
            reason: Reason for activation
            context: Optional context dict (e.g. {"open_positions": 2})

        Returns:
            True if sent successfully, False otherwise
        """
        ctx_str = ""
        if context:
            ctx_str = "\n" + "\n".join(f"  {k}: {v}" for k, v in context.items())

        text = f"KILL SWITCH ACTIVATED\nReason: {reason}{ctx_str}"
        module_logger.warning(f"Sending kill switch alert: {reason}")
        return self.send(text)

    def send_safe_mode_alert(self, reason: str) -> bool:
        """
        Send safe mode activation alert.

        Args:
            reason: Reason for entering safe mode

        Returns:
            True if sent successfully, False otherwise
        """
        text = f"SAFE MODE ACTIVE\nReason: {reason}\nNo new entries until resolved."
        module_logger.warning(f"Sending safe mode alert: {reason}")
        return self.send(text)

    def send_error_alert(self, component: str, message: str) -> bool:
        """
        Send error alert notification.

        Args:
            component: Component that raised the error
            message: Error message

        Returns:
            True if sent successfully, False otherwise
        """
        text = f"ERROR [{component}]\n{message}"
        return self.send(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_with_retry(self, text: str) -> bool:
        """
        Send message with exponential backoff retry.

        Args:
            text: Message text (already truncated)

        Returns:
            True if sent, False after all retries exhausted
        """
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                success = self._send_once(text)
                if success:
                    return True
            except Exception as exc:
                module_logger.warning(
                    f"Telegram send attempt {attempt}/{_MAX_RETRIES} failed: {exc}"
                )

            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)

        module_logger.error(
            f"Telegram send failed after {_MAX_RETRIES} attempts"
        )
        return False

    def _send_once(self, text: str) -> bool:
        """
        Make a single Telegram API call.

        Uses stdlib urllib (no external dependencies).

        Args:
            text: Message text

        Returns:
            True if HTTP 200 received, False otherwise

        Raises:
            URLError: On network errors
            HTTPError: On non-200 responses
        """
        url = _TELEGRAM_API_BASE.format(token=self._token)
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": text,
        }).encode("utf-8")

        req = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(req, timeout=10) as response:
            status = response.status
            if status == 200:
                module_logger.debug("Telegram message sent successfully")
                return True
            else:
                module_logger.warning(f"Telegram API returned status {status}")
                return False

    @staticmethod
    def _truncate(text: str) -> str:
        """
        Truncate text to Telegram's max message length.

        Args:
            text: Input text

        Returns:
            Text truncated to _TELEGRAM_MAX_LEN chars with suffix if cut
        """
        if len(text) <= _TELEGRAM_MAX_LEN:
            return text
        suffix = "\n... [truncated]"
        return text[: _TELEGRAM_MAX_LEN - len(suffix)] + suffix


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_notifier_from_config(
    enabled: bool,
    token_env: str = "TELEGRAM_BOT_TOKEN",
    chat_id_env: str = "TELEGRAM_CHAT_ID",
) -> TelegramNotifier:
    """
    Create TelegramNotifier from configuration values.

    Args:
        enabled: Whether to enable Telegram notifications
        token_env: Environment variable for bot token
        chat_id_env: Environment variable for chat ID

    Returns:
        Configured TelegramNotifier instance
    """
    return TelegramNotifier(
        enabled=enabled,
        token_env=token_env,
        chat_id_env=chat_id_env,
    )
