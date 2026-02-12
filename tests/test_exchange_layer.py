"""Unit tests for exchange layer (using mocks, no real API calls)"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from bot.exchange.binance_client import BinanceFuturesClient
from bot.exchange.order_manager import OrderManager
from bot.exchange.health_monitor import HealthMonitor
from bot.exchange.exceptions import (
    ExchangeError,
    RateLimitError,
    TimestampError,
    AuthError,
)


class TestOrderManager:
    """Test OrderManager functionality"""

    def test_build_client_order_id_deterministic(self):
        """Test that clientOrderId is deterministic"""
        trade_id = "T20240211_001"
        role = "entry"

        # Call multiple times
        id1 = OrderManager.build_client_order_id(trade_id, role)
        id2 = OrderManager.build_client_order_id(trade_id, role)
        id3 = OrderManager.build_client_order_id(trade_id, role)

        # Should be identical
        assert id1 == id2 == id3
        assert id1 == "T20240211_001_entry"

    def test_build_client_order_id_format(self):
        """Test clientOrderId format"""
        test_cases = [
            ("T001", "entry", "T001_entry"),
            ("TRADE_123", "stop", "TRADE_123_stop"),
            ("X", "tp", "X_tp"),
        ]

        for trade_id, role, expected in test_cases:
            result = OrderManager.build_client_order_id(trade_id, role)
            assert result == expected

    def test_build_client_order_id_empty_raises_error(self):
        """Test that empty trade_id or role raises error"""
        with pytest.raises(ValueError, match="cannot be empty"):
            OrderManager.build_client_order_id("", "entry")

        with pytest.raises(ValueError, match="cannot be empty"):
            OrderManager.build_client_order_id("T001", "")

    def test_build_client_order_id_too_long_raises_error(self):
        """Test that too long clientOrderId raises error"""
        # Binance max is 36 chars
        very_long_trade_id = "T" * 50

        with pytest.raises(ValueError, match="clientOrderId too long"):
            OrderManager.build_client_order_id(very_long_trade_id, "entry")

    def test_ensure_unique_checks_cache_first(self):
        """Test that ensure_unique_and_place checks cache first"""
        mock_client = Mock(spec=BinanceFuturesClient)
        manager = OrderManager(mock_client)

        # Populate cache
        cached_order = {'id': '123', 'clientOrderId': 'T001_entry'}
        manager._order_cache['T001_entry'] = cached_order

        # Call ensure_unique_and_place
        result = manager.ensure_unique_and_place(
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=0.001,
            price=50000,
            trade_id="T001",
            role="entry",
        )

        # Should return cached order without calling client
        assert result == cached_order
        mock_client.place_order.assert_not_called()

    def test_ensure_unique_places_new_order_if_not_cached(self):
        """Test that new order is placed if not in cache"""
        mock_client = Mock(spec=BinanceFuturesClient)
        new_order = {'id': '456', 'clientOrderId': 'T002_entry'}
        mock_client.place_order.return_value = new_order

        manager = OrderManager(mock_client)

        result = manager.ensure_unique_and_place(
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=0.001,
            price=50000,
            trade_id="T002",
            role="entry",
        )

        # Should place order via client
        assert result == new_order
        mock_client.place_order.assert_called_once()

        # Check that cache was updated
        assert manager._order_cache['T002_entry'] == new_order


class TestBinanceFuturesClient:
    """Test BinanceFuturesClient functionality"""

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_idempotent_order_placement(self, mock_ccxt_class):
        """Test that duplicate orders are not placed"""
        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        # Mock existing order
        existing_order = {
            'id': '123',
            'clientOrderId': 'TEST_entry',
            'status': 'open',
            'symbol': 'BTC/USDT',  # Required for idempotency check
        }
        mock_exchange.fetch_open_orders.return_value = [existing_order]

        client = BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
        )

        # Try to place order with same clientOrderId
        result = client.place_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=0.001,
            price=50000,
            client_order_id="TEST_entry",
        )

        # Should return existing order without calling create_order
        assert result == existing_order
        mock_exchange.create_order.assert_not_called()

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_retry_on_rate_limit(self, mock_ccxt_class):
        """Test exponential backoff retry on rate limit error"""
        import ccxt

        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        # Mock load_time_difference (called in __init__)
        mock_exchange.load_time_difference.return_value = None

        # First call raises DDoSProtection, second succeeds
        mock_exchange.fetch_time.side_effect = [
            ccxt.DDoSProtection("Rate limit"),
            int(time.time() * 1000),
        ]

        client = BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            max_retries=3,
        )

        with patch('time.sleep') as mock_sleep:
            result = client.get_server_time_ms()

            # Should have retried once
            assert mock_exchange.fetch_time.call_count == 2
            mock_sleep.assert_called_once_with(1)  # First backoff: 2^0 = 1s
            assert isinstance(result, int)

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_retry_on_timestamp_error_syncs_time(self, mock_ccxt_class):
        """Test that timestamp error triggers time sync via load_time_difference"""
        import ccxt

        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        # First call raises InvalidNonce, subsequent calls succeed
        mock_exchange.fetch_balance.side_effect = [
            ccxt.InvalidNonce("Timestamp error"),
            {'USDT': {'total': 1000, 'free': 500, 'used': 500}},
        ]

        client = BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            max_retries=3,
        )

        with patch('time.sleep'):
            result = client.fetch_balance_usdt()

            # Should have called load_time_difference() to sync time
            # (once during init + once during retry = 2 calls)
            assert mock_exchange.load_time_difference.call_count == 2
            # Should have retried balance fetch
            assert mock_exchange.fetch_balance.call_count == 2
            assert result['total'] == 1000

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_retry_on_timestamp_in_error_message(self, mock_ccxt_class):
        """Test that errors containing 'timestamp' or 'recvwindow' trigger time sync"""
        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        # First call raises generic Exception with timestamp keyword, second succeeds
        mock_exchange.fetch_balance.side_effect = [
            Exception("Request failed: timestamp for this request is outside of the recvWindow"),
            {'USDT': {'total': 500, 'free': 250, 'used': 250}},
        ]

        client = BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            max_retries=3,
        )

        with patch('time.sleep'):
            result = client.fetch_balance_usdt()

            # Should have called load_time_difference() to sync time
            # (once during init + once during retry = 2 calls)
            assert mock_exchange.load_time_difference.call_count == 2
            # Should have retried balance fetch
            assert mock_exchange.fetch_balance.call_count == 2
            assert result['total'] == 500

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_auth_error_not_retried(self, mock_ccxt_class):
        """Test that authentication errors are not retried"""
        import ccxt

        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        mock_exchange.fetch_balance.side_effect = ccxt.AuthenticationError("Invalid API key")

        client = BinanceFuturesClient(
            api_key="bad_key",
            api_secret="bad_secret",
        )

        with pytest.raises(AuthError, match="Authentication failed"):
            client.fetch_balance_usdt()

        # Should only try once (no retry)
        assert mock_exchange.fetch_balance.call_count == 1

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_max_retries_exceeded_raises_error(self, mock_ccxt_class):
        """Test that max retries exceeded raises error"""
        import ccxt

        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        # Mock load_time_difference (called in __init__)
        mock_exchange.load_time_difference.return_value = None

        # Always raise rate limit error
        mock_exchange.fetch_time.side_effect = ccxt.DDoSProtection("Rate limit")

        client = BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            max_retries=2,
        )

        with patch('time.sleep'):
            with pytest.raises(ExchangeError, match="Rate limit"):  # Error message changed
                client.get_server_time_ms()

            # Should have tried max_retries times
            assert mock_exchange.fetch_time.call_count == 2

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_unsupported_order_type_raises_not_implemented(self, mock_ccxt_class):
        """Test that unsupported order types raise NotImplementedError (Fix #3)"""
        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        client = BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
        )

        # Test STOP_MARKET
        with pytest.raises(NotImplementedError, match="Order type STOP_MARKET not supported until Milestone 7"):
            client.place_order(
                symbol="BTC/USDT",
                side="buy",
                order_type="STOP_MARKET",
                quantity=0.001,
                price=50000,
            )

        # Test TAKE_PROFIT
        with pytest.raises(NotImplementedError, match="Order type TAKE_PROFIT not supported until Milestone 7"):
            client.place_order(
                symbol="BTC/USDT",
                side="sell",
                order_type="TAKE_PROFIT",
                quantity=0.001,
                price=52000,
            )

        # Test case-insensitivity (lowercase should also raise)
        with pytest.raises(NotImplementedError, match="Order type stop_market not supported until Milestone 7"):
            client.place_order(
                symbol="BTC/USDT",
                side="buy",
                order_type="stop_market",
                quantity=0.001,
            )

        # Verify LIMIT and MARKET are still allowed (should not raise NotImplementedError)
        mock_exchange.fetch_open_orders.return_value = []
        mock_exchange.create_order.return_value = {'id': '123', 'status': 'new'}

        # LIMIT should work
        client.place_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="LIMIT",
            quantity=0.001,
            price=50000,
        )

        # MARKET should work
        client.place_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="MARKET",
            quantity=0.001,
        )

        # Lowercase should work too
        client.place_order(
            symbol="BTC/USDT",
            side="buy",
            order_type="limit",
            quantity=0.001,
            price=50000,
        )

        # Should have called create_order 3 times (for the valid order types)
        assert mock_exchange.create_order.call_count == 3

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_health_monitor_records_success(self, mock_ccxt_class):
        """Test that health monitor records success on successful requests (Fix #4)"""
        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        # Mock load_time_difference (called in __init__)
        mock_exchange.load_time_difference.return_value = None

        mock_exchange.fetch_time.return_value = int(time.time() * 1000)

        client = BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
        )

        # Initial state should be healthy with 0 consecutive errors
        assert client.health_monitor.is_healthy()
        assert client.health_monitor.consecutive_errors == 0

        # Make a successful request
        client.get_server_time_ms()

        # Health monitor should record success
        assert client.health_monitor.consecutive_errors == 0
        assert client.health_monitor.last_success_ts is not None

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_health_monitor_records_failures(self, mock_ccxt_class):
        """Test that health monitor records failures on errors (Fix #4)"""
        import ccxt

        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        # Mock load_time_difference (called in __init__)
        mock_exchange.load_time_difference.return_value = None

        # Always fail with rate limit error
        mock_exchange.fetch_time.side_effect = ccxt.DDoSProtection("Rate limit")

        client = BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            max_retries=3,
            health_error_threshold=3,
        )

        # Initial state
        assert client.health_monitor.is_healthy()
        assert client.health_monitor.consecutive_errors == 0

        with patch('time.sleep'):
            # Attempt request (will fail all retries)
            with pytest.raises(ExchangeError):
                client.get_server_time_ms()

        # Health monitor should record failures (each retry attempt)
        # With max_retries=3, total attempts = 3, so 3 failures recorded
        # But _retry_with_backoff records on each retry, not final raise
        assert client.health_monitor.consecutive_errors == 3 or client.health_monitor.consecutive_errors == 4
        assert not client.health_monitor.is_healthy()  # >= threshold of 3

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_health_monitor_resets_on_success_after_failures(self, mock_ccxt_class):
        """Test that health monitor resets error count after success (Fix #4)"""
        import ccxt

        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        # Mock load_time_difference (called in __init__)
        mock_exchange.load_time_difference.return_value = None

        client = BinanceFuturesClient(
            api_key="test_key",
            api_secret="test_secret",
            max_retries=2,
        )

        # First request fails then succeeds
        mock_exchange.fetch_time.side_effect = [
            ccxt.DDoSProtection("Rate limit"),
            int(time.time() * 1000),
        ]

        with patch('time.sleep'):
            result = client.get_server_time_ms()

        # Should have 1 failure recorded (first attempt), then success resets counter
        assert client.health_monitor.consecutive_errors == 0
        assert client.health_monitor.is_healthy()
        assert client.health_monitor.last_success_ts is not None

    @patch('bot.exchange.binance_client.ccxt.binance')
    def test_health_monitor_records_non_retryable_errors(self, mock_ccxt_class):
        """Test that health monitor records non-retryable errors (Fix #4)"""
        import ccxt

        mock_exchange = MagicMock()
        mock_ccxt_class.return_value = mock_exchange

        mock_exchange.fetch_balance.side_effect = ccxt.AuthenticationError("Invalid API key")

        client = BinanceFuturesClient(
            api_key="bad_key",
            api_secret="bad_secret",
        )

        # Initial state
        assert client.health_monitor.consecutive_errors == 0

        # Non-retryable error should still be recorded in health monitor
        with pytest.raises(AuthError):
            client.fetch_balance_usdt()

        # Should have recorded the failure
        assert client.health_monitor.consecutive_errors == 1
        assert client.health_monitor.last_error is not None


class TestHealthMonitor:
    """Test HealthMonitor functionality"""

    def test_initial_state_healthy(self):
        """Test that monitor starts in healthy state"""
        monitor = HealthMonitor(error_threshold=3)

        assert monitor.is_healthy()
        assert monitor.consecutive_errors == 0
        assert monitor.last_success_ts is None

    def test_record_success_resets_error_count(self):
        """Test that success resets consecutive error count"""
        monitor = HealthMonitor(error_threshold=3)

        # Simulate errors
        monitor.record_failure(Exception("Error 1"))
        monitor.record_failure(Exception("Error 2"))
        assert monitor.consecutive_errors == 2

        # Success should reset
        monitor.record_success()
        assert monitor.consecutive_errors == 0
        assert monitor.is_healthy()
        assert monitor.last_success_ts is not None

    def test_becomes_unhealthy_after_threshold(self):
        """Test that monitor becomes unhealthy after error threshold"""
        monitor = HealthMonitor(error_threshold=3)

        assert monitor.is_healthy()

        monitor.record_failure(Exception("Error 1"))
        assert monitor.is_healthy()  # 1 < 3

        monitor.record_failure(Exception("Error 2"))
        assert monitor.is_healthy()  # 2 < 3

        monitor.record_failure(Exception("Error 3"))
        assert not monitor.is_healthy()  # 3 >= 3

    def test_get_status(self):
        """Test get_status returns correct snapshot"""
        monitor = HealthMonitor(error_threshold=3)

        monitor.record_success()
        time.sleep(0.01)
        monitor.record_failure(Exception("Test error"))

        status = monitor.get_status()

        assert status.consecutive_errors == 1
        assert status.last_error == "Test error"
        assert status.last_success_ts is not None
        assert status.last_error_ts is not None
        assert status.is_healthy  # 1 < 3

    def test_error_rate_calculation(self):
        """Test error rate calculation"""
        monitor = HealthMonitor()

        assert monitor.get_error_rate() == 0.0  # No requests yet

        monitor.record_success()
        assert monitor.get_error_rate() == 0.0  # 0/1

        monitor.record_failure(Exception("Error"))
        assert monitor.get_error_rate() == 0.5  # 1/2

        monitor.record_failure(Exception("Error"))
        assert monitor.get_error_rate() == pytest.approx(0.666, rel=0.01)  # 2/3

    def test_reset(self):
        """Test that reset clears all state"""
        monitor = HealthMonitor()

        monitor.record_success()
        monitor.record_failure(Exception("Error"))

        assert monitor.consecutive_errors > 0
        assert monitor.last_success_ts is not None

        monitor.reset()

        assert monitor.consecutive_errors == 0
        assert monitor.last_success_ts is None
        assert monitor.last_error is None
        assert monitor.get_error_rate() == 0.0
