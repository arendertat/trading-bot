"""Custom exceptions for exchange layer"""


class ExchangeError(Exception):
    """Base exception for all exchange-related errors"""
    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded"""
    pass


class TimestampError(ExchangeError):
    """Timestamp sync issue (recvWindow validation failed)"""
    pass


class AuthError(ExchangeError):
    """Authentication failed (invalid API key/secret)"""
    pass


class OrderNotFoundError(ExchangeError):
    """Order not found"""
    pass


class InsufficientBalanceError(ExchangeError):
    """Insufficient balance for order"""
    pass
