"""Pure technical indicator calculation functions (no I/O)"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def ema(series: pd.Series, period: int) -> Optional[pd.Series]:
    """
    Calculate Exponential Moving Average.

    Args:
        series: Price series
        period: EMA period

    Returns:
        EMA series or None if insufficient data
    """
    if len(series) < period:
        return None

    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> Optional[pd.Series]:
    """
    Calculate Relative Strength Index.

    Args:
        series: Price series
        period: RSI period

    Returns:
        RSI series (0-100) or None if insufficient data
    """
    if len(series) < period + 1:
        return None

    # Calculate price changes
    delta = series.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate average gains and losses
    avg_gains = gains.ewm(span=period, adjust=False).mean()
    avg_losses = losses.ewm(span=period, adjust=False).mean()

    # Handle edge cases:
    # 1. Flat series (avg_gains = 0 AND avg_losses = 0): RSI is undefined (NaN)
    # 2. Only gains (avg_losses = 0, avg_gains > 0): RSI = 100
    # 3. Only losses (avg_gains = 0, avg_losses > 0): RSI = 0
    # 4. Normal case: RSI = 100 - (100 / (1 + RS))

    # Avoid division by zero
    rs = pd.Series(index=series.index, dtype=float)

    for idx in avg_gains.index:
        if avg_losses[idx] == 0 or avg_losses[idx] == -0.0:
            if avg_gains[idx] == 0:
                rs[idx] = np.nan  # Flat: undefined
            else:
                rs[idx] = np.inf  # Only gains: RSI = 100
        else:
            rs[idx] = avg_gains[idx] / avg_losses[idx]

    rsi_values = 100 - (100 / (1 + rs))

    # Handle inf (RSI = 100) and NaN (flat series)
    rsi_values = rsi_values.replace([np.inf, -np.inf], 100)

    return rsi_values


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Optional[pd.Series]:
    """
    Calculate Average True Range.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period

    Returns:
        ATR series or None if insufficient data
    """
    if len(high) < period + 1:
        return None

    # True Range components
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    # True Range is max of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is EMA of True Range
    atr_values = tr.ewm(span=period, adjust=False).mean()

    return atr_values


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Optional[pd.Series]:
    """
    Calculate Average Directional Index.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ADX period

    Returns:
        ADX series or None if insufficient data
    """
    if len(high) < period * 2:
        return None

    # Directional movement
    high_diff = high.diff()
    low_diff = -low.diff()

    # Positive and negative directional movement
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed directional indicators
    atr_values = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_values)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_values)

    # Directional Index
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    # ADX is smoothed DX
    adx_values = dx.ewm(span=period, adjust=False).mean()

    return adx_values


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std: float = 2.0
) -> Optional[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    """
    Calculate Bollinger Bands.

    Args:
        close: Close price series
        period: Moving average period
        std: Standard deviation multiplier

    Returns:
        Tuple of (middle, upper, lower, width) or None if insufficient data
    """
    if len(close) < period:
        return None

    # Middle band (SMA)
    middle = close.rolling(window=period).mean()

    # Standard deviation
    rolling_std = close.rolling(window=period).std()

    # Upper and lower bands
    upper = middle + (rolling_std * std)
    lower = middle - (rolling_std * std)

    # Bollinger Band width (normalized)
    width = (upper - lower) / middle

    return middle, upper, lower, width


def zscore(values: pd.Series, window: int = 100) -> Optional[pd.Series]:
    """
    Calculate rolling z-score.

    Args:
        values: Value series
        window: Rolling window size

    Returns:
        Z-score series or None if insufficient data
    """
    if len(values) < window:
        return None

    # Rolling mean and std
    rolling_mean = values.rolling(window=window).mean()
    rolling_std = values.rolling(window=window).std()

    # Z-score
    z = (values - rolling_mean) / rolling_std.replace(0, np.nan)

    return z


def log_returns(close: pd.Series) -> Optional[pd.Series]:
    """
    Calculate log returns for correlation analysis.

    Args:
        close: Close price series

    Returns:
        Log returns series or None if insufficient data
    """
    if len(close) < 2:
        return None

    return np.log(close / close.shift())
