"""Universe selection engine for filtering and ranking symbols"""

import logging
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd

from bot.config.models import UniverseConfig
from bot.exchange.binance_client import BinanceFuturesClient
from bot.data.candle_store import CandleStore
from bot.data import features
from bot.universe.models import SymbolEligibility


logger = logging.getLogger("trading_bot.universe")


class UniverseSelector:
    """
    Daily universe builder that filters and ranks symbols.

    Applies eligibility filters:
    - 24h quote volume
    - Spread
    - Funding rate
    - ATR ratio

    Ranks by liquidity/quality score and returns top N.
    """

    def __init__(
        self,
        exchange_client: BinanceFuturesClient,
        candle_store: CandleStore,
        config: UniverseConfig,
    ):
        """
        Initialize UniverseSelector.

        Args:
            exchange_client: Binance futures client
            candle_store: Candle store for ATR calculation
            config: Universe configuration
        """
        self.client = exchange_client
        self.candle_store = candle_store
        self.config = config

        logger.info("UniverseSelector initialized")

    def build_daily_universe(self, now_utc: datetime) -> List[str]:
        """
        Build daily universe of eligible symbols.

        Filters all USDT-M perpetual symbols by:
        - Volume, spread, funding, ATR ratio
        - Whitelist/blacklist

        Ranks by liquidity/quality score and returns top N.

        Args:
            now_utc: Current UTC datetime (for logging/versioning)

        Returns:
            List of top N eligible symbol names
        """
        logger.info(f"Building daily universe at {now_utc.isoformat()}")

        # Step 1: Get all USDT-M perpetual symbols
        all_symbols = self.client.list_usdtm_perp_symbols()
        logger.info(f"Found {len(all_symbols)} total USDT-M perpetual symbols")

        # Step 2: Apply whitelist/blacklist
        candidate_symbols = self._apply_whitelist_blacklist(all_symbols)
        logger.info(f"After whitelist/blacklist: {len(candidate_symbols)} candidates")

        if not candidate_symbols:
            logger.warning("No candidate symbols after whitelist/blacklist filter")
            return []

        # Step 3: Fetch market data
        tickers = self.client.fetch_24h_tickers(candidate_symbols)
        funding_rates = self.client.fetch_funding_rates(candidate_symbols)

        # Step 4: Fetch historical data for ATR calculation (5m, 14 periods)
        # We need at least 14 candles for ATR(14)
        self._warmup_candles_for_atr(candidate_symbols)

        # Step 5: Evaluate eligibility for each symbol
        eligibility_results: List[SymbolEligibility] = []

        for symbol in candidate_symbols:
            eligibility = self._evaluate_symbol(
                symbol=symbol,
                ticker=tickers.get(symbol),
                funding_rate=funding_rates.get(symbol, 0.0),
            )
            eligibility_results.append(eligibility)

        # Step 6: Filter eligible symbols
        eligible = [e for e in eligibility_results if e.is_eligible]
        logger.info(f"Eligible symbols: {len(eligible)} / {len(candidate_symbols)}")

        if not eligible:
            logger.warning("No symbols passed eligibility filters")
            return []

        # Step 7: Rank by score (descending) and take top N
        eligible.sort(key=lambda e: e.score, reverse=True)
        top_n = eligible[:self.config.max_monitored_symbols]

        selected_symbols = [e.symbol for e in top_n]

        logger.info(f"Selected top {len(selected_symbols)} symbols: {selected_symbols}")
        logger.debug(f"Top symbol scores: {[(e.symbol, e.score) for e in top_n[:5]]}")

        return selected_symbols

    def _apply_whitelist_blacklist(self, symbols: List[str]) -> List[str]:
        """
        Apply whitelist and blacklist filters.

        Rules:
        - If whitelist is non-empty, only use whitelisted symbols
        - Always exclude blacklisted symbols

        Args:
            symbols: Input symbol list

        Returns:
            Filtered symbol list
        """
        # Apply whitelist first
        if self.config.whitelist:
            symbols = [s for s in symbols if s in self.config.whitelist]
            logger.debug(f"Whitelist applied: {len(symbols)} symbols")

        # Apply blacklist
        if self.config.blacklist:
            symbols = [s for s in symbols if s not in self.config.blacklist]
            logger.debug(f"Blacklist applied: {len(symbols)} symbols")

        return symbols

    def _warmup_candles_for_atr(self, symbols: List[str]) -> None:
        """
        Fetch recent candles for ATR calculation (5m, 14 periods minimum).

        Args:
            symbols: List of symbols to warm up
        """
        for symbol in symbols:
            try:
                # Fetch last 20 candles (more than 14 needed for ATR)
                klines = self.client.fetch_klines(symbol, timeframe="5m", limit=20)

                # Store in candle store
                for kline_dict in klines:
                    from bot.core.types import Candle
                    candle = Candle(
                        timestamp=int(kline_dict['timestamp']),
                        open=float(kline_dict['open']),
                        high=float(kline_dict['high']),
                        low=float(kline_dict['low']),
                        close=float(kline_dict['close']),
                        volume=float(kline_dict['volume']),
                    )
                    self.candle_store.add_candle(symbol, "5m", candle)

                logger.debug(f"Warmed up {len(klines)} candles for {symbol}")

            except Exception as e:
                logger.warning(f"Failed to warm up candles for {symbol}: {e}")

    def _evaluate_symbol(
        self,
        symbol: str,
        ticker: Optional[Dict],
        funding_rate: float,
    ) -> SymbolEligibility:
        """
        Evaluate a single symbol for eligibility.

        Args:
            symbol: Symbol name
            ticker: 24h ticker data (or None)
            funding_rate: Current funding rate

        Returns:
            SymbolEligibility result
        """
        reasons = []
        pass_volume = False
        pass_spread = False
        pass_funding = False
        pass_atr_ratio = False
        score = 0.0

        # Check ticker data exists
        if ticker is None:
            reasons.append("No ticker data")
            return SymbolEligibility(
                symbol=symbol,
                pass_volume=False,
                pass_spread=False,
                pass_funding=False,
                pass_atr_ratio=False,
                score=0.0,
                reasons=reasons,
            )

        # Filter 1: 24h quote volume
        quote_volume = ticker.get('quote_volume_usdt', 0)
        pass_volume = quote_volume >= self.config.min_24h_volume_usdt
        if not pass_volume:
            reasons.append(
                f"Volume too low: {quote_volume:.0f} < {self.config.min_24h_volume_usdt:.0f}"
            )

        # Filter 2: Spread
        bid = ticker.get('bid')
        ask = ticker.get('ask')

        if bid is not None and ask is not None and bid > 0:
            spread_pct = (ask - bid) / bid
            pass_spread = spread_pct <= self.config.max_spread_pct
            if not pass_spread:
                reasons.append(
                    f"Spread too wide: {spread_pct:.6f} > {self.config.max_spread_pct:.6f}"
                )
        else:
            reasons.append("Missing bid/ask data")
            spread_pct = 1.0  # Assign high spread if missing

        # Filter 3: Funding rate
        abs_funding = abs(funding_rate)
        pass_funding = abs_funding <= self.config.max_abs_funding_rate
        if not pass_funding:
            reasons.append(
                f"Funding rate too high: {abs_funding:.6f} > {self.config.max_abs_funding_rate:.6f}"
            )

        # Filter 4: ATR ratio
        atr_ratio = self._calculate_atr_ratio(symbol)
        if atr_ratio is not None:
            pass_atr_ratio = atr_ratio >= self.config.min_atr_ratio
            if not pass_atr_ratio:
                reasons.append(
                    f"ATR ratio too low: {atr_ratio:.6f} < {self.config.min_atr_ratio:.6f}"
                )
        else:
            reasons.append("ATR calculation failed (insufficient data)")

        # Calculate liquidity/quality score
        # Default formula: (volume / min_volume) / max(spread, epsilon)
        # Higher volume + lower spread = higher score
        if pass_volume and pass_spread and pass_funding and pass_atr_ratio:
            volume_ratio = quote_volume / self.config.min_24h_volume_usdt
            spread_penalty = max(spread_pct, 1e-9)

            # Base score: volume efficiency / spread
            score = volume_ratio / spread_penalty

            # Optional small funding penalty (closer to max = worse)
            funding_penalty = 1.0 - (abs_funding / self.config.max_abs_funding_rate)
            score *= funding_penalty

        return SymbolEligibility(
            symbol=symbol,
            pass_volume=pass_volume,
            pass_spread=pass_spread,
            pass_funding=pass_funding,
            pass_atr_ratio=pass_atr_ratio,
            score=score,
            reasons=reasons,
        )

    def _calculate_atr_ratio(self, symbol: str) -> Optional[float]:
        """
        Calculate ATR(14) / price ratio for a symbol.

        Args:
            symbol: Symbol name

        Returns:
            ATR ratio or None if insufficient data
        """
        try:
            # Get candles from store
            candles = self.candle_store.get_candles(symbol, "5m")

            if len(candles) < 15:  # Need 14+1 for ATR
                return None

            # Convert to DataFrame
            df = pd.DataFrame({
                'high': [c.high for c in candles],
                'low': [c.low for c in candles],
                'close': [c.close for c in candles],
            })

            # Calculate ATR(14)
            atr_series = features.atr(df['high'], df['low'], df['close'], period=14)

            if atr_series is None:
                return None

            # Get latest ATR and price
            latest_atr = atr_series.iloc[-1]
            latest_price = candles[-1].close

            if latest_price <= 0:
                return None

            atr_ratio = latest_atr / latest_price

            return float(atr_ratio)

        except Exception as e:
            logger.debug(f"ATR calculation failed for {symbol}: {e}")
            return None
