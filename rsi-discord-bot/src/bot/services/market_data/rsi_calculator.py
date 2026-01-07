"""
RSI Calculator module for the RSI Discord Bot.

This module provides a unified interface for RSI calculation using
configurable data providers (TradingView Screener or yfinance).

The calculator maintains backwards compatibility with the existing
codebase while using the new provider system under the hood.
"""
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from bot.config import BATCH_SIZE, RSI_PROVIDER
from bot.services.market_data.providers import get_provider, RSIData

logger = logging.getLogger(__name__)


@dataclass
class RSIResult:
    """
    Result of RSI calculation for a single ticker.
    
    This class maintains backwards compatibility with the existing codebase
    while wrapping the new RSIData from providers.
    """
    ticker: str
    rsi_values: Dict[int, float]  # period -> RSI value
    last_date: str
    last_close: float
    success: bool
    error: Optional[str] = None
    data_timestamp: Optional[datetime] = None  # NEW: When data was fetched
    name: Optional[str] = None  # NEW: Company name if available
    
    @classmethod
    def from_rsi_data(cls, data: RSIData) -> 'RSIResult':
        """Create RSIResult from provider RSIData."""
        rsi_values = data.rsi_values or {}
        if data.rsi_14 is not None and 14 not in rsi_values:
            rsi_values[14] = data.rsi_14
        
        last_date = ""
        if data.data_timestamp:
            last_date = data.data_timestamp.strftime("%Y-%m-%d")
        
        return cls(
            ticker=data.ticker,
            rsi_values=rsi_values,
            last_date=last_date,
            last_close=data.close or 0.0,
            success=data.success,
            error=data.error,
            data_timestamp=data.data_timestamp,
            name=data.name
        )


def calculate_rsi(price_series: pd.Series, window: int = 14) -> Optional[float]:
    """
    Calculate Wilder's RSI for the given price series.
    
    This function is kept for backwards compatibility and for cases
    where RSI needs to be calculated from raw price data.
    
    Args:
        price_series: Pandas Series of prices (adjusted close preferred)
        window: RSI period (default 14)
    
    Returns:
        RSI value as float, or None if calculation fails
    """
    if len(price_series) < window + 1:
        return None

    # Compute price changes (differences) day by day
    delta = price_series.diff()

    # Separate positive gains and negative losses
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    # Use Wilder's exponential moving average for average gain and loss
    avg_gain = gains.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

    # Calculate Relative Strength (RS) and RSI
    last_avg_gain = avg_gain.iloc[-1]
    last_avg_loss = avg_loss.iloc[-1]

    if last_avg_loss == 0:
        if last_avg_gain == 0:
            return 50.0  # No gain and no loss -> neutral RSI
        else:
            return 100.0  # No losses -> RSI = 100

    rs = last_avg_gain / last_avg_loss
    rsi_value = 100 - (100 / (1 + rs))

    return rsi_value


def calculate_rsi_series(price_series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate RSI for entire price series (for historical analysis).
    
    Returns a Series of RSI values aligned with the input price series.
    """
    delta = price_series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


class RSICalculator:
    """
    Handles RSI calculation for multiple tickers using configurable providers.
    
    This class uses the provider system to fetch RSI data, making it easy
    to switch between TradingView Screener and yfinance.
    """
    
    def __init__(self, batch_size: int = BATCH_SIZE, provider_name: Optional[str] = None):
        """
        Initialize the RSI Calculator.
        
        Args:
            batch_size: Number of tickers to process per batch
            provider_name: Optional override for RSI provider ("tradingview" or "yfinance")
        """
        self.batch_size = batch_size
        self._provider_name = provider_name
        self._provider = None
    
    @property
    def provider(self):
        """Get the RSI provider (lazy initialization)."""
        if self._provider is None:
            self._provider = get_provider(self._provider_name)
        return self._provider
    
    async def calculate_rsi_for_tickers(
        self,
        ticker_periods: Dict[str, List[int]]
    ) -> Dict[str, RSIResult]:
        """
        Calculate RSI for multiple tickers with their required periods.
        
        Args:
            ticker_periods: Dict mapping ticker -> list of RSI periods needed
        
        Returns:
            Dict mapping ticker -> RSIResult
        """
        results: Dict[str, RSIResult] = {}
        tickers = list(ticker_periods.keys())

        if not tickers:
            return results

        logger.info(f"Fetching RSI data for {len(tickers)} tickers using {self.provider.name}")

        # Collect all unique periods
        all_periods = set()
        for periods in ticker_periods.values():
            all_periods.update(periods)
        
        # Fetch data using provider
        provider_results = await self.provider.get_rsi_for_tickers(
            tickers=tickers,
            periods=list(all_periods)
        )
        
        # Convert to RSIResult format
        for ticker, rsi_data in provider_results.items():
            results[ticker] = RSIResult.from_rsi_data(rsi_data)
        
        # Ensure all requested tickers have a result
        for ticker in tickers:
            if ticker not in results:
                results[ticker] = RSIResult(
                    ticker=ticker,
                    rsi_values={},
                    last_date="",
                    last_close=0.0,
                    success=False,
                    error="No result from provider"
                )
        
        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        logger.info(f"RSI calculation complete: {successful} successful, {failed} failed")

        return results

    def count_consecutive_days(
        self,
        price_series: pd.Series,
        threshold: float,
        condition: str,
        period: int = 14
    ) -> int:
        """
        Count consecutive trading days where RSI is above/below threshold.
        
        Note: This requires full price history and calculates RSI locally.
        
        Args:
            price_series: Historical price data
            threshold: RSI threshold
            condition: 'UNDER' or 'OVER'
            period: RSI period
        
        Returns:
            Number of consecutive days meeting the condition
        """
        rsi_series = calculate_rsi_series(price_series, window=period)
        rsi_series = rsi_series.dropna()

        if rsi_series.empty:
            return 0

        count = 0
        # Iterate backwards from most recent
        for rsi in reversed(rsi_series.values):
            if condition == "UNDER" and rsi < threshold:
                count += 1
            elif condition == "OVER" and rsi > threshold:
                count += 1
            else:
                break

        return count
