"""
RSI Calculator module for the RSI Discord Bot.
Adapted from yFinanceRSIcalc.py with support for multiple periods.
"""
import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from bot.config import BATCH_SIZE, BATCH_DELAY_SECONDS, PRICE_HISTORY_PERIOD, MIN_DATA_POINTS

logger = logging.getLogger(__name__)


@dataclass
class RSIResult:
    """Result of RSI calculation for a single ticker."""
    ticker: str
    rsi_values: Dict[int, float]  # period -> RSI value
    last_date: str
    last_close: float
    success: bool
    error: Optional[str] = None


def calculate_rsi(price_series: pd.Series, window: int = 14) -> Optional[float]:
    """
    Calculate Wilder's RSI for the given price series.
    
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
    Handles RSI calculation for multiple tickers with batched data fetching.
    """
    
    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _fetch_batch_sync(self, ticker_list: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Synchronously fetch data for a batch of tickers.
        Returns (DataFrame, error_message).
        """
        try:
            data = yf.download(
                tickers=ticker_list,
                period=PRICE_HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                progress=False
            )
            return data, None
        except Exception as e:
            return pd.DataFrame(), str(e)

    async def fetch_batch(self, ticker_list: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Asynchronously fetch data for a batch of tickers.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._fetch_batch_sync,
            ticker_list
        )

    def _process_ticker_data(
        self,
        ticker: str,
        data: pd.DataFrame,
        required_periods: List[int]
    ) -> RSIResult:
        """
        Process data for a single ticker and calculate RSI for all required periods.
        """
        try:
            # Handle MultiIndex columns (multi-ticker batch)
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0):
                    return RSIResult(
                        ticker=ticker,
                        rsi_values={},
                        last_date="",
                        last_close=0.0,
                        success=False,
                        error="No data returned"
                    )
                hist_df = data[ticker]
            else:
                hist_df = data

            if hist_df.empty:
                return RSIResult(
                    ticker=ticker,
                    rsi_values={},
                    last_date="",
                    last_close=0.0,
                    success=False,
                    error="Empty data"
                )

            # Use adjusted close if available, otherwise use close
            price_col = "Adj Close" if "Adj Close" in hist_df.columns else "Close"
            prices = hist_df[price_col].dropna()

            if len(prices) < MIN_DATA_POINTS:
                return RSIResult(
                    ticker=ticker,
                    rsi_values={},
                    last_date="",
                    last_close=0.0,
                    success=False,
                    error=f"Not enough data ({len(prices)} points)"
                )

            # Calculate RSI for each required period
            rsi_values = {}
            for period in required_periods:
                rsi = calculate_rsi(prices, window=period)
                if rsi is not None:
                    rsi_values[period] = rsi

            if not rsi_values:
                return RSIResult(
                    ticker=ticker,
                    rsi_values={},
                    last_date="",
                    last_close=0.0,
                    success=False,
                    error="RSI calculation failed"
                )

            last_date = prices.index[-1].strftime("%Y-%m-%d")
            last_close = float(prices.iloc[-1])

            return RSIResult(
                ticker=ticker,
                rsi_values=rsi_values,
                last_date=last_date,
                last_close=last_close,
                success=True
            )

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return RSIResult(
                ticker=ticker,
                rsi_values={},
                last_date="",
                last_close=0.0,
                success=False,
                error=str(e)
            )

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

        logger.info(f"Fetching RSI data for {len(tickers)} tickers")

        # Process in batches
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tickers) + self.batch_size - 1) // self.batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)")

            # Fetch data for batch
            data, error = await self.fetch_batch(batch)

            if error:
                logger.warning(f"Batch fetch error: {error}")
                # Try splitting the batch if it's larger than 1
                if len(batch) > 1:
                    mid = len(batch) // 2
                    sub_results = await self._process_failed_batch(
                        batch, ticker_periods
                    )
                    results.update(sub_results)
                else:
                    # Single ticker failed
                    ticker = batch[0]
                    results[ticker] = RSIResult(
                        ticker=ticker,
                        rsi_values={},
                        last_date="",
                        last_close=0.0,
                        success=False,
                        error=error
                    )
            else:
                # Process each ticker in the batch
                for ticker in batch:
                    result = self._process_ticker_data(
                        ticker, data, ticker_periods[ticker]
                    )
                    results[ticker] = result

            # Delay between batches to avoid rate limits
            if i + self.batch_size < len(tickers):
                await asyncio.sleep(BATCH_DELAY_SECONDS)

        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        logger.info(f"RSI calculation complete: {successful} successful, {failed} failed")

        return results

    async def _process_failed_batch(
        self,
        batch: List[str],
        ticker_periods: Dict[str, List[int]]
    ) -> Dict[str, RSIResult]:
        """
        Handle a failed batch by recursively splitting it.
        """
        results = {}

        if len(batch) == 1:
            ticker = batch[0]
            # Try one more time for single ticker
            data, error = await self.fetch_batch([ticker])
            if error or data.empty:
                results[ticker] = RSIResult(
                    ticker=ticker,
                    rsi_values={},
                    last_date="",
                    last_close=0.0,
                    success=False,
                    error=error or "No data"
                )
            else:
                results[ticker] = self._process_ticker_data(
                    ticker, data, ticker_periods[ticker]
                )
            return results

        # Split batch in half and process each
        mid = len(batch) // 2
        batch1 = batch[:mid]
        batch2 = batch[mid:]

        for sub_batch in [batch1, batch2]:
            await asyncio.sleep(BATCH_DELAY_SECONDS)
            data, error = await self.fetch_batch(sub_batch)

            if error:
                sub_results = await self._process_failed_batch(sub_batch, ticker_periods)
                results.update(sub_results)
            else:
                for ticker in sub_batch:
                    result = self._process_ticker_data(
                        ticker, data, ticker_periods[ticker]
                    )
                    results[ticker] = result

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
