"""
yfinance RSI Provider.

Uses the yfinance package to fetch price data and calculate RSI locally.
This is the fallback provider when TradingView Screener is not available.

Advantages:
- Supports any RSI period (not just 14)
- More control over calculation
- Historical data access

Disadvantages:
- Requires downloading price history (slower)
- Subject to Yahoo Finance rate limits
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from bot.config import (
    BATCH_SIZE, BATCH_DELAY_SECONDS, PRICE_HISTORY_PERIOD, MIN_DATA_POINTS,
    RETRY_MAX_ATTEMPTS, RETRY_DELAY_SECONDS, RETRY_BATCH_SIZE
)
from bot.services.market_data.providers.base import RSIProviderBase, RSIData

logger = logging.getLogger(__name__)


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


class YFinanceProvider(RSIProviderBase):
    """
    RSI provider using Yahoo Finance (yfinance).
    
    Features:
    - Supports any RSI period
    - Downloads price history and calculates RSI locally
    - Batch processing with rate limit handling
    """
    
    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    @property
    def name(self) -> str:
        return "Yahoo Finance (yfinance)"
    
    def _fetch_batch_sync(
        self,
        tickers: List[str],
        periods: List[int]
    ) -> Dict[str, RSIData]:
        """
        Synchronously fetch data and calculate RSI for a batch of tickers.
        """
        results = {}
        fetch_time = datetime.utcnow()
        
        try:
            # Download data for all tickers in batch
            data = yf.download(
                tickers=tickers,
                period=PRICE_HISTORY_PERIOD,
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                progress=False
            )
            
            # Process each ticker
            for ticker in tickers:
                try:
                    # Handle MultiIndex columns (multi-ticker batch)
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker not in data.columns.get_level_values(0):
                            results[ticker] = RSIData(
                                ticker=ticker,
                                name=None,
                                rsi_14=None,
                                close=None,
                                data_timestamp=fetch_time,
                                success=False,
                                error="No data returned"
                            )
                            continue
                        hist_df = data[ticker]
                    else:
                        # Single ticker case
                        if len(tickers) == 1:
                            hist_df = data
                        else:
                            results[ticker] = RSIData(
                                ticker=ticker,
                                name=None,
                                rsi_14=None,
                                close=None,
                                data_timestamp=fetch_time,
                                success=False,
                                error="Unexpected data format"
                            )
                            continue
                    
                    if hist_df.empty:
                        results[ticker] = RSIData(
                            ticker=ticker,
                            name=None,
                            rsi_14=None,
                            close=None,
                            data_timestamp=fetch_time,
                            success=False,
                            error="Empty data"
                        )
                        continue
                    
                    # Use adjusted close if available, otherwise use close
                    price_col = "Adj Close" if "Adj Close" in hist_df.columns else "Close"
                    prices = hist_df[price_col].dropna()
                    
                    if len(prices) < MIN_DATA_POINTS:
                        results[ticker] = RSIData(
                            ticker=ticker,
                            name=None,
                            rsi_14=None,
                            close=None,
                            data_timestamp=fetch_time,
                            success=False,
                            error=f"Not enough data ({len(prices)} points)"
                        )
                        continue
                    
                    # Calculate RSI for each requested period
                    rsi_values = {}
                    for period in periods:
                        rsi = calculate_rsi(prices, window=period)
                        if rsi is not None:
                            rsi_values[period] = rsi
                    
                    rsi_14 = rsi_values.get(14)
                    
                    if not rsi_values:
                        results[ticker] = RSIData(
                            ticker=ticker,
                            name=None,
                            rsi_14=None,
                            close=None,
                            data_timestamp=fetch_time,
                            success=False,
                            error="RSI calculation failed"
                        )
                        continue
                    
                    # Get last date and close
                    last_date = prices.index[-1]
                    if hasattr(last_date, 'to_pydatetime'):
                        last_date = last_date.to_pydatetime()
                    elif hasattr(last_date, 'strftime'):
                        last_date = datetime.strptime(last_date.strftime("%Y-%m-%d"), "%Y-%m-%d")
                    else:
                        last_date = fetch_time
                    
                    last_close = float(prices.iloc[-1])
                    
                    results[ticker] = RSIData(
                        ticker=ticker,
                        name=None,  # yfinance doesn't provide name in batch download
                        rsi_14=rsi_14,
                        close=last_close,
                        data_timestamp=last_date,
                        success=True,
                        rsi_values=rsi_values
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    results[ticker] = RSIData(
                        ticker=ticker,
                        name=None,
                        rsi_14=None,
                        close=None,
                        data_timestamp=fetch_time,
                        success=False,
                        error=str(e)
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"yfinance batch fetch error: {e}")
            # Return error results for all tickers
            for ticker in tickers:
                results[ticker] = RSIData(
                    ticker=ticker,
                    name=None,
                    rsi_14=None,
                    close=None,
                    data_timestamp=fetch_time,
                    success=False,
                    error=str(e)
                )
            return results
    
    async def get_rsi_for_tickers(
        self,
        tickers: List[str],
        periods: Optional[List[int]] = None
    ) -> Dict[str, RSIData]:
        """
        Fetch RSI data for multiple tickers.
        
        Args:
            tickers: List of Yahoo Finance ticker symbols
            periods: RSI periods to calculate (default: [14])
        
        Failed tickers are automatically retried up to RETRY_MAX_ATTEMPTS times.
        """
        if not tickers:
            return {}
        
        if periods is None:
            periods = [14]
        
        results: Dict[str, RSIData] = {}
        logger.info(f"Fetching RSI for {len(tickers)} tickers via yfinance")
        
        loop = asyncio.get_event_loop()
        
        # Process in batches
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(tickers) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)")
            
            # Run sync function in executor
            batch_results = await loop.run_in_executor(
                self._executor,
                self._fetch_batch_sync,
                batch,
                periods
            )
            
            results.update(batch_results)
            
            # Delay between batches
            if i + self.batch_size < len(tickers):
                await asyncio.sleep(BATCH_DELAY_SECONDS)
        
        # Collect failed tickers for retry
        failed_tickers = [t for t in tickers if t in results and not results[t].success]
        
        # Retry failed tickers
        if failed_tickers:
            results = await self._retry_failed_tickers(results, failed_tickers, periods, loop)
        
        # Log final summary
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        logger.info(f"yfinance fetch complete: {successful} successful, {failed} failed")
        
        return results
    
    async def _retry_failed_tickers(
        self,
        results: Dict[str, RSIData],
        failed_tickers: List[str],
        periods: List[int],
        loop
    ) -> Dict[str, RSIData]:
        """
        Retry failed tickers with smaller batches and delays.
        
        Args:
            results: Current results dict to update
            failed_tickers: List of tickers that failed
            periods: RSI periods to calculate
            loop: Event loop for executor
        
        Returns:
            Updated results dict
        """
        logger.info(f"Retrying {len(failed_tickers)} failed tickers (up to {RETRY_MAX_ATTEMPTS} attempts)")
        
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            if not failed_tickers:
                break
            
            logger.info(f"Retry attempt {attempt}/{RETRY_MAX_ATTEMPTS} for {len(failed_tickers)} tickers")
            
            # Wait before retry
            await asyncio.sleep(RETRY_DELAY_SECONDS)
            
            still_failed = []
            
            # Process in smaller batches for retries
            for i in range(0, len(failed_tickers), RETRY_BATCH_SIZE):
                batch = failed_tickers[i:i + RETRY_BATCH_SIZE]
                
                logger.info(f"Retry batch: {len(batch)} tickers")
                
                # Run sync function in executor
                batch_results = await loop.run_in_executor(
                    self._executor,
                    self._fetch_batch_sync,
                    batch,
                    periods
                )
                
                # Check results and update
                for ticker in batch:
                    if ticker in batch_results:
                        result = batch_results[ticker]
                        if result.success:
                            results[ticker] = result
                            logger.info(f"Retry successful for {ticker}")
                        else:
                            still_failed.append(ticker)
                
                # Small delay between retry batches
                if i + RETRY_BATCH_SIZE < len(failed_tickers):
                    await asyncio.sleep(RETRY_DELAY_SECONDS / 2)
            
            failed_tickers = still_failed
            
            if not failed_tickers:
                logger.info(f"All retries successful after attempt {attempt}")
                break
        
        if failed_tickers:
            logger.warning(f"Still failed after {RETRY_MAX_ATTEMPTS} retries: {failed_tickers}")
        
        return results
    
    async def get_rsi_single(
        self,
        ticker: str,
        periods: Optional[List[int]] = None
    ) -> RSIData:
        """Fetch RSI for a single ticker."""
        results = await self.get_rsi_for_tickers([ticker], periods)
        return results.get(ticker, RSIData(
            ticker=ticker,
            name=None,
            rsi_14=None,
            close=None,
            data_timestamp=datetime.utcnow(),
            success=False,
            error="Ticker not found in results"
        ))
