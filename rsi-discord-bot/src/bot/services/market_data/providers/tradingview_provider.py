"""
TradingView Screener RSI Provider.

Uses the tradingview_screener package to fetch RSI data from TradingView's screener API.
This is the default provider as it provides pre-calculated RSI values efficiently.

Documentation: https://shner-elmo.github.io/TradingView-Screener/
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

from bot.config import TV_BATCH_SIZE, TV_BATCH_DELAY_SECONDS
from bot.services.market_data.providers.base import RSIProviderBase, RSIData

logger = logging.getLogger(__name__)


def yahoo_to_tradingview(yahoo_ticker: str) -> Optional[str]:
    """
    Convert Yahoo Finance ticker format to TradingView format.
    
    Yahoo Format: EQNR.OL, AAPL, TSLA
    TradingView Format: OSL:EQNR, NASDAQ:AAPL, NASDAQ:TSLA
    
    This is a best-effort conversion. Some tickers may not map correctly.
    """
    yahoo_ticker = yahoo_ticker.upper().strip()
    
    # Common Yahoo suffix to TradingView exchange mapping
    suffix_map = {
        '.OL': 'OSL',       # Oslo Stock Exchange
        '.ST': 'STO',       # Stockholm (OMX Stockholm)
        '.CO': 'CSE',       # Copenhagen
        '.HE': 'HEL',       # Helsinki
        '.AS': 'EURONEXT',  # Amsterdam
        '.PA': 'EURONEXT',  # Paris
        '.DE': 'XETR',      # Frankfurt (Xetra)
        '.F': 'FWB',        # Frankfurt
        '.L': 'LSE',        # London
        '.MI': 'MIL',       # Milan
        '.MC': 'BME',       # Madrid
        '.SW': 'SIX',       # Zurich
        '.VI': 'VIE',       # Vienna
        '.BR': 'EURONEXT',  # Brussels
        '.LS': 'EURONEXT',  # Lisbon
        '.TO': 'TSX',       # Toronto
        '.V': 'TSXV',       # TSX Venture
        '.HK': 'HKEX',      # Hong Kong
        '.T': 'TSE',        # Tokyo
        '.SS': 'SSE',       # Shanghai
        '.SZ': 'SZSE',      # Shenzhen
    }
    
    # Check if ticker has a suffix
    for suffix, exchange in suffix_map.items():
        if yahoo_ticker.endswith(suffix):
            base_symbol = yahoo_ticker[:-len(suffix)]
            return f"{exchange}:{base_symbol}"
    
    # No suffix - assume US stock
    # Try to determine exchange (simplified logic)
    # In reality, you'd need a lookup table for accurate mapping
    return f"NASDAQ:{yahoo_ticker}"  # Default to NASDAQ


class TradingViewProvider(RSIProviderBase):
    """
    RSI provider using TradingView Screener API.
    
    Features:
    - Batch queries for efficiency (up to 50 tickers per request)
    - Pre-calculated RSI values from TradingView
    - Includes timestamp information
    """
    
    def __init__(self, batch_size: int = TV_BATCH_SIZE):
        self.batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    @property
    def name(self) -> str:
        return "TradingView Screener"
    
    def _fetch_batch_sync(
        self,
        tv_tickers: List[str],
        yahoo_tickers: List[str]
    ) -> Dict[str, RSIData]:
        """
        Synchronously fetch RSI data for a batch of tickers.
        
        Args:
            tv_tickers: List of TradingView-formatted tickers (e.g., "OSL:EQNR")
            yahoo_tickers: Corresponding Yahoo Finance tickers for result mapping
        
        Returns:
            Dict mapping Yahoo ticker -> RSIData
        """
        from tradingview_screener import Query
        
        results = {}
        fetch_time = datetime.utcnow()
        
        try:
            # Build query for specific tickers
            # Request: name, close, RSI, and update_mode for timestamp info
            query = (
                Query()
                .select('name', 'close', 'RSI', 'RSI[1]', 'update_mode')
                .set_tickers(*tv_tickers)
                .limit(len(tv_tickers))
            )
            
            # Execute query
            count, df = query.get_scanner_data()
            
            if df is None or df.empty:
                # No data returned
                for yf_ticker in yahoo_tickers:
                    results[yf_ticker] = RSIData(
                        ticker=yf_ticker,
                        name=None,
                        rsi_14=None,
                        close=None,
                        data_timestamp=fetch_time,
                        success=False,
                        error="No data from TradingView"
                    )
                return results
            
            # Map TradingView tickers back to Yahoo tickers
            tv_to_yahoo = dict(zip(tv_tickers, yahoo_tickers))
            
            # Process results
            for _, row in df.iterrows():
                tv_ticker = row.get('ticker', '')
                if tv_ticker not in tv_to_yahoo:
                    continue
                
                yf_ticker = tv_to_yahoo[tv_ticker]
                
                try:
                    rsi_value = row.get('RSI')
                    if rsi_value is not None:
                        rsi_value = float(rsi_value)
                    
                    close_value = row.get('close')
                    if close_value is not None:
                        close_value = float(close_value)
                    
                    name = row.get('name', yf_ticker)
                    
                    results[yf_ticker] = RSIData(
                        ticker=yf_ticker,
                        name=str(name) if name else None,
                        rsi_14=rsi_value,
                        close=close_value,
                        data_timestamp=fetch_time,
                        success=rsi_value is not None,
                        error=None if rsi_value is not None else "RSI value not available",
                        rsi_values={14: rsi_value} if rsi_value is not None else None
                    )
                except Exception as e:
                    results[yf_ticker] = RSIData(
                        ticker=yf_ticker,
                        name=None,
                        rsi_14=None,
                        close=None,
                        data_timestamp=fetch_time,
                        success=False,
                        error=str(e)
                    )
            
            # Mark any missing tickers as failed
            for yf_ticker in yahoo_tickers:
                if yf_ticker not in results:
                    results[yf_ticker] = RSIData(
                        ticker=yf_ticker,
                        name=None,
                        rsi_14=None,
                        close=None,
                        data_timestamp=fetch_time,
                        success=False,
                        error="Ticker not found in TradingView results"
                    )
            
            return results
            
        except Exception as e:
            logger.error(f"TradingView batch fetch error: {e}")
            # Return error results for all tickers in batch
            for yf_ticker in yahoo_tickers:
                results[yf_ticker] = RSIData(
                    ticker=yf_ticker,
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
        
        Note: TradingView Screener only provides RSI14 (14-period RSI).
        Other periods are not available through this API.
        """
        if not tickers:
            return {}
        
        results: Dict[str, RSIData] = {}
        
        # Convert Yahoo tickers to TradingView format
        ticker_mapping = []  # List of (tv_ticker, yahoo_ticker) pairs
        for yf_ticker in tickers:
            tv_ticker = yahoo_to_tradingview(yf_ticker)
            if tv_ticker:
                ticker_mapping.append((tv_ticker, yf_ticker))
            else:
                # Can't convert - mark as failed immediately
                results[yf_ticker] = RSIData(
                    ticker=yf_ticker,
                    name=None,
                    rsi_14=None,
                    close=None,
                    data_timestamp=datetime.utcnow(),
                    success=False,
                    error="Could not convert to TradingView format"
                )
        
        if not ticker_mapping:
            return results
        
        logger.info(f"Fetching RSI for {len(ticker_mapping)} tickers via TradingView Screener")
        
        # Process in batches
        loop = asyncio.get_event_loop()
        
        for i in range(0, len(ticker_mapping), self.batch_size):
            batch = ticker_mapping[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(ticker_mapping) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tickers)")
            
            tv_tickers = [t[0] for t in batch]
            yf_tickers = [t[1] for t in batch]
            
            # Run sync function in executor
            batch_results = await loop.run_in_executor(
                self._executor,
                self._fetch_batch_sync,
                tv_tickers,
                yf_tickers
            )
            
            results.update(batch_results)
            
            # Delay between batches
            if i + self.batch_size < len(ticker_mapping):
                await asyncio.sleep(TV_BATCH_DELAY_SECONDS)
        
        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        failed = len(results) - successful
        logger.info(f"TradingView fetch complete: {successful} successful, {failed} failed")
        
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
