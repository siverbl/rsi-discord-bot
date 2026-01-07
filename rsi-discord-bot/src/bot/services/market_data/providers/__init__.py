"""
RSI Data Providers Package.

This package provides a unified interface for fetching RSI data from different sources.
The bot can switch between providers by changing the RSI_PROVIDER config setting.

Supported Providers:
- "tradingview" (default): Uses TradingView Screener API
- "yfinance": Uses Yahoo Finance via yfinance package

Usage:
    from bot.services.market_data.providers import get_provider
    
    provider = get_provider()  # Uses RSI_PROVIDER config
    results = await provider.get_rsi_for_tickers(["AAPL", "MSFT"])
"""
import logging
from typing import Optional

from bot.config import RSI_PROVIDER
from bot.services.market_data.providers.base import RSIProviderBase, RSIData

logger = logging.getLogger(__name__)

# Global cached provider instance
_provider_instance: Optional[RSIProviderBase] = None


def get_provider(provider_name: Optional[str] = None) -> RSIProviderBase:
    """
    Get an RSI provider instance.
    
    Args:
        provider_name: Optional provider name override ("tradingview" or "yfinance").
                      If not specified, uses RSI_PROVIDER from config.
    
    Returns:
        RSIProviderBase implementation
    
    Raises:
        ValueError: If provider_name is not supported
    """
    global _provider_instance
    
    name = provider_name or RSI_PROVIDER
    name = name.lower().strip()
    
    # Return cached instance if available and matches requested provider
    if _provider_instance is not None:
        if (name == "tradingview" and _provider_instance.name == "TradingView Screener") or \
           (name == "yfinance" and _provider_instance.name == "Yahoo Finance (yfinance)"):
            return _provider_instance
    
    if name == "tradingview":
        from bot.services.market_data.providers.tradingview_provider import TradingViewProvider
        _provider_instance = TradingViewProvider()
        logger.info(f"Using RSI provider: {_provider_instance.name}")
        return _provider_instance
    
    elif name == "yfinance":
        from bot.services.market_data.providers.yfinance_provider import YFinanceProvider
        _provider_instance = YFinanceProvider()
        logger.info(f"Using RSI provider: {_provider_instance.name}")
        return _provider_instance
    
    else:
        raise ValueError(
            f"Unknown RSI provider: '{name}'. "
            f"Supported providers: 'tradingview', 'yfinance'"
        )


def reset_provider():
    """Reset the cached provider instance (useful for testing)."""
    global _provider_instance
    _provider_instance = None


__all__ = [
    'RSIProviderBase',
    'RSIData',
    'get_provider',
    'reset_provider',
]
