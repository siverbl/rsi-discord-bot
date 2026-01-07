"""
Configuration settings for the RSI Discord Bot.

RSI Provider Selection:
-----------------------
The bot supports two RSI data providers:
1. "tradingview" (default) - Uses TradingView Screener API via tradingview_screener package
2. "yfinance" - Uses Yahoo Finance via yfinance package

To switch providers, change the RSI_PROVIDER environment variable or update the default below.
Example: RSI_PROVIDER=yfinance python main.py
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REFDATA_DIR = DATA_DIR / "refdata"
RUNTIME_DIR = PROJECT_ROOT / "runtime"

# Create runtime dir automatically (DB/log need this). Data dir should already exist.
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Core file paths
TICKERS_FILE = DATA_DIR / "tickers.csv"
DB_PATH = RUNTIME_DIR / "rsi_bot.db"
LOG_PATH = RUNTIME_DIR / "rsi_bot.log"

# Environment

# Bot token (set via environment variable or .env loader)
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()

# =============================================================================
# RSI Provider Configuration
# =============================================================================
# Options: "tradingview" (default) or "yfinance"
# TradingView Screener is the default as it provides pre-calculated RSI values
# and is more efficient for batch queries.
RSI_PROVIDER = os.getenv("RSI_PROVIDER", "tradingview").lower().strip()

# TradingView Screener settings
TV_BATCH_SIZE = 50  # Tickers per request (safe limit for TradingView)
TV_BATCH_DELAY_SECONDS = 3.0  # Delay between batches to avoid rate limiting

# Retry settings for failed tickers
RETRY_MAX_ATTEMPTS = 3  # Number of retry attempts for failed tickers
RETRY_DELAY_SECONDS = 5.0  # Delay before each retry attempt
RETRY_BATCH_SIZE = 10  # Smaller batch size for retries (more reliable)

# =============================================================================
# RSI calculation defaults
# =============================================================================
DEFAULT_RSI_PERIOD = 14
DEFAULT_OVERSOLD_THRESHOLD = 30
DEFAULT_OVERBOUGHT_THRESHOLD = 70

# =============================================================================
# Auto-Scan Default Thresholds (can be changed by admin via Discord)
# =============================================================================
DEFAULT_AUTO_OVERSOLD_THRESHOLD = 34  # Default for automatic hourly scans
DEFAULT_AUTO_OVERBOUGHT_THRESHOLD = 70  # Default for automatic hourly scans

# =============================================================================
# Scheduling defaults
# =============================================================================
DEFAULT_TIMEZONE = "Europe/Oslo"
DEFAULT_SCHEDULE_TIME = "18:30"

# Market Hours (Europe/Oslo timezone)
# Norway/Europe market hours: 09:30 - 17:30
EUROPE_MARKET_START_HOUR = 9
EUROPE_MARKET_START_MINUTE = 30
EUROPE_MARKET_END_HOUR = 17
EUROPE_MARKET_END_MINUTE = 30

# US/Canada market hours (in Europe/Oslo time): 15:30 - 22:30
US_MARKET_START_HOUR = 15
US_MARKET_START_MINUTE = 30
US_MARKET_END_HOUR = 22
US_MARKET_END_MINUTE = 30

# =============================================================================
# Anti-spam / alert behavior
# =============================================================================
DEFAULT_COOLDOWN_HOURS = 24
DEFAULT_HYSTERESIS = 2.0
DEFAULT_ALERT_MODE = "CROSSING"  # CROSSING or LEVEL

# =============================================================================
# Data fetching (yfinance fallback)
# =============================================================================
BATCH_SIZE = 100
BATCH_DELAY_SECONDS = 1.5
PRICE_HISTORY_PERIOD = "1y"
MIN_DATA_POINTS = 15  # Minimum data points needed for RSI calculation

# =============================================================================
# Discord rate limits / formatting
# =============================================================================
MAX_ALERTS_PER_MESSAGE = 25
DISCORD_MESSAGE_LIMIT = 2000  # Discord's character limit per message
DISCORD_SAFE_LIMIT = 1900  # Safe limit to allow for formatting

# =============================================================================
# Links
# =============================================================================
TRADINGVIEW_URL_TEMPLATE = "https://www.tradingview.com/chart/?symbol={tradingview_slug}&interval=1D"

# =============================================================================
# Discord channels
# =============================================================================

# Fixed alert channel names (automatic routing)
OVERSOLD_CHANNEL_NAME = "rsi-oversold"       # For UNDER alerts
OVERBOUGHT_CHANNEL_NAME = "rsi-overbought"   # For OVER alerts

# Feature channels
REQUEST_CHANNEL_NAME = "request"             # For ticker add requests
CHANGELOG_CHANNEL_NAME = "server-changelog"  # For server status and admin logs

# =============================================================================
# Market Region Detection
# =============================================================================
# Yahoo Finance suffixes that indicate European markets
EUROPEAN_SUFFIXES = {
    '.OL',   # Oslo Stock Exchange (Norway)
    '.ST',   # Stockholm (Sweden)
    '.CO',   # Copenhagen (Denmark)
    '.HE',   # Helsinki (Finland)
    '.AS',   # Amsterdam (Netherlands)
    '.PA',   # Paris (France)
    '.DE',   # Frankfurt (Germany)
    '.L',    # London (UK)
    '.MI',   # Milan (Italy)
    '.MC',   # Madrid (Spain)
    '.SW',   # Zurich (Switzerland)
    '.VI',   # Vienna (Austria)
    '.BR',   # Brussels (Belgium)
    '.LS',   # Lisbon (Portugal)
    '.AT',   # Athens (Greece)
    '.WA',   # Warsaw (Poland)
    '.PR',   # Prague (Czech Republic)
}

# Yahoo Finance suffixes that indicate US/Canada markets
US_CANADA_SUFFIXES = {
    '.TO',   # Toronto Stock Exchange (Canada)
    '.V',    # TSX Venture (Canada)
    '.NE',   # NEO Exchange (Canada)
    '.CN',   # Canadian Securities Exchange
}

# US stocks typically have no suffix (just the ticker symbol)
