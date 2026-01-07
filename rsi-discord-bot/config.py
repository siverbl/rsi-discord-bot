"""
Configuration settings for the RSI Discord Bot.
"""
import os
from pathlib import Path

# Bot token (set via environment variable)
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")

# File paths
BASE_DIR = Path(__file__).parent
TICKERS_FILE = BASE_DIR / "tickers.csv"
DATABASE_FILE = BASE_DIR / "rsi_bot.db"

# RSI calculation defaults
DEFAULT_RSI_PERIOD = 14
DEFAULT_OVERSOLD_THRESHOLD = 30
DEFAULT_OVERBOUGHT_THRESHOLD = 70

# Scheduling defaults
DEFAULT_TIMEZONE = "Europe/Oslo"
DEFAULT_SCHEDULE_TIME = "18:30"

# Anti-spam defaults
DEFAULT_COOLDOWN_HOURS = 24
DEFAULT_HYSTERESIS = 2.0
DEFAULT_ALERT_MODE = "CROSSING"  # CROSSING or LEVEL

# Data fetching
BATCH_SIZE = 100
BATCH_DELAY_SECONDS = 1.5
PRICE_HISTORY_PERIOD = "1y"
MIN_DATA_POINTS = 15  # Minimum data points needed for RSI calculation

# Discord rate limits
MAX_ALERTS_PER_MESSAGE = 25

# TradingView URL template (replaced Nordnet)
TRADINGVIEW_URL_TEMPLATE = "https://www.tradingview.com/chart/?symbol={tradingview_slug}&interval=1D"

# Reference data directory for exchange lookups
REFDATA_DIR = BASE_DIR / "refdata"

# Fixed alert channel names (no channel selection - automatic routing)
OVERSOLD_CHANNEL_NAME = "rsi-oversold"   # For UNDER alerts
OVERBOUGHT_CHANNEL_NAME = "rsi-overbought"  # For OVER alerts

# Feature channels
REQUEST_CHANNEL_NAME = "request"  # For ticker add requests
CHANGELOG_CHANNEL_NAME = "server-changelog"  # For server status and admin logs

# Server monitoring settings
SERVER_HEALTH_CHECK_URL = os.getenv("SERVER_HEALTH_URL", "")  # e.g., "http://localhost:8080/health"
SERVER_HEALTH_CHECK_INTERVAL = 60  # seconds between health checks
SERVER_HEALTH_CHECK_TIMEOUT = 10  # seconds to wait for response

# Server control scripts (paths to wrapper scripts)
SERVER_RESTART_SCRIPT = os.getenv("SERVER_RESTART_SCRIPT", "")  # e.g., "/opt/scripts/restart_server.sh"
SERVER_SHUTDOWN_SCRIPT = os.getenv("SERVER_SHUTDOWN_SCRIPT", "")  # e.g., "/opt/scripts/shutdown_server.sh"
