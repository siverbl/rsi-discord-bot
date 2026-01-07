"""
Configuration settings for the RSI Discord Bot.
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

# RSI calculation defaults

DEFAULT_RSI_PERIOD = 14
DEFAULT_OVERSOLD_THRESHOLD = 30
DEFAULT_OVERBOUGHT_THRESHOLD = 70

# Scheduling defaults

DEFAULT_TIMEZONE = "Europe/Oslo"
DEFAULT_SCHEDULE_TIME = "18:30"

# Anti-spam / alert behavior

DEFAULT_COOLDOWN_HOURS = 24
DEFAULT_HYSTERESIS = 2.0
DEFAULT_ALERT_MODE = "CROSSING"  # CROSSING or LEVEL

# Data fetching

BATCH_SIZE = 100
BATCH_DELAY_SECONDS = 1.5
PRICE_HISTORY_PERIOD = "1y"
MIN_DATA_POINTS = 15  # Minimum data points needed for RSI calculation

# Discord rate limits / formatting
MAX_ALERTS_PER_MESSAGE = 25

# Links

TRADINGVIEW_URL_TEMPLATE = "https://www.tradingview.com/chart/?symbol={tradingview_slug}&interval=1D"

# Discord channels

# Fixed alert channel names (automatic routing)
OVERSOLD_CHANNEL_NAME = "rsi-oversold"       # For UNDER alerts
OVERBOUGHT_CHANNEL_NAME = "rsi-overbought"   # For OVER alerts

# Feature channels
REQUEST_CHANNEL_NAME = "request"             # For ticker add requests
CHANGELOG_CHANNEL_NAME = "server-changelog"  # For server status and admin logs
