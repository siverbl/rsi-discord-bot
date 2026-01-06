"""
Ticker Request Handler for RSI Discord Bot.
Handles automatic ticker addition from #request channel messages.

Expected message format (3 lines):
    https://finance.yahoo.com/quote/CINT.ST/
    Cint Group AB
    https://www.nordnet.no/aksjer/kurser/cint-group-cint-xsto
"""
import asyncio
import csv
import re
import logging
from pathlib import Path
from typing import Optional, Tuple

import discord

from config import TICKERS_FILE, REQUEST_CHANNEL_NAME

logger = logging.getLogger(__name__)

# Async lock for thread-safe file access
_csv_lock = asyncio.Lock()

# Regex patterns for URL parsing
YAHOO_URL_PATTERN = re.compile(
    r'https?://(?:www\.)?finance\.yahoo\.com/quote/([A-Za-z0-9\.\-\^]+)/?',
    re.IGNORECASE
)
NORDNET_URL_PATTERN = re.compile(
    r'https?://(?:www\.)?nordnet\.no/aksjer/kurser/([a-z0-9\-]+)/?',
    re.IGNORECASE
)


def parse_ticker_request(content: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse a ticker request message.
    
    Args:
        content: Message content with 3 lines
        
    Returns:
        Tuple of (ticker, name, nordnet_slug, error_message)
        If parsing fails, first 3 values are None and error_message explains why.
    """
    # Split into lines and clean up
    lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
    
    if len(lines) != 3:
        return None, None, None, f"Expected 3 lines, got {len(lines)}. Format:\n```\nhttps://finance.yahoo.com/quote/TICKER/\nCompany Name\nhttps://www.nordnet.no/aksjer/kurser/slug\n```"
    
    yahoo_line = lines[0]
    name_line = lines[1]
    nordnet_line = lines[2]
    
    # Parse Yahoo URL for ticker
    yahoo_match = YAHOO_URL_PATTERN.search(yahoo_line)
    if not yahoo_match:
        return None, None, None, f"Could not parse Yahoo Finance URL: `{yahoo_line}`\nExpected format: `https://finance.yahoo.com/quote/TICKER/`"
    
    ticker = yahoo_match.group(1).upper()
    
    # Validate name (should not be a URL)
    if name_line.startswith('http'):
        return None, None, None, f"Line 2 should be the company name, not a URL: `{name_line}`"
    
    name = name_line.strip()
    if not name:
        return None, None, None, "Company name (line 2) cannot be empty"
    
    # Parse Nordnet URL for slug
    nordnet_match = NORDNET_URL_PATTERN.search(nordnet_line)
    if not nordnet_match:
        return None, None, None, f"Could not parse Nordnet URL: `{nordnet_line}`\nExpected format: `https://www.nordnet.no/aksjer/kurser/slug`"
    
    nordnet_slug = nordnet_match.group(1).lower()
    
    return ticker, name, nordnet_slug, None


async def ticker_exists(ticker: str) -> bool:
    """
    Check if a ticker already exists in tickers.csv (case-insensitive).
    """
    async with _csv_lock:
        if not TICKERS_FILE.exists():
            return False
        
        try:
            with open(TICKERS_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('ticker', '').upper() == ticker.upper():
                        return True
        except Exception as e:
            logger.error(f"Error checking ticker existence: {e}")
            return False
    
    return False


async def add_ticker(ticker: str, name: str, nordnet_slug: str) -> Tuple[bool, str]:
    """
    Add a ticker to tickers.csv.
    
    Args:
        ticker: Yahoo Finance ticker symbol
        name: Company name
        nordnet_slug: Nordnet URL slug
        
    Returns:
        Tuple of (success, message)
    """
    async with _csv_lock:
        try:
            # Check if file exists and has header
            file_exists = TICKERS_FILE.exists()
            needs_header = not file_exists
            
            if file_exists:
                # Check if file is empty or has no header
                with open(TICKERS_FILE, 'r', newline='', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if not first_line or first_line != 'ticker,name,nordnet_slug':
                        needs_header = True
            
            # Check for duplicate (case-insensitive)
            if file_exists:
                with open(TICKERS_FILE, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('ticker', '').upper() == ticker.upper():
                            return False, f"Ticker `{ticker}` already exists in catalog"
            
            # Append to file
            with open(TICKERS_FILE, 'a', newline='', encoding='utf-8') as f:
                if needs_header:
                    f.write('ticker,name,nordnet_slug\n')
                
                # Write the new row
                writer = csv.writer(f)
                writer.writerow([ticker, name, nordnet_slug])
            
            logger.info(f"Added ticker: {ticker} ({name})")
            return True, f"✅ Added `{ticker}` — {name}"
            
        except Exception as e:
            logger.error(f"Error adding ticker: {e}")
            return False, f"❌ Error adding ticker: {str(e)}"


async def handle_request_message(message: discord.Message) -> Optional[str]:
    """
    Handle a message in the #request channel.
    
    Args:
        message: Discord message
        
    Returns:
        Response message to send, or None if message should be ignored
    """
    # Ignore bot messages
    if message.author.bot:
        return None
    
    # Only process in #request channel
    if message.channel.name != REQUEST_CHANNEL_NAME:
        return None
    
    # Parse the request
    ticker, name, nordnet_slug, error = parse_ticker_request(message.content)
    
    if error:
        return f"❌ **Parse Error**\n{error}"
    
    # Check if already exists
    if await ticker_exists(ticker):
        return f"ℹ️ Ticker `{ticker}` already exists in catalog"
    
    # Add the ticker
    success, response = await add_ticker(ticker, name, nordnet_slug)
    
    return response


class TickerRequestCog:
    """
    Handles ticker request messages in #request channel.
    Integrated into the main bot.
    """
    
    def __init__(self, bot):
        self.bot = bot
    
    async def on_message(self, message: discord.Message):
        """Process messages in #request channel."""
        # Only process in #request channel
        if not hasattr(message.channel, 'name') or message.channel.name != REQUEST_CHANNEL_NAME:
            return
        
        response = await handle_request_message(message)
        
        if response:
            try:
                await message.reply(response, mention_author=False)
            except discord.HTTPException as e:
                logger.error(f"Failed to reply to request message: {e}")
