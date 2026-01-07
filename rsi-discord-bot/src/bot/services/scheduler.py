"""
Scheduler module for RSI Discord Bot.
Handles the daily scheduled RSI check and alert delivery.
"""
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple

import discord
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from bot.config import (
    DEFAULT_TIMEZONE, DEFAULT_SCHEDULE_TIME,
    OVERSOLD_CHANNEL_NAME, OVERBOUGHT_CHANNEL_NAME
)
from bot.repositories.database import Database
from bot.services.market_data.rsi_calculator import RSICalculator
from bot.cogs.alert_engine import AlertEngine, format_alert_list

logger = logging.getLogger(__name__)


def get_alert_channels(guild: discord.Guild) -> Tuple[Optional[discord.TextChannel], Optional[discord.TextChannel]]:
    """
    Get the fixed alert channels for a guild.
    
    Returns:
        Tuple of (oversold_channel, overbought_channel)
    """
    oversold_channel = discord.utils.get(guild.text_channels, name=OVERSOLD_CHANNEL_NAME)
    overbought_channel = discord.utils.get(guild.text_channels, name=OVERBOUGHT_CHANNEL_NAME)
    return oversold_channel, overbought_channel


def can_send_to_channel(channel: discord.TextChannel, bot_member: discord.Member) -> bool:
    """Check if the bot can send messages to a channel."""
    if not channel:
        return False
    perms = channel.permissions_for(bot_member)
    return perms.send_messages


class RSIScheduler:
    """
    Manages the scheduled daily RSI check job.
    """

    def __init__(self, bot):
        self.bot = bot
        self.db: Database = bot.db
        self.rsi_calculator = RSICalculator()
        self.alert_engine = AlertEngine(self.db)
        self.scheduler = AsyncIOScheduler()
        self.timezone = pytz.timezone(DEFAULT_TIMEZONE)

        # Track scheduled jobs per guild
        self._guild_jobs: Dict[int, str] = {}

    async def start(self):
        """Start the scheduler and set up default job."""
        logger.info("Starting RSI scheduler...")

        # Add default job at configured time
        self._add_default_job()

        self.scheduler.start()
        logger.info("RSI scheduler started")

    def _add_default_job(self):
        """Add the default scheduled job."""
        try:
            hour, minute = map(int, DEFAULT_SCHEDULE_TIME.split(":"))
        except ValueError:
            hour, minute = 18, 30

        trigger = CronTrigger(
            hour=hour,
            minute=minute,
            timezone=self.timezone
        )

        job = self.scheduler.add_job(
            self._run_daily_check,
            trigger=trigger,
            id="daily_rsi_check",
            name="Daily RSI Check",
            replace_existing=True
        )

        logger.info(f"Scheduled daily RSI check at {hour:02d}:{minute:02d} {DEFAULT_TIMEZONE}")

    async def _run_daily_check(self):
        """
        Execute the daily RSI check for all guilds.
        """
        start_time = datetime.now(self.timezone)
        logger.info(f"Starting daily RSI check at {start_time.isoformat()}")

        try:
            # Step 1: Load all active subscriptions
            subscriptions_data = await self.db.get_subscriptions_with_state()

            if not subscriptions_data:
                logger.info("No active subscriptions found")
                return

            logger.info(f"Found {len(subscriptions_data)} active subscriptions")

            # Step 2: Determine unique tickers and periods needed
            ticker_periods: Dict[str, List[int]] = {}
            guilds_with_subs: Set[int] = set()

            for sub in subscriptions_data:
                ticker = sub['ticker']
                period = sub['period']
                guild_id = sub['guild_id']

                if ticker not in ticker_periods:
                    ticker_periods[ticker] = []
                if period not in ticker_periods[ticker]:
                    ticker_periods[ticker].append(period)

                guilds_with_subs.add(guild_id)

            logger.info(
                f"Need RSI data for {len(ticker_periods)} tickers "
                f"across {len(guilds_with_subs)} guilds"
            )

            # Step 3: Fetch historical data and calculate RSI
            rsi_results = await self.rsi_calculator.calculate_rsi_for_tickers(
                ticker_periods
            )

            successful = sum(1 for r in rsi_results.values() if r.success)
            failed = len(rsi_results) - successful
            logger.info(f"RSI calculation: {successful} success, {failed} failed")

            # Log failed tickers
            for ticker, result in rsi_results.items():
                if not result.success:
                    logger.warning(f"Failed to get RSI for {ticker}: {result.error}")

            # Step 4: Evaluate subscriptions and generate alerts
            alerts_by_condition = await self.alert_engine.evaluate_subscriptions(
                rsi_results, dry_run=False
            )

            under_alerts = alerts_by_condition.get('UNDER', [])
            over_alerts = alerts_by_condition.get('OVER', [])
            total_alerts = len(under_alerts) + len(over_alerts)
            
            logger.info(f"Generated {total_alerts} alerts (UNDER: {len(under_alerts)}, OVER: {len(over_alerts)})")

            # Step 5: Send alerts to channels (grouped by guild)
            sent_count = 0
            error_count = 0

            # Group alerts by guild
            alerts_by_guild: Dict[int, Dict[str, List]] = {}
            for alert in under_alerts:
                if alert.guild_id not in alerts_by_guild:
                    alerts_by_guild[alert.guild_id] = {'UNDER': [], 'OVER': []}
                alerts_by_guild[alert.guild_id]['UNDER'].append(alert)
            
            for alert in over_alerts:
                if alert.guild_id not in alerts_by_guild:
                    alerts_by_guild[alert.guild_id] = {'UNDER': [], 'OVER': []}
                alerts_by_guild[alert.guild_id]['OVER'].append(alert)

            # Process each guild
            for guild_id in guilds_with_subs:
                guild = self.bot.get_guild(guild_id)
                if not guild:
                    logger.warning(f"Guild {guild_id} not found")
                    continue

                oversold_ch, overbought_ch = get_alert_channels(guild)
                
                if not oversold_ch:
                    logger.warning(f"Channel #{OVERSOLD_CHANNEL_NAME} not found in guild {guild_id}")
                if not overbought_ch:
                    logger.warning(f"Channel #{OVERBOUGHT_CHANNEL_NAME} not found in guild {guild_id}")

                guild_alerts = alerts_by_guild.get(guild_id, {'UNDER': [], 'OVER': []})

                # Send UNDER alerts to oversold channel
                if oversold_ch:
                    if can_send_to_channel(oversold_ch, guild.me):
                        try:
                            if guild_alerts['UNDER']:
                                messages = format_alert_list(guild_alerts['UNDER'], 'UNDER')
                                for msg in messages:
                                    await oversold_ch.send(msg)
                                    sent_count += 1
                            else:
                                # Only send "no alerts" if there are subscriptions but no triggers
                                pass  # Don't spam with "no alerts" messages daily
                        except discord.Forbidden:
                            logger.error(f"Permission denied sending to #{OVERSOLD_CHANNEL_NAME} in guild {guild_id}")
                            error_count += 1
                        except Exception as e:
                            logger.error(f"Error sending to #{OVERSOLD_CHANNEL_NAME} in guild {guild_id}: {e}")
                            error_count += 1
                    else:
                        logger.warning(f"No permission to send to #{OVERSOLD_CHANNEL_NAME} in guild {guild_id}")

                # Send OVER alerts to overbought channel
                if overbought_ch:
                    if can_send_to_channel(overbought_ch, guild.me):
                        try:
                            if guild_alerts['OVER']:
                                messages = format_alert_list(guild_alerts['OVER'], 'OVER')
                                for msg in messages:
                                    await overbought_ch.send(msg)
                                    sent_count += 1
                            else:
                                pass  # Don't spam with "no alerts" messages daily
                        except discord.Forbidden:
                            logger.error(f"Permission denied sending to #{OVERBOUGHT_CHANNEL_NAME} in guild {guild_id}")
                            error_count += 1
                        except Exception as e:
                            logger.error(f"Error sending to #{OVERBOUGHT_CHANNEL_NAME} in guild {guild_id}: {e}")
                            error_count += 1
                    else:
                        logger.warning(f"No permission to send to #{OVERBOUGHT_CHANNEL_NAME} in guild {guild_id}")

            # Step 6: Log completion
            end_time = datetime.now(self.timezone)
            duration = (end_time - start_time).total_seconds()

            logger.info(
                f"Daily RSI check complete in {duration:.1f}s - "
                f"Tickers: {successful}/{len(ticker_periods)} | "
                f"Subscriptions: {len(subscriptions_data)} | "
                f"Alerts: {total_alerts} | "
                f"Messages sent: {sent_count} | "
                f"Errors: {error_count}"
            )

        except Exception as e:
            logger.error(f"Error in daily RSI check: {e}", exc_info=True)

    async def run_for_guild(self, guild_id: int, dry_run: bool = False) -> dict:
        """
        Run RSI check for a specific guild.

        Returns:
            Dict with results summary
        """
        logger.info(f"Running RSI check for guild {guild_id} (dry_run={dry_run})")

        # Get subscriptions for this guild
        subs = await self.db.get_subscriptions_by_guild(
            guild_id=guild_id, enabled_only=True
        )

        if not subs:
            return {
                "success": True,
                "message": "No active subscriptions",
                "subscriptions": 0,
                "tickers": 0,
                "alerts": 0
            }

        # Determine unique tickers and periods
        ticker_periods: Dict[str, List[int]] = {}
        for sub in subs:
            if sub.ticker not in ticker_periods:
                ticker_periods[sub.ticker] = []
            if sub.period not in ticker_periods[sub.ticker]:
                ticker_periods[sub.ticker].append(sub.period)

        # Calculate RSI
        rsi_results = await self.rsi_calculator.calculate_rsi_for_tickers(
            ticker_periods
        )

        successful = sum(1 for r in rsi_results.values() if r.success)
        failed = len(rsi_results) - successful

        # Evaluate subscriptions
        alerts_by_condition = await self.alert_engine.evaluate_subscriptions(
            rsi_results, dry_run=dry_run
        )

        under_alerts = alerts_by_condition.get('UNDER', [])
        over_alerts = alerts_by_condition.get('OVER', [])
        total_alerts = len(under_alerts) + len(over_alerts)

        return {
            "success": True,
            "subscriptions": len(subs),
            "tickers_requested": len(ticker_periods),
            "tickers_success": successful,
            "tickers_failed": failed,
            "alerts": total_alerts,
            "under_alerts": under_alerts,
            "over_alerts": over_alerts,
            "rsi_results": rsi_results
        }

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("RSI scheduler stopped")


async def setup_scheduler(bot):
    """
    Set up the scheduler for a bot instance.
    """
    scheduler = RSIScheduler(bot)
    await scheduler.start()
    return scheduler
