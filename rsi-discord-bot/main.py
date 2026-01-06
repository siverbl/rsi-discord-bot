#!/usr/bin/env python3
"""
RSI Discord Bot - Main Entry Point

A Discord bot that sends RSI alerts for stocks.
Supports scheduled daily checks and slash commands for subscription management.

Usage:
    export DISCORD_TOKEN=your_bot_token
    python main.py

Or:
    DISCORD_TOKEN=your_bot_token python main.py
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

import discord
import pytz
from discord import app_commands
from discord.ext import commands
from health_server import start_health_server

from config import (
    DISCORD_TOKEN, DEFAULT_RSI_PERIOD, DEFAULT_OVERSOLD_THRESHOLD,
    DEFAULT_OVERBOUGHT_THRESHOLD, DEFAULT_COOLDOWN_HOURS,
    OVERSOLD_CHANNEL_NAME, OVERBOUGHT_CHANNEL_NAME,
    CHANGELOG_CHANNEL_NAME, REQUEST_CHANNEL_NAME, DEFAULT_TIMEZONE
)
from database import Database
from ticker_catalog import get_catalog, validate_ticker
from rsi_calculator import RSICalculator
from alert_engine import AlertEngine, format_alert_list, format_no_alerts_message
from scheduler import RSIScheduler
from ticker_request import TickerRequestCog, handle_request_message
from server_monitor import ServerMonitor, ActionType, log_admin_action

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rsi_bot.log')
    ]
)
logger = logging.getLogger(__name__)


def get_alert_channels(guild: discord.Guild) -> Tuple[Optional[discord.TextChannel], Optional[discord.TextChannel], str]:
    """
    Get the fixed alert channels for a guild and verify permissions.
    
    Returns:
        Tuple of (oversold_channel, overbought_channel, error_message)
        If channels are found and have permissions, error_message is empty.
    """
    oversold_channel = discord.utils.get(guild.text_channels, name=OVERSOLD_CHANNEL_NAME)
    overbought_channel = discord.utils.get(guild.text_channels, name=OVERBOUGHT_CHANNEL_NAME)
    
    errors = []
    
    # Check if channels exist
    if not oversold_channel:
        errors.append(f"Channel `#{OVERSOLD_CHANNEL_NAME}` not found")
    if not overbought_channel:
        errors.append(f"Channel `#{OVERBOUGHT_CHANNEL_NAME}` not found")
    
    # Check bot permissions in each channel
    bot_member = guild.me
    if oversold_channel:
        perms = oversold_channel.permissions_for(bot_member)
        if not perms.send_messages:
            errors.append(f"Bot lacks **Send Messages** permission in `#{OVERSOLD_CHANNEL_NAME}`")
        if not perms.embed_links:
            errors.append(f"Bot lacks **Embed Links** permission in `#{OVERSOLD_CHANNEL_NAME}` (recommended)")
    
    if overbought_channel:
        perms = overbought_channel.permissions_for(bot_member)
        if not perms.send_messages:
            errors.append(f"Bot lacks **Send Messages** permission in `#{OVERBOUGHT_CHANNEL_NAME}`")
        if not perms.embed_links:
            errors.append(f"Bot lacks **Embed Links** permission in `#{OVERBOUGHT_CHANNEL_NAME}` (recommended)")
    
    error_msg = ""
    if errors:
        error_msg = (
            "❌ **Channel/Permission Issues:**\n" +
            "\n".join(f"• {e}" for e in errors) +
            "\n\n**To fix:**\n"
            "1. Create the channels if they don't exist\n"
            "2. Go to channel settings → Permissions\n"
            "3. Add the bot role and enable **Send Messages** and **Embed Links**"
        )
    
    return oversold_channel, overbought_channel, error_msg

class RSIBot(commands.Bot):
    """Discord bot for RSI alerts with integrated scheduler."""

    def __init__(self):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        intents.message_content = True  # Required for reading message content in #request

        super().__init__(
            command_prefix="!",  # Not used, but required
            intents=intents
        )

        self.db = Database()
        self.catalog = get_catalog()
        self.rsi_calculator = RSICalculator()
        self.alert_engine = AlertEngine(self.db)
        self.scheduler: Optional[RSIScheduler] = None
        self.server_monitor: Optional[ServerMonitor] = None
        self.ticker_request_handler = TickerRequestCog(self)

        self.health_runner = None  # add this

    async def setup_hook(self):
        """Initialize bot components."""
        logger.info("Initializing database...")
        await self.db.initialize()

        logger.info("Loading ticker catalog...")
        self.catalog.load()

        logger.info("Starting scheduler...")
        self.scheduler = RSIScheduler(self)
        await self.scheduler.start()

        logger.info("Starting local health server (localhost-only)...")
        try:
            self.health_runner = await start_health_server(host="127.0.0.1", port=8080)
        except OSError as e:
            logger.error(f"Failed to start health server on 127.0.0.1:8080: {e}")
            # Optional: if you want, disable server monitor here or let it warn.

        logger.info("Starting server monitor...")
        self.server_monitor = ServerMonitor(self)
        await self.server_monitor.start()

        logger.info("Syncing slash commands...")
        await self.tree.sync()



        logger.info("Bot setup complete")

    async def on_ready(self):
        """Called when bot is ready."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Connected to {len(self.guilds)} guilds")
        logger.info(f"Ticker catalog contains {len(self.catalog)} instruments")

        # Set presence
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="RSI levels"
            )
        )

    async def on_message(self, message: discord.Message):
        """Handle messages - used for #request channel ticker additions."""
        # Ignore bot messages
        if message.author.bot:
            return
        
        # Handle ticker requests in #request channel
        if hasattr(message.channel, 'name') and message.channel.name == REQUEST_CHANNEL_NAME:
            response = await handle_request_message(message)
            if response:
                try:
                    await message.reply(response, mention_author=False)
                    # Reload catalog if ticker was added
                    if response.startswith("✅"):
                        self.catalog.reload()
                except discord.HTTPException as e:
                    logger.error(f"Failed to reply to request: {e}")

    async def close(self):
        """Clean shutdown."""
        if self.scheduler:
            self.scheduler.stop()
        if self.server_monitor:
            await self.server_monitor.stop()

        if self.health_runner:
            await self.health_runner.cleanup()

        await super().close()


# Create bot instance
bot = RSIBot()


# ==================== Slash Commands ====================

@bot.tree.command(name="subscribe", description="Create an RSI alert subscription")
@app_commands.describe(
    ticker="Stock ticker symbol (must exist in tickers.csv)",
    condition="Alert condition: 'under' or 'over'",
    threshold="RSI threshold value (0-100)",
    period="RSI period (default: server default or 14)",
    cooldown="Hours between alerts for same rule (default: server default or 24)"
)
@app_commands.choices(condition=[
    app_commands.Choice(name="under (oversold)", value="UNDER"),
    app_commands.Choice(name="over (overbought)", value="OVER")
])
async def subscribe(
    interaction: discord.Interaction,
    ticker: str,
    condition: app_commands.Choice[str],
    threshold: float,
    period: Optional[int] = None,
    cooldown: Optional[int] = None
):
    """Create a new RSI alert subscription."""
    await interaction.response.defer(ephemeral=True)

    # Validate ticker
    is_valid, error = validate_ticker(ticker)
    if not is_valid:
        await interaction.followup.send(f"❌ {error}", ephemeral=True)
        return

    # Validate threshold
    if not 0 <= threshold <= 100:
        await interaction.followup.send(
            "❌ Threshold must be between 0 and 100",
            ephemeral=True
        )
        return

    # Validate period
    if period is not None and not 2 <= period <= 200:
        await interaction.followup.send(
            "❌ Period must be between 2 and 200",
            ephemeral=True
        )
        return

    # Check that alert channels exist
    oversold_ch, overbought_ch, error_msg = get_alert_channels(interaction.guild)
    if error_msg:
        await interaction.followup.send(error_msg, ephemeral=True)
        return

    # Get guild config for defaults
    config = await bot.db.get_or_create_guild_config(interaction.guild_id)

    # Apply defaults
    target_period = period if period is not None else config.default_rsi_period
    target_cooldown = cooldown if cooldown is not None else config.default_cooldown_hours

    ticker = ticker.upper().strip()

    # Determine target channel based on condition
    if condition.value == "UNDER":
        target_channel = oversold_ch
    else:
        target_channel = overbought_ch

    # Check for duplicates
    exists = await bot.db.subscription_exists(
        guild_id=interaction.guild_id,
        ticker=ticker,
        condition=condition.value,
        threshold=threshold,
        period=target_period
    )

    if exists:
        await interaction.followup.send(
            f"❌ A subscription with these exact parameters already exists",
            ephemeral=True
        )
        return

    # Create subscription
    try:
        sub = await bot.db.create_subscription(
            guild_id=interaction.guild_id,
            ticker=ticker,
            condition=condition.value,
            threshold=threshold,
            period=target_period,
            cooldown_hours=target_cooldown,
            created_by_user_id=interaction.user.id
        )

        instrument = bot.catalog.get_instrument(ticker)
        name = instrument.name if instrument else ticker

        await interaction.followup.send(
            f"✅ **Subscription created** (ID: `{sub.id}`)\n"
            f"• **Ticker:** {ticker} — {name}\n"
            f"• **Condition:** RSI{target_period} {condition.value} {threshold}\n"
            f"• **Alerts to:** {target_channel.mention}\n"
            f"• **Cooldown:** {target_cooldown} hours",
            ephemeral=True
        )

    except Exception as e:
        logger.error(f"Error creating subscription: {e}")
        await interaction.followup.send(
            f"❌ Failed to create subscription: {str(e)}",
            ephemeral=True
        )


@bot.tree.command(
    name="subscribe-bands",
    description="Create both oversold and overbought alerts for a ticker"
)
@app_commands.describe(
    ticker="Stock ticker symbol (must exist in tickers.csv)",
    oversold="Oversold threshold (default: 30)",
    overbought="Overbought threshold (default: 70)",
    period="RSI period (default: server default or 14)",
    cooldown="Hours between alerts (default: server default or 24)"
)
async def subscribe_bands(
    interaction: discord.Interaction,
    ticker: str,
    oversold: Optional[float] = None,
    overbought: Optional[float] = None,
    period: Optional[int] = None,
    cooldown: Optional[int] = None
):
    """Create both oversold (UNDER) and overbought (OVER) subscriptions."""
    await interaction.response.defer(ephemeral=True)

    # Validate ticker
    is_valid, error = validate_ticker(ticker)
    if not is_valid:
        await interaction.followup.send(f"❌ {error}", ephemeral=True)
        return

    # Check that alert channels exist
    oversold_ch, overbought_ch, error_msg = get_alert_channels(interaction.guild)
    if error_msg:
        await interaction.followup.send(error_msg, ephemeral=True)
        return

    # Apply defaults for thresholds
    oversold_threshold = oversold if oversold is not None else DEFAULT_OVERSOLD_THRESHOLD
    overbought_threshold = overbought if overbought is not None else DEFAULT_OVERBOUGHT_THRESHOLD

    # Validate thresholds
    if not 0 <= oversold_threshold <= 100:
        await interaction.followup.send(
            "❌ Oversold threshold must be between 0 and 100",
            ephemeral=True
        )
        return

    if not 0 <= overbought_threshold <= 100:
        await interaction.followup.send(
            "❌ Overbought threshold must be between 0 and 100",
            ephemeral=True
        )
        return

    if oversold_threshold >= overbought_threshold:
        await interaction.followup.send(
            "❌ Oversold threshold must be less than overbought threshold",
            ephemeral=True
        )
        return

    # Validate period
    if period is not None and not 2 <= period <= 200:
        await interaction.followup.send(
            "❌ Period must be between 2 and 200",
            ephemeral=True
        )
        return

    # Get guild config for defaults
    config = await bot.db.get_or_create_guild_config(interaction.guild_id)

    # Apply defaults
    target_period = period if period is not None else config.default_rsi_period
    target_cooldown = cooldown if cooldown is not None else config.default_cooldown_hours

    ticker = ticker.upper().strip()
    instrument = bot.catalog.get_instrument(ticker)
    name = instrument.name if instrument else ticker

    created_subs = []
    errors = []

    # Create UNDER (oversold) subscription
    try:
        exists = await bot.db.subscription_exists(
            guild_id=interaction.guild_id,
            ticker=ticker,
            condition="UNDER",
            threshold=oversold_threshold,
            period=target_period
        )

        if exists:
            errors.append(f"UNDER {oversold_threshold} already exists")
        else:
            sub = await bot.db.create_subscription(
                guild_id=interaction.guild_id,
                ticker=ticker,
                condition="UNDER",
                threshold=oversold_threshold,
                period=target_period,
                cooldown_hours=target_cooldown,
                created_by_user_id=interaction.user.id
            )
            created_subs.append(f"UNDER {oversold_threshold} (ID: `{sub.id}`) → {oversold_ch.mention}")
    except Exception as e:
        errors.append(f"UNDER: {str(e)}")

    # Create OVER (overbought) subscription
    try:
        exists = await bot.db.subscription_exists(
            guild_id=interaction.guild_id,
            ticker=ticker,
            condition="OVER",
            threshold=overbought_threshold,
            period=target_period
        )

        if exists:
            errors.append(f"OVER {overbought_threshold} already exists")
        else:
            sub = await bot.db.create_subscription(
                guild_id=interaction.guild_id,
                ticker=ticker,
                condition="OVER",
                threshold=overbought_threshold,
                period=target_period,
                cooldown_hours=target_cooldown,
                created_by_user_id=interaction.user.id
            )
            created_subs.append(f"OVER {overbought_threshold} (ID: `{sub.id}`) → {overbought_ch.mention}")
    except Exception as e:
        errors.append(f"OVER: {str(e)}")

    # Build response
    response_lines = [f"**{ticker} — {name}**\n"]

    if created_subs:
        response_lines.append("✅ **Created:**")
        for sub_info in created_subs:
            response_lines.append(f"• RSI{target_period} {sub_info}")
        response_lines.append(f"• Cooldown: {target_cooldown} hours")

    if errors:
        response_lines.append("\n⚠️ **Warnings:**")
        for error in errors:
            response_lines.append(f"• {error}")

    await interaction.followup.send("\n".join(response_lines), ephemeral=True)


@bot.tree.command(name="unsubscribe", description="Remove an RSI alert subscription (your own only)")
@app_commands.describe(
    id="Subscription ID to remove (from /list)"
)
async def unsubscribe(
    interaction: discord.Interaction,
    id: int
):
    """Remove a subscription by ID. Users can only remove their own subscriptions."""
    await interaction.response.defer(ephemeral=True)

    # Verify subscription exists and belongs to this guild
    sub = await bot.db.get_subscription(id)

    if not sub:
        await interaction.followup.send(
            f"❌ Subscription ID `{id}` not found",
            ephemeral=True
        )
        return

    if sub.guild_id != interaction.guild_id:
        await interaction.followup.send(
            f"❌ Subscription ID `{id}` does not belong to this server",
            ephemeral=True
        )
        return

    # Check ownership - users can only delete their own subscriptions
    if sub.created_by_user_id != interaction.user.id:
        await interaction.followup.send(
            f"❌ **Permission Denied**\n"
            f"You can only remove subscriptions you created.\n"
            f"This subscription was created by <@{sub.created_by_user_id}>.\n\n"
            f"If you're an admin and need to remove this, use `/admin-unsubscribe`.",
            ephemeral=True
        )
        return

    # Delete subscription
    deleted = await bot.db.delete_subscription(id, interaction.guild_id)

    if deleted:
        instrument = bot.catalog.get_instrument(sub.ticker)
        name = instrument.name if instrument else sub.ticker

        await interaction.followup.send(
            f"✅ **Subscription removed** (ID: `{id}`)\n"
            f"• **Ticker:** {sub.ticker} — {name}\n"
            f"• **Condition:** RSI{sub.period} {sub.condition} {sub.threshold}",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"❌ Failed to remove subscription ID `{id}`",
            ephemeral=True
        )


@bot.tree.command(name="unsubscribe-all", description="Remove all your subscriptions")
async def unsubscribe_all(interaction: discord.Interaction):
    """Remove all subscriptions created by the user."""
    await interaction.response.defer(ephemeral=True)

    # Get user's subscriptions first for confirmation
    user_subs = await bot.db.get_user_subscriptions(
        interaction.guild_id, 
        interaction.user.id
    )

    if not user_subs:
        await interaction.followup.send(
            "📋 You have no subscriptions to remove.",
            ephemeral=True
        )
        return

    # Delete all user's subscriptions
    deleted_count = await bot.db.delete_user_subscriptions(
        interaction.guild_id,
        interaction.user.id
    )

    if deleted_count > 0:
        await interaction.followup.send(
            f"✅ **Removed {deleted_count} subscription(s)**\n\n"
            f"All your RSI alert subscriptions have been cleared.",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            "❌ Failed to remove subscriptions. Please try again.",
            ephemeral=True
        )


@bot.tree.command(name="admin-unsubscribe", description="[Admin] Remove any subscription by ID")
@app_commands.default_permissions(administrator=True)
@app_commands.describe(
    id="Subscription ID to remove",
    reason="Reason for removal (will be logged)"
)
async def admin_unsubscribe(
    interaction: discord.Interaction,
    id: int,
    reason: Optional[str] = None
):
    """Admin command to remove any subscription regardless of ownership."""
    await interaction.response.defer(ephemeral=True)

    # Check admin permission
    if not interaction.user.guild_permissions.administrator:
        await interaction.followup.send(
            "❌ **Permission Denied**\nThis command requires Administrator permission.",
            ephemeral=True
        )
        return

    # Verify subscription exists and belongs to this guild
    sub = await bot.db.get_subscription(id)

    if not sub:
        await interaction.followup.send(
            f"❌ Subscription ID `{id}` not found",
            ephemeral=True
        )
        return

    if sub.guild_id != interaction.guild_id:
        await interaction.followup.send(
            f"❌ Subscription ID `{id}` does not belong to this server",
            ephemeral=True
        )
        return

    # Get details for logging before deletion
    instrument = bot.catalog.get_instrument(sub.ticker)
    name = instrument.name if instrument else sub.ticker
    original_owner_id = sub.created_by_user_id

    # Delete subscription
    deleted = await bot.db.delete_subscription(id, interaction.guild_id)

    if deleted:
        # Log the admin action to #server-changelog
        log_details = (
            f"Removed subscription ID `{id}`\n"
            f"Ticker: {sub.ticker} ({name})\n"
            f"Condition: RSI{sub.period} {sub.condition} {sub.threshold}\n"
            f"Originally created by: <@{original_owner_id}>"
        )
        if reason:
            log_details += f"\nReason: {reason}"

        await log_admin_action(
            bot,
            "Unsubscribe Override",
            interaction.user.id,
            str(interaction.user),
            log_details
        )

        await interaction.followup.send(
            f"✅ **Subscription removed by admin** (ID: `{id}`)\n"
            f"• **Ticker:** {sub.ticker} — {name}\n"
            f"• **Condition:** RSI{sub.period} {sub.condition} {sub.threshold}\n"
            f"• **Originally created by:** <@{original_owner_id}>\n"
            f"• **Action logged to:** `#{CHANGELOG_CHANNEL_NAME}`",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"❌ Failed to remove subscription ID `{id}`",
            ephemeral=True
        )


@bot.tree.command(name="list", description="List RSI alert subscriptions")
@app_commands.describe(
    ticker="Filter by ticker (optional)"
)
async def list_subscriptions(
    interaction: discord.Interaction,
    ticker: Optional[str] = None
):
    """List all subscriptions for this server."""
    await interaction.response.defer(ephemeral=True)

    subs = await bot.db.get_subscriptions_by_guild(
        guild_id=interaction.guild_id,
        ticker=ticker.upper().strip() if ticker else None
    )

    if not subs:
        filter_text = f" for ticker `{ticker.upper()}`" if ticker else ""
        await interaction.followup.send(
            f"📋 No subscriptions found{filter_text}",
            ephemeral=True
        )
        return

    # Build response grouped by condition
    under_subs = [s for s in subs if s.condition == "UNDER"]
    over_subs = [s for s in subs if s.condition == "OVER"]

    lines = [f"📋 **Subscriptions** ({len(subs)} total)\n"]

    if under_subs:
        lines.append(f"**#{OVERSOLD_CHANNEL_NAME}** (UNDER/Oversold):")
        for sub in under_subs:
            instrument = bot.catalog.get_instrument(sub.ticker)
            name = instrument.name if instrument else sub.ticker
            lines.append(
                f"`{sub.id}` — **{sub.ticker}** ({name}) "
                f"| RSI{sub.period} < {sub.threshold}"
            )
        lines.append("")

    if over_subs:
        lines.append(f"**#{OVERBOUGHT_CHANNEL_NAME}** (OVER/Overbought):")
        for sub in over_subs:
            instrument = bot.catalog.get_instrument(sub.ticker)
            name = instrument.name if instrument else sub.ticker
            lines.append(
                f"`{sub.id}` — **{sub.ticker}** ({name}) "
                f"| RSI{sub.period} > {sub.threshold}"
            )
        lines.append("")

    # Handle Discord message length limit
    response = "\n".join(lines)
    if len(response) > 1900:
        response = response[:1900] + "\n...(truncated)"

    await interaction.followup.send(response, ephemeral=True)


@bot.tree.command(name="run-now", description="Manually trigger RSI check (Admin)")
@app_commands.default_permissions(manage_guild=True)
async def run_now(interaction: discord.Interaction):
    """Manually trigger RSI evaluation and post alerts to channels."""
    await interaction.response.defer(ephemeral=True)

    # Check that alert channels exist
    oversold_ch, overbought_ch, error_msg = get_alert_channels(interaction.guild)
    if error_msg:
        await interaction.followup.send(error_msg, ephemeral=True)
        return

    # Get subscriptions for this guild
    subs = await bot.db.get_subscriptions_by_guild(
        guild_id=interaction.guild_id,
        enabled_only=True
    )

    if not subs:
        await interaction.followup.send(
            "📋 No active subscriptions in this server",
            ephemeral=True
        )
        return

    # Determine unique tickers and periods needed
    ticker_periods = {}
    for sub in subs:
        if sub.ticker not in ticker_periods:
            ticker_periods[sub.ticker] = []
        if sub.period not in ticker_periods[sub.ticker]:
            ticker_periods[sub.ticker].append(sub.period)

    await interaction.followup.send(
        f"⏳ Fetching RSI data for {len(ticker_periods)} tickers...",
        ephemeral=True
    )

    # Calculate RSI
    rsi_results = await bot.rsi_calculator.calculate_rsi_for_tickers(ticker_periods)

    # Evaluate subscriptions (not dry run - update state)
    alerts_by_condition = await bot.alert_engine.evaluate_subscriptions(
        rsi_results, dry_run=False
    )

    # Report results
    successful = sum(1 for r in rsi_results.values() if r.success)
    failed = len(rsi_results) - successful
    under_count = len(alerts_by_condition.get('UNDER', []))
    over_count = len(alerts_by_condition.get('OVER', []))

    # Send alerts to the appropriate channels
    messages_sent = 0
    send_errors = []

    # Send UNDER alerts to oversold channel
    under_alerts = alerts_by_condition.get('UNDER', [])
    try:
        if under_alerts:
            messages = format_alert_list(under_alerts, 'UNDER')
            for msg in messages:
                await oversold_ch.send(msg)
                messages_sent += 1
        else:
            await oversold_ch.send(format_no_alerts_message('UNDER'))
            messages_sent += 1
    except discord.Forbidden:
        send_errors.append(f"Cannot send to {oversold_ch.mention} - missing permissions")
    except Exception as e:
        send_errors.append(f"Error sending to {oversold_ch.mention}: {str(e)}")

    # Send OVER alerts to overbought channel
    over_alerts = alerts_by_condition.get('OVER', [])
    try:
        if over_alerts:
            messages = format_alert_list(over_alerts, 'OVER')
            for msg in messages:
                await overbought_ch.send(msg)
                messages_sent += 1
        else:
            await overbought_ch.send(format_no_alerts_message('OVER'))
            messages_sent += 1
    except discord.Forbidden:
        send_errors.append(f"Cannot send to {overbought_ch.mention} - missing permissions")
    except Exception as e:
        send_errors.append(f"Error sending to {overbought_ch.mention}: {str(e)}")

    # Send summary to user
    summary = (
        f"✅ **RSI Check Complete**\n"
        f"• Tickers fetched: {successful} success, {failed} failed\n"
        f"• Subscriptions evaluated: {len(subs)}\n"
        f"• Alerts triggered: {under_count + over_count}\n"
        f"  - {oversold_ch.mention}: {under_count} oversold alerts\n"
        f"  - {overbought_ch.mention}: {over_count} overbought alerts\n"
        f"• Messages sent: {messages_sent}"
    )
    
    if send_errors:
        summary += "\n\n⚠️ **Errors:**\n" + "\n".join(f"• {e}" for e in send_errors)
        summary += "\n\n**Fix:** Go to channel settings → Permissions → Add bot role → Enable **Send Messages**"

    await interaction.edit_original_response(content=summary)


@bot.tree.command(name="set-defaults", description="Set server defaults (Admin)")
@app_commands.default_permissions(manage_guild=True)
@app_commands.describe(
    default_period="Default RSI period (2-200)",
    default_cooldown="Default cooldown hours",
    schedule_time="Daily run time in HH:MM format (Europe/Oslo)",
    alert_mode="Alert mode: CROSSING or LEVEL",
    hysteresis="Hysteresis value for crossing detection"
)
@app_commands.choices(alert_mode=[
    app_commands.Choice(name="CROSSING", value="CROSSING"),
    app_commands.Choice(name="LEVEL", value="LEVEL")
])
async def set_defaults(
    interaction: discord.Interaction,
    default_period: Optional[int] = None,
    default_cooldown: Optional[int] = None,
    schedule_time: Optional[str] = None,
    alert_mode: Optional[app_commands.Choice[str]] = None,
    hysteresis: Optional[float] = None
):
    """Set server-level default configuration."""
    await interaction.response.defer(ephemeral=True)

    # Validate inputs
    if default_period is not None and not 2 <= default_period <= 200:
        await interaction.followup.send(
            "❌ Period must be between 2 and 200",
            ephemeral=True
        )
        return

    if default_cooldown is not None and default_cooldown < 0:
        await interaction.followup.send(
            "❌ Cooldown must be non-negative",
            ephemeral=True
        )
        return

    if schedule_time is not None:
        try:
            parts = schedule_time.split(":")
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError()
        except (ValueError, IndexError):
            await interaction.followup.send(
                "❌ Schedule time must be in HH:MM format (e.g., 18:30)",
                ephemeral=True
            )
            return

    if hysteresis is not None and hysteresis < 0:
        await interaction.followup.send(
            "❌ Hysteresis must be non-negative",
            ephemeral=True
        )
        return

    # Update configuration
    config = await bot.db.update_guild_config(
        guild_id=interaction.guild_id,
        default_rsi_period=default_period,
        default_schedule_time=schedule_time,
        default_cooldown_hours=default_cooldown,
        alert_mode=alert_mode.value if alert_mode else None,
        hysteresis=hysteresis
    )

    await interaction.followup.send(
        f"✅ **Server defaults updated**\n"
        f"• **Default RSI period:** {config.default_rsi_period}\n"
        f"• **Default cooldown:** {config.default_cooldown_hours} hours\n"
        f"• **Schedule time:** {config.default_schedule_time} (Europe/Oslo)\n"
        f"• **Alert mode:** {config.alert_mode}\n"
        f"• **Hysteresis:** {config.hysteresis}\n\n"
        f"**Fixed alert channels:**\n"
        f"• Oversold (UNDER): `#{OVERSOLD_CHANNEL_NAME}`\n"
        f"• Overbought (OVER): `#{OVERBOUGHT_CHANNEL_NAME}`",
        ephemeral=True
    )


@bot.tree.command(name="ticker-info", description="Get information about a ticker")
@app_commands.describe(
    ticker="Stock ticker symbol to look up"
)
async def ticker_info(
    interaction: discord.Interaction,
    ticker: str
):
    """Get information about a ticker from the catalog, including subscriptions and RSI."""
    await interaction.response.defer(ephemeral=True)

    ticker = ticker.upper().strip()
    instrument = bot.catalog.get_instrument(ticker)

    if not instrument:
        # Try to search
        results = bot.catalog.search_tickers(ticker, limit=5)
        if results:
            suggestions = "\n".join(
                f"• `{i.ticker}` — {i.name}" for i in results
            )
            await interaction.followup.send(
                f"❌ Ticker `{ticker}` not found in catalog.\n\n"
                f"**Did you mean:**\n{suggestions}",
                ephemeral=True
            )
        else:
            await interaction.followup.send(
                f"❌ Ticker `{ticker}` not found in catalog.\n"
                f"Add it to `tickers.csv` to enable subscriptions.",
                ephemeral=True
            )
        return

    # Build response
    lines = [
        f"**{instrument.ticker} — {instrument.name}**",
        f"🔗 [Nordnet]({instrument.nordnet_url})",
        ""
    ]

    # Get subscriptions for this ticker in this guild
    subs = await bot.db.get_subscriptions_by_guild(
        guild_id=interaction.guild_id,
        ticker=ticker
    )

    # Get RSI data from subscription states if available
    rsi_data = None
    rsi_stale = False
    if subs:
        # Get the most recent state from any subscription
        for sub in subs:
            state = await bot.db.get_subscription_state(sub.id)
            if state and state.last_rsi is not None and state.last_date:
                # Check if data is fresh (within 24 hours based on last_date)
                from datetime import datetime, timedelta
                try:
                    last_date = datetime.strptime(state.last_date, "%Y-%m-%d")
                    days_old = (datetime.now() - last_date).days
                    if rsi_data is None or state.last_date > rsi_data['date']:
                        rsi_data = {
                            'rsi': state.last_rsi,
                            'close': state.last_close,
                            'date': state.last_date,
                            'period': sub.period,
                            'days_old': days_old
                        }
                        rsi_stale = days_old > 1  # More than 1 trading day old
                except ValueError:
                    pass

    # Show RSI data if available
    if rsi_data:
        if rsi_stale:
            lines.append(f"⚠️ **RSI Data (STALE - {rsi_data['days_old']} days old):**")
        else:
            lines.append("📊 **RSI Data:**")
        lines.append(f"• RSI{rsi_data['period']}: **{rsi_data['rsi']:.1f}**")
        lines.append(f"• Last Close: {rsi_data['close']:.2f} ({rsi_data['date']})")
        lines.append("")
    else:
        lines.append("📊 **RSI Data:** Not yet checked")
        lines.append("")

    # Show subscriptions
    if subs:
        under_subs = [s for s in subs if s.condition == "UNDER"]
        over_subs = [s for s in subs if s.condition == "OVER"]

        lines.append(f"🔔 **Active Subscriptions:** ({len(subs)} total)")
        
        if under_subs:
            for sub in under_subs:
                lines.append(f"• `{sub.id}` — RSI{sub.period} < {sub.threshold} → #{OVERSOLD_CHANNEL_NAME}")
        
        if over_subs:
            for sub in over_subs:
                lines.append(f"• `{sub.id}` — RSI{sub.period} > {sub.threshold} → #{OVERBOUGHT_CHANNEL_NAME}")
    else:
        lines.append("🔔 **Active Subscriptions:** None")
        lines.append("Use `/subscribe` or `/subscribe-bands` to add alerts for this ticker.")

    await interaction.followup.send("\n".join(lines), ephemeral=True)


@bot.tree.command(name="catalog-stats", description="Show ticker catalog and subscription statistics")
async def catalog_stats(interaction: discord.Interaction):
    """Show statistics about the ticker catalog and subscriptions."""
    await interaction.response.defer(ephemeral=True)

    catalog_count = len(bot.catalog)
    
    # Get subscription counts for this guild
    all_subs = await bot.db.get_subscriptions_by_guild(
        guild_id=interaction.guild_id,
        enabled_only=False
    )
    
    total_subs = len(all_subs)
    enabled_subs = sum(1 for s in all_subs if s.enabled)
    under_subs = sum(1 for s in all_subs if s.condition == "UNDER" and s.enabled)
    over_subs = sum(1 for s in all_subs if s.condition == "OVER" and s.enabled)
    unique_tickers = len(set(s.ticker for s in all_subs if s.enabled))

    await interaction.followup.send(
        f"📊 **Bot Statistics**\n\n"
        f"**Ticker Catalog:**\n"
        f"• Total instruments: {catalog_count}\n"
        f"• File: `tickers.csv`\n\n"
        f"**Subscriptions (this server):**\n"
        f"• Total active: **{enabled_subs}**\n"
        f"• Oversold alerts (UNDER): {under_subs}\n"
        f"• Overbought alerts (OVER): {over_subs}\n"
        f"• Unique tickers watched: {unique_tickers}\n\n"
        f"**Alert Channels:**\n"
        f"• `#{OVERSOLD_CHANNEL_NAME}` — UNDER alerts\n"
        f"• `#{OVERBOUGHT_CHANNEL_NAME}` — OVER alerts",
        ephemeral=True
    )


# ==================== Server Control Commands ====================

@bot.tree.command(name="server-status", description="Check server status and scheduled actions")
async def server_status(interaction: discord.Interaction):
    """Get current server status."""
    await interaction.response.defer(ephemeral=True)

    if not bot.server_monitor:
        await interaction.followup.send(
            "❌ Server monitoring is not enabled.",
            ephemeral=True
        )
        return

    status = bot.server_monitor.get_status()
    
    # Format status
    status_emoji = {
        "online": "🟢",
        "offline": "🔴",
        "unknown": "⚪"
    }
    
    lines = [
        f"{status_emoji.get(status['status'], '⚪')} **Server Status: {status['status'].upper()}**",
        ""
    ]
    
    if status['last_check']:
        lines.append(f"• Last check: {status['last_check']}")
    
    if status['last_change']:
        lines.append(f"• Last status change: {status['last_change']}")
    
    if status['scheduled_action']:
        action = status['scheduled_action']
        lines.append("")
        lines.append(f"⏰ **Scheduled {action['type'].capitalize()}:**")
        lines.append(f"• Time: {action['time']}")
        lines.append(f"• Scheduled by: {action['by']}")
        if action['reason']:
            lines.append(f"• Reason: {action['reason']}")

    await interaction.followup.send("\n".join(lines), ephemeral=True)


@bot.tree.command(name="schedule-restart", description="[Admin] Schedule a server restart")
@app_commands.default_permissions(administrator=True)
@app_commands.describe(
    minutes="Minutes from now to restart (use this OR time, not both)",
    time="Specific time to restart (HH:MM format, Europe/Oslo timezone)",
    reason="Reason for restart (optional)"
)
async def schedule_restart(
    interaction: discord.Interaction,
    minutes: Optional[int] = None,
    time: Optional[str] = None,
    reason: Optional[str] = None
):
    """Schedule a server restart."""
    await interaction.response.defer(ephemeral=True)

    # Check admin permission
    if not interaction.user.guild_permissions.administrator:
        await interaction.followup.send(
            "❌ **Permission Denied**\nThis command requires Administrator permission.",
            ephemeral=True
        )
        return

    if not bot.server_monitor:
        await interaction.followup.send(
            "❌ Server monitoring is not enabled.",
            ephemeral=True
        )
        return

    # Validate input - must have exactly one of minutes or time
    if (minutes is None) == (time is None):
        await interaction.followup.send(
            "❌ Please provide either `minutes` OR `time`, not both or neither.",
            ephemeral=True
        )
        return

    # Calculate scheduled time
    tz = pytz.timezone(DEFAULT_TIMEZONE)
    now = datetime.now(tz)
    
    if minutes is not None:
        if minutes <= 0:
            await interaction.followup.send(
                "❌ Minutes must be a positive number.",
                ephemeral=True
            )
            return
        scheduled_time = now + timedelta(minutes=minutes)
    else:
        try:
            # Parse HH:MM format
            hour, minute = map(int, time.split(":"))
            scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # If time is in the past today, schedule for tomorrow
            if scheduled_time <= now:
                scheduled_time += timedelta(days=1)
        except (ValueError, AttributeError):
            await interaction.followup.send(
                "❌ Invalid time format. Use HH:MM (e.g., 18:30).",
                ephemeral=True
            )
            return

    # Schedule the restart
    success, message = await bot.server_monitor.schedule_action(
        action_type=ActionType.RESTART,
        scheduled_time=scheduled_time,
        user_id=interaction.user.id,
        user_name=str(interaction.user),
        reason=reason
    )

    await interaction.followup.send(message, ephemeral=True)


@bot.tree.command(name="schedule-shutdown", description="[Admin] Schedule a server shutdown")
@app_commands.default_permissions(administrator=True)
@app_commands.describe(
    minutes="Minutes from now to shutdown (use this OR time, not both)",
    time="Specific time to shutdown (HH:MM format, Europe/Oslo timezone)",
    reason="Reason for shutdown (optional)"
)
async def schedule_shutdown(
    interaction: discord.Interaction,
    minutes: Optional[int] = None,
    time: Optional[str] = None,
    reason: Optional[str] = None
):
    """Schedule a server shutdown."""
    await interaction.response.defer(ephemeral=True)

    # Check admin permission
    if not interaction.user.guild_permissions.administrator:
        await interaction.followup.send(
            "❌ **Permission Denied**\nThis command requires Administrator permission.",
            ephemeral=True
        )
        return

    if not bot.server_monitor:
        await interaction.followup.send(
            "❌ Server monitoring is not enabled.",
            ephemeral=True
        )
        return

    # Validate input - must have exactly one of minutes or time
    if (minutes is None) == (time is None):
        await interaction.followup.send(
            "❌ Please provide either `minutes` OR `time`, not both or neither.",
            ephemeral=True
        )
        return

    # Calculate scheduled time
    tz = pytz.timezone(DEFAULT_TIMEZONE)
    now = datetime.now(tz)
    
    if minutes is not None:
        if minutes <= 0:
            await interaction.followup.send(
                "❌ Minutes must be a positive number.",
                ephemeral=True
            )
            return
        scheduled_time = now + timedelta(minutes=minutes)
    else:
        try:
            # Parse HH:MM format
            hour, minute = map(int, time.split(":"))
            scheduled_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # If time is in the past today, schedule for tomorrow
            if scheduled_time <= now:
                scheduled_time += timedelta(days=1)
        except (ValueError, AttributeError):
            await interaction.followup.send(
                "❌ Invalid time format. Use HH:MM (e.g., 18:30).",
                ephemeral=True
            )
            return

    # Schedule the shutdown
    success, message = await bot.server_monitor.schedule_action(
        action_type=ActionType.SHUTDOWN,
        scheduled_time=scheduled_time,
        user_id=interaction.user.id,
        user_name=str(interaction.user),
        reason=reason
    )

    await interaction.followup.send(message, ephemeral=True)


@bot.tree.command(name="cancel-scheduled-action", description="[Admin] Cancel a scheduled restart/shutdown")
@app_commands.default_permissions(administrator=True)
async def cancel_scheduled_action(interaction: discord.Interaction):
    """Cancel a pending restart/shutdown."""
    await interaction.response.defer(ephemeral=True)

    # Check admin permission
    if not interaction.user.guild_permissions.administrator:
        await interaction.followup.send(
            "❌ **Permission Denied**\nThis command requires Administrator permission.",
            ephemeral=True
        )
        return

    if not bot.server_monitor:
        await interaction.followup.send(
            "❌ Server monitoring is not enabled.",
            ephemeral=True
        )
        return

    success, message = await bot.server_monitor.cancel_action(
        user_id=interaction.user.id,
        user_name=str(interaction.user)
    )

    await interaction.followup.send(message, ephemeral=True)


@bot.tree.command(name="reload-catalog", description="[Admin] Reload the ticker catalog from tickers.csv")
@app_commands.default_permissions(administrator=True)
async def reload_catalog(interaction: discord.Interaction):
    """Reload the ticker catalog."""
    await interaction.response.defer(ephemeral=True)

    # Check admin permission
    if not interaction.user.guild_permissions.administrator:
        await interaction.followup.send(
            "❌ **Permission Denied**\nThis command requires Administrator permission.",
            ephemeral=True
        )
        return

    old_count = len(bot.catalog)
    success = bot.catalog.reload()
    new_count = len(bot.catalog)

    if success:
        await interaction.followup.send(
            f"✅ **Ticker catalog reloaded**\n"
            f"• Previous: {old_count} instruments\n"
            f"• Current: {new_count} instruments\n"
            f"• Change: {new_count - old_count:+d}",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"❌ **Failed to reload catalog**\nCheck the logs for details.",
            ephemeral=True
        )


# ==================== Autocomplete ====================

@subscribe.autocomplete('ticker')
@subscribe_bands.autocomplete('ticker')
@ticker_info.autocomplete('ticker')
@list_subscriptions.autocomplete('ticker')
async def ticker_autocomplete(
    interaction: discord.Interaction,
    current: str
):
    """Autocomplete ticker symbols."""
    if not current:
        return []

    results = bot.catalog.search_tickers(current, limit=25)
    return [
        app_commands.Choice(name=f"{i.ticker} — {i.name[:40]}", value=i.ticker)
        for i in results
    ]


# ==================== Main ====================

def main():
    """Run the bot."""
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN environment variable not set")
        print("Error: Please set the DISCORD_TOKEN environment variable")
        print("  export DISCORD_TOKEN=your_bot_token")
        print("  python main.py")
        sys.exit(1)

    logger.info("Starting RSI Discord Bot...")
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
