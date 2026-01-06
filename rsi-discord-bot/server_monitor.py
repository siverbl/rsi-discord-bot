"""
Server Monitor for RSI Discord Bot.
Handles server health monitoring, status announcements, and scheduled restarts/shutdowns.
"""
import asyncio
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
import discord
import pytz

from config import (
    SERVER_HEALTH_CHECK_URL,
    SERVER_HEALTH_CHECK_INTERVAL,
    SERVER_HEALTH_CHECK_TIMEOUT,
    SERVER_RESTART_SCRIPT,
    SERVER_SHUTDOWN_SCRIPT,
    CHANGELOG_CHANNEL_NAME,
    DEFAULT_TIMEZONE
)

logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class ActionType(Enum):
    RESTART = "restart"
    SHUTDOWN = "shutdown"


@dataclass
class ScheduledAction:
    """Represents a scheduled server action."""
    action_type: ActionType
    scheduled_time: datetime
    scheduled_by: int  # User ID
    scheduled_by_name: str
    reason: Optional[str]
    warning_sent_10min: bool = False
    warning_sent_1min: bool = False
    cancelled: bool = False


class ServerMonitor:
    """
    Monitors server health and handles scheduled actions.
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.current_status = ServerStatus.UNKNOWN
        self.last_check_time: Optional[datetime] = None
        self.last_status_change: Optional[datetime] = None
        self.scheduled_action: Optional[ScheduledAction] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._action_task: Optional[asyncio.Task] = None
        self.timezone = pytz.timezone(DEFAULT_TIMEZONE)
    
    async def start(self):
        """Start the server monitor."""
        if SERVER_HEALTH_CHECK_URL:
            logger.info(f"Starting server monitor (checking {SERVER_HEALTH_CHECK_URL})")
            self._monitor_task = asyncio.create_task(self._monitor_loop())
        else:
            logger.info("Server health check URL not configured - monitoring disabled")
    
    async def stop(self):
        """Stop the server monitor."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._action_task:
            self._action_task.cancel()
            try:
                await self._action_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await self._check_health()
                await self._check_scheduled_action()
                await asyncio.sleep(SERVER_HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(SERVER_HEALTH_CHECK_INTERVAL)
    
    async def _check_health(self):
        """Check server health and announce status changes."""
        if not SERVER_HEALTH_CHECK_URL:
            return
        
        new_status = await self._perform_health_check()
        self.last_check_time = datetime.now(self.timezone)
        
        # Detect status change
        if self.current_status != new_status and self.current_status != ServerStatus.UNKNOWN:
            old_status = self.current_status
            self.current_status = new_status
            self.last_status_change = self.last_check_time
            
            # Announce status change
            await self._announce_status_change(old_status, new_status)
        else:
            self.current_status = new_status
    
    async def _perform_health_check(self) -> ServerStatus:
        """Perform the actual health check."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    SERVER_HEALTH_CHECK_URL,
                    timeout=aiohttp.ClientTimeout(total=SERVER_HEALTH_CHECK_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        return ServerStatus.ONLINE
                    else:
                        logger.warning(f"Health check returned status {response.status}")
                        return ServerStatus.OFFLINE
        except asyncio.TimeoutError:
            logger.warning("Health check timed out")
            return ServerStatus.OFFLINE
        except aiohttp.ClientError as e:
            logger.warning(f"Health check failed: {e}")
            return ServerStatus.OFFLINE
        except Exception as e:
            logger.error(f"Unexpected health check error: {e}")
            return ServerStatus.OFFLINE
    
    async def _announce_status_change(self, old_status: ServerStatus, new_status: ServerStatus):
        """Announce a status change to #server-changelog."""
        channel = await self._get_changelog_channel()
        if not channel:
            return
        
        timestamp = self.last_status_change.strftime("%Y-%m-%d %H:%M:%S %Z")
        
        if new_status == ServerStatus.ONLINE:
            message = f"🟢 **Server Online**\nThe server is now online.\n*{timestamp}*"
        else:
            message = f"🔴 **Server Offline**\nThe server appears to be offline.\n*{timestamp}*"
        
        try:
            await channel.send(message)
            logger.info(f"Announced status change: {old_status.value} -> {new_status.value}")
        except discord.HTTPException as e:
            logger.error(f"Failed to announce status change: {e}")
    
    async def _get_changelog_channel(self) -> Optional[discord.TextChannel]:
        """Get the #server-changelog channel from any guild."""
        for guild in self.bot.guilds:
            channel = discord.utils.get(guild.text_channels, name=CHANGELOG_CHANNEL_NAME)
            if channel:
                return channel
        return None
    
    async def _check_scheduled_action(self):
        """Check if a scheduled action should be executed or warned about."""
        if not self.scheduled_action or self.scheduled_action.cancelled:
            return
        
        now = datetime.now(self.timezone)
        action = self.scheduled_action
        time_until = (action.scheduled_time - now).total_seconds()
        
        # 10-minute warning
        if not action.warning_sent_10min and 540 <= time_until <= 660:
            await self._send_warning(action, "10 minutes")
            action.warning_sent_10min = True
        
        # 1-minute warning
        if not action.warning_sent_1min and 30 <= time_until <= 90:
            await self._send_warning(action, "1 minute")
            action.warning_sent_1min = True
        
        # Execute action
        if time_until <= 0:
            await self._execute_scheduled_action(action)
            self.scheduled_action = None
    
    async def _send_warning(self, action: ScheduledAction, time_left: str):
        """Send a warning about upcoming action."""
        channel = await self._get_changelog_channel()
        if not channel:
            return
        
        action_name = action.action_type.value.capitalize()
        message = f"⚠️ **{action_name} Warning**\nServer {action.action_type.value} in **{time_left}**."
        if action.reason:
            message += f"\nReason: {action.reason}"
        
        try:
            await channel.send(message)
        except discord.HTTPException as e:
            logger.error(f"Failed to send warning: {e}")
    
    async def _execute_scheduled_action(self, action: ScheduledAction):
        """Execute a scheduled restart/shutdown."""
        channel = await self._get_changelog_channel()
        
        # Announce execution
        action_name = action.action_type.value.capitalize()
        timestamp = datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
        
        if channel:
            message = f"🔄 **Executing {action_name}**\nScheduled by {action.scheduled_by_name}\n*{timestamp}*"
            try:
                await channel.send(message)
            except discord.HTTPException:
                pass
        
        # Execute the action
        success, result = await self._run_server_command(action.action_type)
        
        if channel:
            if success:
                message = f"✅ **{action_name} command executed successfully**"
            else:
                message = f"❌ **{action_name} command failed**\n{result}"
            
            try:
                await channel.send(message)
            except discord.HTTPException:
                pass
    
    async def _run_server_command(self, action_type: ActionType) -> tuple[bool, str]:
        """Run the server control script."""
        if action_type == ActionType.RESTART:
            script = SERVER_RESTART_SCRIPT
        else:
            script = SERVER_SHUTDOWN_SCRIPT
        
        if not script:
            return False, f"No {action_type.value} script configured"
        
        try:
            # Run the script asynchronously
            process = await asyncio.create_subprocess_exec(
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30
            )
            
            if process.returncode == 0:
                return True, stdout.decode() if stdout else "Success"
            else:
                return False, stderr.decode() if stderr else f"Exit code: {process.returncode}"
                
        except FileNotFoundError:
            return False, f"Script not found: {script}"
        except asyncio.TimeoutError:
            return False, "Script execution timed out"
        except Exception as e:
            return False, str(e)
    
    def get_status(self) -> dict:
        """Get current server status information."""
        return {
            "status": self.current_status.value,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "last_change": self.last_status_change.isoformat() if self.last_status_change else None,
            "scheduled_action": {
                "type": self.scheduled_action.action_type.value,
                "time": self.scheduled_action.scheduled_time.isoformat(),
                "by": self.scheduled_action.scheduled_by_name,
                "reason": self.scheduled_action.reason
            } if self.scheduled_action and not self.scheduled_action.cancelled else None
        }
    
    async def schedule_action(
        self,
        action_type: ActionType,
        scheduled_time: datetime,
        user_id: int,
        user_name: str,
        reason: Optional[str] = None
    ) -> tuple[bool, str]:
        """Schedule a restart or shutdown."""
        # Check if there's already a scheduled action
        if self.scheduled_action and not self.scheduled_action.cancelled:
            return False, f"There is already a scheduled {self.scheduled_action.action_type.value}. Cancel it first."
        
        # Validate time is in the future
        now = datetime.now(self.timezone)
        if scheduled_time <= now:
            return False, "Scheduled time must be in the future"
        
        # Check script is configured
        if action_type == ActionType.RESTART and not SERVER_RESTART_SCRIPT:
            return False, "Restart script not configured (set SERVER_RESTART_SCRIPT env var)"
        if action_type == ActionType.SHUTDOWN and not SERVER_SHUTDOWN_SCRIPT:
            return False, "Shutdown script not configured (set SERVER_SHUTDOWN_SCRIPT env var)"
        
        # Create scheduled action
        self.scheduled_action = ScheduledAction(
            action_type=action_type,
            scheduled_time=scheduled_time,
            scheduled_by=user_id,
            scheduled_by_name=user_name,
            reason=reason
        )
        
        # Announce scheduling
        channel = await self._get_changelog_channel()
        if channel:
            timestamp = scheduled_time.strftime("%Y-%m-%d %H:%M:%S %Z")
            message = (
                f"📅 **{action_type.value.capitalize()} Scheduled**\n"
                f"Scheduled by: {user_name}\n"
                f"Time: {timestamp}\n"
            )
            if reason:
                message += f"Reason: {reason}"
            
            try:
                await channel.send(message)
            except discord.HTTPException:
                pass
        
        logger.info(f"{action_type.value.capitalize()} scheduled by {user_name} for {scheduled_time}")
        return True, f"✅ {action_type.value.capitalize()} scheduled for {scheduled_time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    async def cancel_action(self, user_id: int, user_name: str) -> tuple[bool, str]:
        """Cancel a scheduled action."""
        if not self.scheduled_action or self.scheduled_action.cancelled:
            return False, "No scheduled action to cancel"
        
        action = self.scheduled_action
        action.cancelled = True
        
        # Announce cancellation
        channel = await self._get_changelog_channel()
        if channel:
            message = (
                f"🚫 **{action.action_type.value.capitalize()} Cancelled**\n"
                f"Cancelled by: {user_name}\n"
                f"Originally scheduled by: {action.scheduled_by_name}"
            )
            try:
                await channel.send(message)
            except discord.HTTPException:
                pass
        
        self.scheduled_action = None
        logger.info(f"Scheduled {action.action_type.value} cancelled by {user_name}")
        return True, f"✅ Scheduled {action.action_type.value} has been cancelled"


async def log_admin_action(
    bot,
    action: str,
    admin_id: int,
    admin_name: str,
    details: str
):
    """Log an admin action to #server-changelog."""
    for guild in bot.guilds:
        channel = discord.utils.get(guild.text_channels, name=CHANGELOG_CHANNEL_NAME)
        if channel:
            timestamp = datetime.now(pytz.timezone(DEFAULT_TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S %Z")
            message = (
                f"🔧 **Admin Action: {action}**\n"
                f"By: {admin_name} (ID: {admin_id})\n"
                f"Details: {details}\n"
                f"*{timestamp}*"
            )
            try:
                await channel.send(message)
            except discord.HTTPException as e:
                logger.error(f"Failed to log admin action: {e}")
            break
