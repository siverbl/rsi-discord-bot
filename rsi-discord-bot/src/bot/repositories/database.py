"""
Database module for RSI Discord Bot.
Uses SQLite for persistent storage of subscriptions and state.
"""
import aiosqlite
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from bot.config import DEFAULT_RSI_PERIOD, DEFAULT_COOLDOWN_HOURS, DEFAULT_SCHEDULE_TIME, DEFAULT_ALERT_MODE, DEFAULT_HYSTERESIS, DB_PATH


class Condition(Enum):
    UNDER = "UNDER"
    OVER = "OVER"


class AlertMode(Enum):
    CROSSING = "CROSSING"
    LEVEL = "LEVEL"


class Status(Enum):
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class GuildConfig:
    guild_id: int
    default_channel_id: Optional[int]
    default_rsi_period: int
    default_schedule_time: str
    default_cooldown_hours: int
    alert_mode: str
    hysteresis: float


@dataclass
class Subscription:
    id: int
    guild_id: int
    channel_id: Optional[int]
    ticker: str
    condition: str
    threshold: float
    period: int
    cooldown_hours: int
    enabled: bool
    created_by_user_id: Optional[int]
    created_at: datetime
    updated_at: datetime


@dataclass
class SubscriptionState:
    subscription_id: int
    last_rsi: Optional[float]
    last_close: Optional[float]
    last_date: Optional[str]
    last_status: str
    last_alert_at: Optional[datetime]
    days_in_zone: int  # consecutive days under/over threshold


class Database:
    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Create database tables if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:
            # Guild configuration table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS guild_config (
                    guild_id INTEGER PRIMARY KEY,
                    default_channel_id INTEGER,
                    default_rsi_period INTEGER DEFAULT 14,
                    default_schedule_time TEXT DEFAULT '18:30',
                    default_cooldown_hours INTEGER DEFAULT 24,
                    alert_mode TEXT DEFAULT 'CROSSING',
                    hysteresis REAL DEFAULT 2.0
                )
            """)

            # Subscriptions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER,
                    ticker TEXT NOT NULL,
                    condition TEXT NOT NULL CHECK (condition IN ('UNDER', 'OVER')),
                    threshold REAL NOT NULL,
                    period INTEGER NOT NULL DEFAULT 14,
                    cooldown_hours INTEGER NOT NULL DEFAULT 24,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_by_user_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Subscription state table for anti-spam and crossing detection
            await db.execute("""
                CREATE TABLE IF NOT EXISTS subscription_state (
                    subscription_id INTEGER PRIMARY KEY,
                    last_rsi REAL,
                    last_close REAL,
                    last_date TEXT,
                    last_status TEXT DEFAULT 'UNKNOWN',
                    last_alert_at TEXT,
                    days_in_zone INTEGER DEFAULT 0,
                    FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE
                )
            """)

            # Create indexes for faster queries
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_subscriptions_guild 
                ON subscriptions(guild_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_subscriptions_ticker 
                ON subscriptions(ticker)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_subscriptions_enabled 
                ON subscriptions(enabled)
            """)

            await db.commit()

    # ==================== Guild Config Operations ====================

    async def get_guild_config(self, guild_id: int) -> Optional[GuildConfig]:
        """Get guild configuration, returns None if not configured."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM guild_config WHERE guild_id = ?",
                (guild_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return GuildConfig(
                        guild_id=row['guild_id'],
                        default_channel_id=row['default_channel_id'],
                        default_rsi_period=row['default_rsi_period'],
                        default_schedule_time=row['default_schedule_time'],
                        default_cooldown_hours=row['default_cooldown_hours'],
                        alert_mode=row['alert_mode'],
                        hysteresis=row['hysteresis']
                    )
                return None

    async def get_or_create_guild_config(self, guild_id: int) -> GuildConfig:
        """Get guild config, creating default if it doesn't exist."""
        config = await self.get_guild_config(guild_id)
        if config:
            return config

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO guild_config (guild_id, default_rsi_period, 
                   default_schedule_time, default_cooldown_hours, alert_mode, hysteresis)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (guild_id, DEFAULT_RSI_PERIOD, DEFAULT_SCHEDULE_TIME,
                 DEFAULT_COOLDOWN_HOURS, DEFAULT_ALERT_MODE, DEFAULT_HYSTERESIS)
            )
            await db.commit()

        return GuildConfig(
            guild_id=guild_id,
            default_channel_id=None,
            default_rsi_period=DEFAULT_RSI_PERIOD,
            default_schedule_time=DEFAULT_SCHEDULE_TIME,
            default_cooldown_hours=DEFAULT_COOLDOWN_HOURS,
            alert_mode=DEFAULT_ALERT_MODE,
            hysteresis=DEFAULT_HYSTERESIS
        )

    async def update_guild_config(
        self,
        guild_id: int,
        default_channel_id: Optional[int] = None,
        default_rsi_period: Optional[int] = None,
        default_schedule_time: Optional[str] = None,
        default_cooldown_hours: Optional[int] = None,
        alert_mode: Optional[str] = None,
        hysteresis: Optional[float] = None
    ) -> GuildConfig:
        """Update guild configuration with provided values."""
        # Ensure config exists
        await self.get_or_create_guild_config(guild_id)

        updates = []
        params = []

        if default_channel_id is not None:
            updates.append("default_channel_id = ?")
            params.append(default_channel_id)
        if default_rsi_period is not None:
            updates.append("default_rsi_period = ?")
            params.append(default_rsi_period)
        if default_schedule_time is not None:
            updates.append("default_schedule_time = ?")
            params.append(default_schedule_time)
        if default_cooldown_hours is not None:
            updates.append("default_cooldown_hours = ?")
            params.append(default_cooldown_hours)
        if alert_mode is not None:
            updates.append("alert_mode = ?")
            params.append(alert_mode)
        if hysteresis is not None:
            updates.append("hysteresis = ?")
            params.append(hysteresis)

        if updates:
            params.append(guild_id)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"UPDATE guild_config SET {', '.join(updates)} WHERE guild_id = ?",
                    params
                )
                await db.commit()

        return await self.get_guild_config(guild_id)

    # ==================== Subscription Operations ====================

    async def create_subscription(
        self,
        guild_id: int,
        ticker: str,
        condition: str,
        threshold: float,
        period: int,
        cooldown_hours: int,
        created_by_user_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        enabled: bool = True
    ) -> Subscription:
        """Create a new subscription."""
        now = datetime.utcnow().isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """INSERT INTO subscriptions 
                   (guild_id, channel_id, ticker, condition, threshold, period, 
                    cooldown_hours, enabled, created_by_user_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (guild_id, channel_id, ticker.upper(), condition.upper(),
                 threshold, period, cooldown_hours, int(enabled), created_by_user_id, now, now)
            )
            subscription_id = cursor.lastrowid

            # Create initial state record
            await db.execute(
                """INSERT INTO subscription_state 
                   (subscription_id, last_status, days_in_zone)
                   VALUES (?, 'UNKNOWN', 0)""",
                (subscription_id,)
            )

            await db.commit()

            return Subscription(
                id=subscription_id,
                guild_id=guild_id,
                channel_id=channel_id,
                ticker=ticker.upper(),
                condition=condition.upper(),
                threshold=threshold,
                period=period,
                cooldown_hours=cooldown_hours,
                enabled=enabled,
                created_by_user_id=created_by_user_id,
                created_at=datetime.fromisoformat(now),
                updated_at=datetime.fromisoformat(now)
            )

    async def get_subscription(self, subscription_id: int) -> Optional[Subscription]:
        """Get a subscription by ID."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM subscriptions WHERE id = ?",
                (subscription_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_subscription(row)
                return None

    async def get_subscriptions_by_guild(
        self,
        guild_id: int,
        channel_id: Optional[int] = None,
        ticker: Optional[str] = None,
        enabled_only: bool = False
    ) -> List[Subscription]:
        """Get subscriptions for a guild with optional filters."""
        query = "SELECT * FROM subscriptions WHERE guild_id = ?"
        params = [guild_id]

        if channel_id is not None:
            query += " AND channel_id = ?"
            params.append(channel_id)
        if ticker is not None:
            query += " AND ticker = ?"
            params.append(ticker.upper())
        if enabled_only:
            query += " AND enabled = 1"

        query += " ORDER BY ticker, condition, threshold"

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_subscription(row) for row in rows]

    async def get_all_enabled_subscriptions(self) -> List[Subscription]:
        """Get all enabled subscriptions across all guilds."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM subscriptions WHERE enabled = 1"
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_subscription(row) for row in rows]

    async def delete_subscription(self, subscription_id: int, guild_id: int) -> bool:
        """Delete a subscription by ID (must match guild_id for security)."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM subscriptions WHERE id = ? AND guild_id = ?",
                (subscription_id, guild_id)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def subscription_exists(
        self,
        guild_id: int,
        ticker: str,
        condition: str,
        threshold: float,
        period: int
    ) -> bool:
        """Check if a subscription with these exact parameters already exists."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT 1 FROM subscriptions 
                   WHERE guild_id = ? AND ticker = ? 
                   AND condition = ? AND threshold = ? AND period = ?""",
                (guild_id, ticker.upper(), condition.upper(), threshold, period)
            ) as cursor:
                return await cursor.fetchone() is not None

    async def delete_user_subscriptions(self, guild_id: int, user_id: int) -> int:
        """Delete all subscriptions created by a specific user in a guild.
        
        Returns:
            Number of subscriptions deleted
        """
        async with aiosqlite.connect(self.db_path) as db:
            # First get the IDs to delete state records
            async with db.execute(
                """SELECT id FROM subscriptions 
                   WHERE guild_id = ? AND created_by_user_id = ?""",
                (guild_id, user_id)
            ) as cursor:
                rows = await cursor.fetchall()
                sub_ids = [row[0] for row in rows]
            
            if not sub_ids:
                return 0
            
            # Delete state records
            placeholders = ','.join('?' * len(sub_ids))
            await db.execute(
                f"DELETE FROM subscription_state WHERE subscription_id IN ({placeholders})",
                sub_ids
            )
            
            # Delete subscriptions
            cursor = await db.execute(
                "DELETE FROM subscriptions WHERE guild_id = ? AND created_by_user_id = ?",
                (guild_id, user_id)
            )
            await db.commit()
            return cursor.rowcount

    async def get_user_subscriptions(self, guild_id: int, user_id: int) -> List[Subscription]:
        """Get all subscriptions created by a specific user in a guild."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM subscriptions 
                   WHERE guild_id = ? AND created_by_user_id = ?
                   ORDER BY ticker, condition, threshold""",
                (guild_id, user_id)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_subscription(row) for row in rows]

    # ==================== Subscription State Operations ====================

    async def get_subscription_state(self, subscription_id: int) -> Optional[SubscriptionState]:
        """Get state for a subscription."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM subscription_state WHERE subscription_id = ?",
                (subscription_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return SubscriptionState(
                        subscription_id=row['subscription_id'],
                        last_rsi=row['last_rsi'],
                        last_close=row['last_close'],
                        last_date=row['last_date'],
                        last_status=row['last_status'],
                        last_alert_at=datetime.fromisoformat(row['last_alert_at']) if row['last_alert_at'] else None,
                        days_in_zone=row['days_in_zone'] or 0
                    )
                return None

    async def update_subscription_state(
        self,
        subscription_id: int,
        last_rsi: Optional[float] = None,
        last_close: Optional[float] = None,
        last_date: Optional[str] = None,
        last_status: Optional[str] = None,
        last_alert_at: Optional[datetime] = None,
        days_in_zone: Optional[int] = None
    ):
        """Update subscription state."""
        updates = []
        params = []

        if last_rsi is not None:
            updates.append("last_rsi = ?")
            params.append(last_rsi)
        if last_close is not None:
            updates.append("last_close = ?")
            params.append(last_close)
        if last_date is not None:
            updates.append("last_date = ?")
            params.append(last_date)
        if last_status is not None:
            updates.append("last_status = ?")
            params.append(last_status)
        if last_alert_at is not None:
            updates.append("last_alert_at = ?")
            params.append(last_alert_at.isoformat())
        if days_in_zone is not None:
            updates.append("days_in_zone = ?")
            params.append(days_in_zone)

        if updates:
            params.append(subscription_id)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"UPDATE subscription_state SET {', '.join(updates)} WHERE subscription_id = ?",
                    params
                )
                await db.commit()

    async def get_subscriptions_with_state(self, guild_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get subscriptions joined with their state."""
        query = """
            SELECT s.*, st.last_rsi, st.last_close, st.last_date, 
                   st.last_status, st.last_alert_at, st.days_in_zone
            FROM subscriptions s
            LEFT JOIN subscription_state st ON s.id = st.subscription_id
            WHERE s.enabled = 1
        """
        params = []

        if guild_id is not None:
            query += " AND s.guild_id = ?"
            params.append(guild_id)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    # ==================== Helper Methods ====================

    def _row_to_subscription(self, row) -> Subscription:
        """Convert a database row to a Subscription object."""
        return Subscription(
            id=row['id'],
            guild_id=row['guild_id'],
            channel_id=row['channel_id'],
            ticker=row['ticker'],
            condition=row['condition'],
            threshold=row['threshold'],
            period=row['period'],
            cooldown_hours=row['cooldown_hours'],
            enabled=bool(row['enabled']),
            created_by_user_id=row['created_by_user_id'] if 'created_by_user_id' in row.keys() else None,
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )

    async def get_unique_tickers(self) -> List[str]:
        """Get list of unique tickers with active subscriptions."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT DISTINCT ticker FROM subscriptions WHERE enabled = 1"
            ) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]

    async def get_unique_periods_for_ticker(self, ticker: str) -> List[int]:
        """Get unique RSI periods needed for a ticker."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT DISTINCT period FROM subscriptions 
                   WHERE ticker = ? AND enabled = 1""",
                (ticker.upper(),)
            ) as cursor:
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
