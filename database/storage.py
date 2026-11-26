"""Робота з локальною базою даних SQLite для збереження налаштувань та історії запитів."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from config import DB_PATH, DEFAULT_MU


class BotStorage:
    """Асинхронне сховище даних користувачів на базі SQLite."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        """Створює файл БД та необхідні таблиці при першому запуску."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    default_mu REAL NOT NULL DEFAULT 3.5,
                    last_season INTEGER,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    season INTEGER NOT NULL,
                    grand_prix TEXT NOT NULL,
                    session_type TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    drivers TEXT NOT NULL,
                    mu REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.commit()

    async def get_user_mu(self, user_id: int) -> float:
        """Повертає збережений коефіцієнт зчеплення шин μ або значення за замовчуванням."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT default_mu FROM user_settings WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            return float(row[0]) if row else DEFAULT_MU

    async def save_user_settings(
        self,
        user_id: int,
        default_mu: float | None = None,
        last_season: int | None = None,
    ) -> None:
        """Оновлює персональні налаштування користувача."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        current_mu = await self.get_user_mu(user_id)
        mu_value = default_mu if default_mu is not None else current_mu

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO user_settings (user_id, default_mu, last_season, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    default_mu = excluded.default_mu,
                    last_season = COALESCE(excluded.last_season, user_settings.last_season),
                    updated_at = excluded.updated_at
                """,
                (user_id, mu_value, last_season, now),
            )
            await db.commit()

    async def add_history(
        self,
        user_id: int,
        season: int,
        grand_prix: str,
        session_type: str,
        mode: str,
        drivers: list[str],
        mu: float = DEFAULT_MU,
    ) -> None:
        """Додає запис про виконаний аналіз телеметрії в історію."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO query_history
                    (user_id, season, grand_prix, session_type, mode, drivers, mu, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    season,
                    grand_prix,
                    session_type,
                    mode,
                    ",".join(drivers),
                    mu,
                    now,
                ),
            )
            await db.commit()

    async def get_recent_history(
        self, user_id: int, limit: int = 5
    ) -> list[dict[str, str | int | float]]:
        """Повертає останні запити користувача."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT season, grand_prix, session_type, mode, drivers, mu, created_at
                FROM query_history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
