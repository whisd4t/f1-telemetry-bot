"""Головний файл запуску інтерактивного Telegram-бота для аналізу телеметрії Формули-1."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from config import BOT_TOKEN
from database.storage import BotStorage
from handlers import router
from services.telemetry_loader import enable_local_cache


async def main() -> None:
    """Ініціалізує кеш, базу даних SQLite та запускає асинхронний опитувач бота."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if not BOT_TOKEN:
        raise RuntimeError(
            "Не знайдено токен бота! Скопіюйте .env.example у .env та вкажіть BOT_TOKEN."
        )

    enable_local_cache()

    storage = BotStorage()
    await storage.init()

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logging.info("Telegram-бот F1 Telemetry запущено успішно.")
    await dp.start_polling(bot, storage=storage)


if __name__ == "__main__":
    asyncio.run(main())
