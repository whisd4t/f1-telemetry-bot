"""Конфігурація Telegram-бота для аналізу телеметрії Формули-1."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

BOT_TOKEN: str = os.getenv("BOT_TOKEN", "").strip()
CACHE_DIR: Path = Path(os.getenv("FASTF1_CACHE_DIR", str(BASE_DIR / "cache")))
DB_PATH: Path = Path(os.getenv("DB_PATH", str(BASE_DIR / "database" / "bot_data.db")))

# Фізичні константи та параметри моделі за замовчуванням
GRAVITY: float = 9.8  # м/с²
DEFAULT_MU: float = 3.5  # Ефективний коефіцієнт зчеплення шин боліда Ф-1
MIN_MU: float = 1.0
MAX_MU: float = 6.0
DISTANCE_STEP_M: float = 1.0  # Крок інтерполяції Δs = 1 м
TRACK_SEGMENT_M: float = 50.0  # Довжина міні-сектора на схемі траси (50 м)

# Доступні сезони та типи сесій у меню бота
AVAILABLE_SEASONS: list[int] = [2025, 2024, 2023, 2022]

SESSION_TYPES: dict[str, str] = {
    "FP1": "Практика 1 (FP1)",
    "FP2": "Практика 2 (FP2)",
    "FP3": "Практика 3 (FP3)",
    "Q": "Кваліфікація (Q)",
    "R": "Гонка (R)",
}

ANALYSIS_MODES: dict[str, str] = {
    "single": "📊 Аналіз одного пілота",
    "compare": "⚔️ Порівняння двох пілотів",
    "braking": "🛑 Розрахунок точки гальмування",
}
