# F1 Telemetry & Physics Telegram Bot (МАН — 2026)

Інтерактивний Telegram-бот мовою Python для завантаження телеметрії болідів Формули-1, обчислення фізичних параметрів руху (прискорення, перевантаження у `g`, дельта часу на колі, оптимальна точка початку гальмування) та побудови графіків і кольорових схем трас.

## Структура проєкту

```text
f1-telemetry-man-bot/
├── bot.py                        # Точка входу, запуск aiogram 3.x та ініціалізація БД
├── config.py                     # Конфігурація, фізичні константи та параметри за замовчуванням
├── states.py                     # Скінченний автомат (StatesGroup) покрокового діалогу
├── keyboards.py                  # Інлайн-клавіатури вибору сезону, етапу, сесії, пілотів та μ
├── handlers.py                   # Асинхронні обробники команд та кнопок меню
├── database/
│   └── storage.py                # Збереження налаштувань та історії запитів у SQLite
├── services/
│   └── telemetry_loader.py       # Завантаження сесій FastF1, локальний кеш і валідація даних
├── core/
│   └── physics.py                # Інтерполяція (Δs = 1 м), згладжування, кінематика та гальмування
├── visualization/
│   └── plotter.py                # Графіки Matplotlib та схеми трас через io.BytesIO
├── tests/
│   └── test_physics_and_viz.py   # Автоматичні тести фізики, БД і генерації графіків
├── requirements.txt
└── .env.example
```

## Швидкий старт

1. Створіть віртуальне середовище та встановіть залежності:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Скопіюйте `.env.example` у `.env` та додайте свій `BOT_TOKEN`, отриманий у [@BotFather](https://t.me/BotFather).
3. Запустіть бота:
   ```bash
   python bot.py
   ```
4. Для запуску тестів виконайте:
   ```bash
   pytest -v
   ```
