"""Інлайн-клавіатури для меню Telegram-бота."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import ANALYSIS_MODES, AVAILABLE_SEASONS, SESSION_TYPES


def seasons_keyboard() -> InlineKeyboardMarkup:
    """Клавіатура вибору сезону чемпіонату Формули-1."""
    builder = InlineKeyboardBuilder()
    for year in AVAILABLE_SEASONS:
        builder.button(text=f"🏁 Сезон {year}", callback_data=f"season:{year}")
    builder.adjust(2)
    return builder.as_markup()


def grand_prix_keyboard(
    events: list[dict[str, str | int]],
    page: int = 0,
    per_page: int = 8,
) -> InlineKeyboardMarkup:
    """Клавіатура вибору етапу Гран-прі з пагінацією."""
    builder = InlineKeyboardBuilder()
    total_pages = max(1, (len(events) + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))

    start = page * per_page
    for event in events[start : start + per_page]:
        round_num = event["round"]
        name = str(event["name"])
        short_name = name.replace(" Grand Prix", "")
        builder.button(
            text=f"{round_num}. {short_name}",
            callback_data=f"gp:{round_num}",
        )
    builder.adjust(2)

    nav_buttons: list[InlineKeyboardButton] = []
    if page > 0:
        nav_buttons.append(
            InlineKeyboardButton(text="⬅️ Назад", callback_data=f"gp_page:{page - 1}")
        )
    nav_buttons.append(
        InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop")
    )
    if page < total_pages - 1:
        nav_buttons.append(
            InlineKeyboardButton(text="Далі ➡️", callback_data=f"gp_page:{page + 1}")
        )
    builder.row(*nav_buttons)
    builder.row(
        InlineKeyboardButton(text="🔄 Інший сезон", callback_data="restart")
    )
    return builder.as_markup()


def sessions_keyboard() -> InlineKeyboardMarkup:
    """Клавіатура вибору гоночної сесії (FP1–FP3, Q, R)."""
    builder = InlineKeyboardBuilder()
    for code, title in SESSION_TYPES.items():
        builder.button(text=title, callback_data=f"session:{code}")
    builder.adjust(2, 1, 2)
    builder.row(
        InlineKeyboardButton(text="⬅️ До списку Гран-прі", callback_data="back:gp")
    )
    return builder.as_markup()


def modes_keyboard() -> InlineKeyboardMarkup:
    """Клавіатура вибору режиму аналізу телеметрії."""
    builder = InlineKeyboardBuilder()
    for code, title in ANALYSIS_MODES.items():
        builder.button(text=title, callback_data=f"mode:{code}")
    builder.adjust(1)
    builder.row(
        InlineKeyboardButton(text="⬅️ До вибору сесії", callback_data="back:session")
    )
    return builder.as_markup()


def drivers_keyboard(
    drivers: list[dict[str, str]],
    selected: list[str] | None = None,
) -> InlineKeyboardMarkup:
    """Клавіатура вибору одного або двох пілотів зі списку учасників сесії."""
    selected = selected or []
    builder = InlineKeyboardBuilder()
    for drv in drivers:
        code = drv["code"]
        prefix = "✅ " if code in selected else ""
        builder.button(
            text=f"{prefix}{code} ({drv['last_name']})",
            callback_data=f"driver:{code}",
        )
    builder.adjust(2)
    builder.row(
        InlineKeyboardButton(text="⬅️ До вибору режиму", callback_data="back:mode")
    )
    return builder.as_markup()


def mu_presets_keyboard(current_mu: float) -> InlineKeyboardMarkup:
    """Клавіатура швидкого вибору коефіцієнта зчеплення шин μ."""
    presets = [
        ("🌧️ Волога траса (μ = 2.5)", 2.5),
        ("🟡 Середнє зчеплення (μ = 3.2)", 3.2),
        ("🔴 Стандарт Ф-1 (μ = 3.5)", 3.5),
        ("🟣 Висока притискна сила (μ = 4.2)", 4.2),
    ]
    builder = InlineKeyboardBuilder()
    for label, value in presets:
        mark = " ✓" if abs(value - current_mu) < 1e-6 else ""
        builder.button(text=f"{label}{mark}", callback_data=f"mu:{value}")
    builder.adjust(1)
    builder.row(
        InlineKeyboardButton(text="⬅️ До вибору пілота", callback_data="back:drivers")
    )
    return builder.as_markup()


def after_result_keyboard() -> InlineKeyboardMarkup:
    """Кнопки швидких дій після отримання графіків."""
    builder = InlineKeyboardBuilder()
    builder.button(text="👥 Обрати інших пілотів", callback_data="back:drivers")
    builder.button(text="⚙️ Змінити режим аналізу", callback_data="back:mode")
    builder.button(text="🏁 Новий запит", callback_data="restart")
    builder.adjust(1)
    return builder.as_markup()
