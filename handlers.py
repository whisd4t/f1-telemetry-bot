"""Обробники команд і покрокового діалогу Telegram-бота (aiogram 3.x)."""

from __future__ import annotations

import asyncio
import logging
import time

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from config import ANALYSIS_MODES, DEFAULT_MU, MAX_MU, MIN_MU, SESSION_TYPES
from core.physics import (
    analyze_braking_point,
    compare_two_drivers,
    format_lap_time,
    prepare_telemetry_grid,
)
from database.storage import BotStorage
from keyboards import (
    after_result_keyboard,
    drivers_keyboard,
    grand_prix_keyboard,
    modes_keyboard,
    mu_presets_keyboard,
    seasons_keyboard,
    sessions_keyboard,
)
from services.telemetry_loader import (
    TelemetryDataError,
    get_fastest_valid_lap,
    get_season_events,
    get_session_drivers,
    load_session,
)
from states import TelemetryStates
from visualization.plotter import (
    plot_braking_zone_detail,
    plot_drivers_comparison,
    plot_single_driver_telemetry,
    plot_track_dominance_map,
)

logger = logging.getLogger(__name__)
router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext, storage: BotStorage) -> None:
    """Стартова команда бота: вітання та вибір сезону."""
    await state.clear()
    await state.set_state(TelemetryStates.SelectingSeason)
    mu = await storage.get_user_mu(message.from_user.id) if message.from_user else DEFAULT_MU
    await message.answer(
        "🏎️ <b>F1 Telemetry &amp; Physics Bot</b>\n\n"
        "Я допомагаю завантажувати реальну телеметрію болідів Формули-1, "
        "порівнювати пілотів, обчислювати прискорення й перевантаження (g) "
        "та моделювати оптимальну точку початку гальмування.\n\n"
        f"⚙️ Поточний коефіцієнт зчеплення шин: <b>μ = {mu:.2f}</b>\n\n"
        "<b>Крок 1/5.</b> Оберіть сезон чемпіонату:",
        reply_markup=seasons_keyboard(),
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Довідка щодо можливостей та команд бота."""
    await message.answer(
        "ℹ️ <b>Довідка по роботі з ботом</b>\n\n"
        "• /start — почати новий аналіз телеметрії\n"
        "• /history — показати останні 5 ваших запитів\n"
        "• /cancel — скинути поточний діалог\n\n"
        "<b>Доступні режими аналізу:</b>\n"
        "1. <i>Аналіз одного пілота</i> — швидкість, газ, гальмо, перевантаження (g) та перевірка часу кола.\n"
        "2. <i>Порівняння двох пілотів</i> — 4-рівневий графік телеметрії з дельтою часу Δt(s) "
        "та кольорова схема траси (відрізки по 50 м).\n"
        "3. <i>Розрахунок точки гальмування</i> — моделювання гальмівного шляху "
        "за формулою s = (v₁² − v₂²) / (2·μ·g) із можливістю змінювати μ."
    )


@router.message(Command("history"))
async def cmd_history(message: Message, storage: BotStorage) -> None:
    """Відображає історію останніх запитів користувача з SQLite."""
    if not message.from_user:
        return
    records = await storage.get_recent_history(message.from_user.id, limit=5)
    if not records:
        await message.answer("📭 Історія запитів поки порожня. Натисніть /start для початку.")
        return

    lines = ["🕘 <b>Останні запити телеметрії:</b>\n"]
    for idx, rec in enumerate(records, start=1):
        mode_name = ANALYSIS_MODES.get(str(rec["mode"]), str(rec["mode"]))
        lines.append(
            f"{idx}. <b>{rec['season']} {rec['grand_prix']}</b> ({rec['session_type']}) — "
            f"{rec['drivers']} | {mode_name} | μ={float(rec['mu']):.2f}"
        )
    await message.answer("\n".join(lines))


@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    """Скидає стан діалогу."""
    await state.clear()
    await message.answer("❌ Поточний діалог скасовано. Натисніть /start, щоб почати заново.")


@router.callback_query(F.data == "noop")
async def cb_noop(callback: CallbackQuery) -> None:
    """Порожня дія для індикатора сторінки."""
    await callback.answer()


@router.callback_query(F.data == "restart")
async def cb_restart(callback: CallbackQuery, state: FSMContext) -> None:
    """Повернення до першого кроку — вибору сезону."""
    await state.clear()
    await state.set_state(TelemetryStates.SelectingSeason)
    if callback.message:
        await callback.message.answer(
            "<b>Крок 1/5.</b> Оберіть сезон чемпіонату Формули-1:",
            reply_markup=seasons_keyboard(),
        )
    await callback.answer()


@router.callback_query(TelemetryStates.SelectingSeason, F.data.startswith("season:"))
async def cb_select_season(
    callback: CallbackQuery, state: FSMContext, storage: BotStorage
) -> None:
    """Обробка вибору сезону та завантаження календаря етапів Гран-прі."""
    season = int(callback.data.split(":")[1])
    await callback.answer(f"Завантажую календар {season}...")
    if callback.from_user:
        await storage.save_user_settings(callback.from_user.id, last_season=season)

    try:
        events = await asyncio.to_thread(get_season_events, season)
    except TelemetryDataError as err:
        if callback.message:
            await callback.message.edit_text(f"⚠️ {err}", reply_markup=seasons_keyboard())
        return

    await state.update_data(season=season, events=events, gp_page=0)
    await state.set_state(TelemetryStates.SelectingGrandPrix)
    if callback.message:
        await callback.message.edit_text(
            f"🏁 Сезон: <b>{season}</b>\n\n<b>Крок 2/5.</b> Оберіть етап Гран-прі:",
            reply_markup=grand_prix_keyboard(events, page=0),
        )


@router.callback_query(TelemetryStates.SelectingGrandPrix, F.data.startswith("gp_page:"))
async def cb_gp_pagination(callback: CallbackQuery, state: FSMContext) -> None:
    """Перегортання сторінок списку етапів Гран-прі."""
    page = int(callback.data.split(":")[1])
    data = await state.get_data()
    events = data.get("events", [])
    await state.update_data(gp_page=page)
    if callback.message:
        await callback.message.edit_reply_markup(
            reply_markup=grand_prix_keyboard(events, page=page)
        )
    await callback.answer()


@router.callback_query(TelemetryStates.SelectingGrandPrix, F.data.startswith("gp:"))
async def cb_select_gp(callback: CallbackQuery, state: FSMContext) -> None:
    """Обробка вибору етапу Гран-прі."""
    round_num = int(callback.data.split(":")[1])
    data = await state.get_data()
    events = data.get("events", [])
    selected_event = next((e for e in events if e["round"] == round_num), None)
    if not selected_event:
        await callback.answer("Етап не знайдено. Оберіть зі списку.", show_alert=True)
        return

    await state.update_data(gp_round=round_num, gp_name=selected_event["name"])
    await state.set_state(TelemetryStates.SelectingSession)
    if callback.message:
        await callback.message.edit_text(
            f"🏁 <b>{data['season']} {selected_event['name']}</b>\n\n"
            "<b>Крок 3/5.</b> Оберіть гоночну сесію:",
            reply_markup=sessions_keyboard(),
        )
    await callback.answer()


@router.callback_query(F.data == "back:gp")
async def cb_back_to_gp(callback: CallbackQuery, state: FSMContext) -> None:
    """Повернення до вибору Гран-прі."""
    data = await state.get_data()
    events = data.get("events")
    if not events:
        await cb_restart(callback, state)
        return
    await state.set_state(TelemetryStates.SelectingGrandPrix)
    if callback.message:
        await callback.message.edit_text(
            f"🏁 Сезон: <b>{data['season']}</b>\n\n<b>Крок 2/5.</b> Оберіть етап Гран-прі:",
            reply_markup=grand_prix_keyboard(events, page=data.get("gp_page", 0)),
        )
    await callback.answer()


@router.callback_query(TelemetryStates.SelectingSession, F.data.startswith("session:"))
async def cb_select_session(callback: CallbackQuery, state: FSMContext) -> None:
    """Обробка вибору сесії та її завантаження в FastF1 (з кешем)."""
    session_code = callback.data.split(":")[1]
    data = await state.get_data()
    season = data["season"]
    gp_round = data["gp_round"]
    gp_name = data["gp_name"]

    await callback.answer("Завантажую телеметрію сесії...")
    if callback.message:
        await callback.message.edit_text(
            f"⏳ Завантажую дані <b>{season} {gp_name} ({session_code})</b>...\n"
            "<i>Перше завантаження може тривати до 10–15 секунд, наступні беруться з кешу.</i>"
        )

    t0 = time.perf_counter()
    try:
        session = await asyncio.to_thread(load_session, season, gp_round, session_code)
        drivers = await asyncio.to_thread(get_session_drivers, session)
    except TelemetryDataError as err:
        if callback.message:
            await callback.message.edit_text(
                f"⚠️ {err}\n\nОберіть іншу сесію:",
                reply_markup=sessions_keyboard(),
            )
        return

    load_elapsed = time.perf_counter() - t0
    await state.update_data(
        session_code=session_code,
        drivers_list=drivers,
        selected_drivers=[],
        load_time_sec=load_elapsed,
    )
    await state.set_state(TelemetryStates.SelectingMode)
    if callback.message:
        await callback.message.edit_text(
            f"✅ Сесію завантажено за <b>{load_elapsed:.2f} с</b>!\n"
            f"🏁 <b>{season} {gp_name} — {SESSION_TYPES.get(session_code, session_code)}</b>\n\n"
            "<b>Крок 4/5.</b> Оберіть режим аналізу:",
            reply_markup=modes_keyboard(),
        )


@router.callback_query(F.data == "back:session")
async def cb_back_to_session(callback: CallbackQuery, state: FSMContext) -> None:
    """Повернення до вибору сесії."""
    data = await state.get_data()
    if "gp_name" not in data:
        await cb_restart(callback, state)
        return
    await state.set_state(TelemetryStates.SelectingSession)
    if callback.message:
        await callback.message.edit_text(
            f"🏁 <b>{data['season']} {data['gp_name']}</b>\n\n"
            "<b>Крок 3/5.</b> Оберіть гоночну сесію:",
            reply_markup=sessions_keyboard(),
        )
    await callback.answer()


@router.callback_query(TelemetryStates.SelectingMode, F.data.startswith("mode:"))
async def cb_select_mode(callback: CallbackQuery, state: FSMContext) -> None:
    """Обробка вибору режиму аналізу."""
    mode = callback.data.split(":")[1]
    data = await state.get_data()
    drivers = data.get("drivers_list", [])
    await state.update_data(mode=mode, selected_drivers=[])
    await state.set_state(TelemetryStates.SelectingDrivers)

    prompt = (
        "Оберіть <b>двох пілотів</b> для порівняння (спочатку першого, потім другого):"
        if mode == "compare"
        else "Оберіть <b>пілота</b> зі списку учасників сесії:"
    )
    if callback.message:
        await callback.message.edit_text(
            f"⚙️ Режим: <b>{ANALYSIS_MODES[mode]}</b>\n\n<b>Крок 5/5.</b> {prompt}",
            reply_markup=drivers_keyboard(drivers, selected=[]),
        )
    await callback.answer()


@router.callback_query(F.data == "back:mode")
async def cb_back_to_mode(callback: CallbackQuery, state: FSMContext) -> None:
    """Повернення до вибору режиму аналізу."""
    data = await state.get_data()
    if "session_code" not in data:
        await cb_restart(callback, state)
        return
    await state.set_state(TelemetryStates.SelectingMode)
    if callback.message:
        await callback.message.answer(
            f"🏁 <b>{data['season']} {data['gp_name']} ({data['session_code']})</b>\n\n"
            "Оберіть режим аналізу:",
            reply_markup=modes_keyboard(),
        )
    await callback.answer()


@router.callback_query(F.data == "back:drivers")
async def cb_back_to_drivers(callback: CallbackQuery, state: FSMContext) -> None:
    """Повернення до вибору пілотів у поточній сесії."""
    data = await state.get_data()
    if "drivers_list" not in data or "mode" not in data:
        await cb_restart(callback, state)
        return
    await state.update_data(selected_drivers=[])
    await state.set_state(TelemetryStates.SelectingDrivers)
    if callback.message:
        await callback.message.answer(
            f"⚙️ Режим: <b>{ANALYSIS_MODES[data['mode']]}</b>\n\nОберіть пілота(ів):",
            reply_markup=drivers_keyboard(data["drivers_list"], selected=[]),
        )
    await callback.answer()


@router.callback_query(TelemetryStates.SelectingDrivers, F.data.startswith("driver:"))
async def cb_select_driver(
    callback: CallbackQuery, state: FSMContext, storage: BotStorage
) -> None:
    """Обробка вибору одного або двох пілотів."""
    code = callback.data.split(":")[1]
    data = await state.get_data()
    mode = data["mode"]
    selected: list[str] = list(data.get("selected_drivers", []))

    if code in selected:
        selected.remove(code)
    else:
        selected.append(code)

    await state.update_data(selected_drivers=selected)

    if mode == "compare" and len(selected) < 2:
        if callback.message:
            await callback.message.edit_text(
                f"⚙️ Режим: <b>{ANALYSIS_MODES[mode]}</b>\n"
                f"Перший пілот: <b>{selected[0]}</b>\n\n"
                "Тепер оберіть <b>другого пілота</b>:",
                reply_markup=drivers_keyboard(data["drivers_list"], selected=selected),
            )
        await callback.answer(f"Обрано {code}. Оберіть другого пілота.")
        return

    if mode == "braking":
        mu = await storage.get_user_mu(callback.from_user.id)
        await state.set_state(TelemetryStates.ConfiguringParameters)
        if callback.message:
            await callback.message.edit_text(
                f"🛑 <b>Моделювання точки гальмування ({selected[0]})</b>\n\n"
                f"Оберіть коефіцієнт зчеплення шин <b>μ</b> із кнопок нижче "
                f"або надішліть власне число від {MIN_MU} до {MAX_MU} у чат "
                f"(наприклад, <code>3.8</code>):",
                reply_markup=mu_presets_keyboard(mu),
            )
        await callback.answer()
        return

    await callback.answer("Виконую фізичні розрахунки та будую графіки...")
    mu = await storage.get_user_mu(callback.from_user.id)
    if callback.message:
        await _execute_and_send_analysis(callback.message, state, storage, callback.from_user.id, mu)


@router.callback_query(TelemetryStates.ConfiguringParameters, F.data.startswith("mu:"))
async def cb_select_mu_preset(
    callback: CallbackQuery, state: FSMContext, storage: BotStorage
) -> None:
    """Вибір коефіцієнта зчеплення шин μ із готових пресетів."""
    mu = float(callback.data.split(":")[1])
    await storage.save_user_settings(callback.from_user.id, default_mu=mu)
    await callback.answer(f"Встановлено μ = {mu:.2f}")
    if callback.message:
        await _execute_and_send_analysis(callback.message, state, storage, callback.from_user.id, mu)


@router.message(TelemetryStates.ConfiguringParameters)
async def msg_input_mu(message: Message, state: FSMContext, storage: BotStorage) -> None:
    """Обробка ручного введення коефіцієнта зчеплення шин μ із перевіркою коректності."""
    raw_text = (message.text or "").strip().replace(",", ".")
    try:
        mu = float(raw_text)
    except ValueError:
        await message.answer(
            f"⚠️ Некоректне число «{message.text}». "
            f"Введіть десяткове значення коефіцієнта μ від {MIN_MU} до {MAX_MU}, "
            "наприклад: <code>3.5</code>"
        )
        return

    if not (MIN_MU <= mu <= MAX_MU):
        await message.answer(
            f"⚠️ Значення μ = {mu:.2f} виходить за фізично реалістичні межі "
            f"[{MIN_MU}; {MAX_MU}]. Спробуйте ще раз (наприклад, <code>3.5</code>):"
        )
        return

    if message.from_user:
        await storage.save_user_settings(message.from_user.id, default_mu=mu)
        await _execute_and_send_analysis(message, state, storage, message.from_user.id, mu)


@router.callback_query()
async def cb_stale_button(callback: CallbackQuery) -> None:
    """Коректна реакція на повторне натискання застарілих кнопок у чаті."""
    await callback.answer(
        "Ця кнопка вже неактуальна. Натисніть /start для нового запиту.",
        show_alert=True,
    )


async def _execute_and_send_analysis(
    message: Message,
    state: FSMContext,
    storage: BotStorage,
    user_id: int,
    mu: float,
) -> None:
    """Виконує обчислення, генерує графіки у io.BytesIO та надсилає результати в Telegram."""
    data = await state.get_data()
    season: int = data["season"]
    gp_round: int = data["gp_round"]
    gp_name: str = data["gp_name"]
    session_code: str = data["session_code"]
    mode: str = data["mode"]
    selected: list[str] = data["selected_drivers"]

    title = f"{season} {gp_name} ({session_code})"
    status_msg = await message.answer("⏳ Обробляю телеметрію та будую графіки...")

    t_start = time.perf_counter()
    try:
        session = await asyncio.to_thread(load_session, season, gp_round, session_code)

        if mode == "single":
            bundle = await asyncio.to_thread(get_fastest_valid_lap, session, selected[0], 0)
            proc = await asyncio.to_thread(
                prepare_telemetry_grid,
                bundle.telemetry,
                bundle.lap_time_sec,
                bundle.sector_times,
            )
            braking = await asyncio.to_thread(analyze_braking_point, proc, mu)
            img_buf = await asyncio.to_thread(
                plot_single_driver_telemetry,
                proc,
                braking,
                bundle.driver_code,
                bundle.team_color,
                title,
            )

            elapsed = time.perf_counter() - t_start
            caption = (
                f"🏎️ <b>{title} — {bundle.driver_name} ({bundle.driver_code})</b>\n"
                f"• Команда: {bundle.team_name} | Шини: {bundle.compound}\n"
                f"• Офіційний час кола (FIA): <b>{format_lap_time(proc.official_lap_time_sec)}</b>\n"
                f"• Розрахунковий час кола: <b>{format_lap_time(proc.computed_lap_time_sec)}</b> "
                f"(похибка {proc.lap_time_error_pct:.3f}%)\n"
                f"• Максимальна швидкість: <b>{proc.max_speed_kmh:.1f} км/год</b>\n"
                f"• Пікове гальмівне перевантаження: <b>{proc.max_brake_g:.2f} g</b>\n"
                f"• Опт. точка гальмування (μ={mu:.2f}): <b>{braking.s_opt:.1f} м</b> "
                f"(факт: {braking.s_fact:.1f} м, Δs = {braking.delta_s:+.1f} м)\n"
                f"⏱️ Час обробки: {elapsed:.2f} с"
            )
            await message.answer_photo(
                photo=BufferedInputFile(img_buf.getvalue(), filename=f"telemetry_{selected[0]}.png"),
                caption=caption,
                reply_markup=after_result_keyboard(),
            )

        elif mode == "compare":
            b1 = await asyncio.to_thread(get_fastest_valid_lap, session, selected[0], 0)
            b2 = await asyncio.to_thread(get_fastest_valid_lap, session, selected[1], 1)
            comp = await asyncio.to_thread(
                compare_two_drivers,
                b1.telemetry,
                b1.lap_time_sec,
                b1.sector_times,
                b2.telemetry,
                b2.lap_time_sec,
                b2.sector_times,
            )
            braking = await asyncio.to_thread(analyze_braking_point, comp.driver1, mu)

            tel_buf = await asyncio.to_thread(
                plot_drivers_comparison,
                comp,
                braking,
                b1.driver_code,
                b2.driver_code,
                b1.team_color,
                b2.team_color,
                title,
            )
            map_buf = await asyncio.to_thread(
                plot_track_dominance_map,
                comp,
                braking,
                b1.driver_code,
                b2.driver_code,
                b1.team_color,
                b2.team_color,
                title,
            )

            elapsed = time.perf_counter() - t_start
            lap_diff = b2.lap_time_sec - b1.lap_time_sec
            faster = b1.driver_code if lap_diff >= 0 else b2.driver_code
            caption_tel = (
                f"⚔️ <b>{title}: {b1.driver_code} vs {b2.driver_code}</b>\n"
                f"• {b1.driver_code}: <b>{format_lap_time(b1.lap_time_sec)}</b> | "
                f"Vmax {comp.driver1.max_speed_kmh:.1f} км/год | {comp.driver1.max_brake_g:.2f} g\n"
                f"• {b2.driver_code}: <b>{format_lap_time(b2.lap_time_sec)}</b> | "
                f"Vmax {comp.driver2.max_speed_kmh:.1f} км/год | {comp.driver2.max_brake_g:.2f} g\n"
                f"• Різниця часу: <b>{abs(lap_diff):.3f} с</b> на користь <b>{faster}</b>\n"
                f"⏱️ Час обробки: {elapsed:.2f} с"
            )
            await message.answer_photo(
                photo=BufferedInputFile(
                    tel_buf.getvalue(),
                    filename=f"compare_{b1.driver_code}_{b2.driver_code}.png",
                ),
                caption=caption_tel,
            )
            await message.answer_photo(
                photo=BufferedInputFile(
                    map_buf.getvalue(),
                    filename=f"track_map_{b1.driver_code}_{b2.driver_code}.png",
                ),
                caption=(
                    f"🗺️ <b>Схема траси {gp_name}</b>\n"
                    f"Кожен 50-метровий відрізок зафарбовано в колір пілота з вищою середньою швидкістю."
                ),
                reply_markup=after_result_keyboard(),
            )

        else:  # mode == "braking"
            bundle = await asyncio.to_thread(get_fastest_valid_lap, session, selected[0], 0)
            proc = await asyncio.to_thread(
                prepare_telemetry_grid,
                bundle.telemetry,
                bundle.lap_time_sec,
                bundle.sector_times,
            )
            braking = await asyncio.to_thread(analyze_braking_point, proc, mu)
            brake_buf = await asyncio.to_thread(
                plot_braking_zone_detail,
                proc,
                braking,
                bundle.driver_code,
                bundle.team_color,
                title,
            )

            elapsed = time.perf_counter() - t_start
            advice = (
                f"Пілот має теоретичний запас гальмувати пізніше на <b>{braking.delta_s:.1f} м</b>."
                if braking.delta_s > 0
                else f"За заданого μ пілоту варто почати гальмування раніше на <b>{abs(braking.delta_s):.1f} м</b>."
            )
            caption_brake = (
                f"🛑 <b>Аналіз гальмування: {bundle.driver_name} ({bundle.driver_code})</b>\n"
                f"• Етап: {title}\n"
                f"• Коефіцієнт зчеплення шин: <b>μ = {braking.mu:.2f}</b> "
                f"(a = {braking.a_brake:.1f} м/с²)\n"
                f"• Швидкість перед гальмуванням: <b>{braking.v_start_kmh:.1f} км/год</b>\n"
                f"• Мінімальна швидкість в апексі: <b>{braking.v_apex_kmh:.1f} км/год</b>\n"
                f"• Розрахунковий гальмівний шлях: <b>{braking.s_brake:.1f} м</b>\n"
                f"• Фактична точка гальмування: <b>{braking.s_fact:.1f} м</b>\n"
                f"• Оптимальна точка гальмування: <b>{braking.s_opt:.1f} м</b>\n"
                f"📌 {advice}\n"
                f"⏱️ Час обробки: {elapsed:.2f} с"
            )
            await message.answer_photo(
                photo=BufferedInputFile(
                    brake_buf.getvalue(),
                    filename=f"braking_{bundle.driver_code}.png",
                ),
                caption=caption_brake,
                reply_markup=after_result_keyboard(),
            )

        await storage.add_history(
            user_id=user_id,
            season=season,
            grand_prix=gp_name,
            session_type=session_code,
            mode=mode,
            drivers=selected,
            mu=mu,
        )
    except TelemetryDataError as err:
        await message.answer(
            f"⚠️ <b>Не вдалося виконати аналіз:</b>\n{err}",
            reply_markup=after_result_keyboard(),
        )
    except Exception:
        logger.exception("Несподівана помилка під час обробки запиту телеметрії")
        await message.answer(
            "⚠️ Виникла непередбачена помилка під час розрахунку. Спробуйте обрати іншого пілота або етап.",
            reply_markup=after_result_keyboard(),
        )
    finally:
        try:
            await status_msg.delete()
        except Exception:
            pass
