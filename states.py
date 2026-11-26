"""Машина станів (FSM) для покрокового діалогу користувача з Telegram-ботом."""

from aiogram.fsm.state import State, StatesGroup


class TelemetryStates(StatesGroup):
    """Стани покрокового вибору параметрів телеметрії Формули-1."""

    SelectingSeason = State()
    SelectingGrandPrix = State()
    SelectingSession = State()
    SelectingMode = State()
    SelectingDrivers = State()
    ConfiguringParameters = State()
