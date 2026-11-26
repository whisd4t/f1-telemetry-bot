"""Автоматичні тести модулів фізики, БД SQLite, клавіатур та візуалізації."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.physics import (
    analyze_braking_point,
    compare_two_drivers,
    format_lap_time,
    prepare_telemetry_grid,
)
from database.storage import BotStorage
from keyboards import (
    drivers_keyboard,
    grand_prix_keyboard,
    modes_keyboard,
    mu_presets_keyboard,
    seasons_keyboard,
    sessions_keyboard,
)
from visualization.plotter import (
    plot_braking_zone_detail,
    plot_drivers_comparison,
    plot_single_driver_telemetry,
    plot_track_dominance_map,
)


def _make_synthetic_lap(speed_offset_kmh: float = 0.0) -> tuple[pd.DataFrame, float]:
    """Створює реалістичне синтетичне коло довжиною 3000 м із зоною жорсткого гальмування."""
    n = 600
    distance = np.linspace(0.0, 3000.0, n)
    speed_kmh = np.full(n, 310.0 + speed_offset_kmh)

    # Зона гальмування перед повільним поворотом (800..950 м)
    brake = np.zeros(n, dtype=bool)
    throttle = np.full(n, 100.0)
    for i, s in enumerate(distance):
        if 800.0 <= s <= 940.0:
            frac = (s - 800.0) / 140.0
            speed_kmh[i] = (310.0 + speed_offset_kmh) - 220.0 * frac
            brake[i] = True
            throttle[i] = 0.0
        elif 940.0 < s <= 1150.0:
            frac = (s - 940.0) / 210.0
            speed_kmh[i] = (90.0 + speed_offset_kmh) + 220.0 * frac
            throttle[i] = 85.0

    # Інтегруємо час по довжині кола
    speed_ms = speed_kmh / 3.6
    dt = np.diff(distance) / ((speed_ms[:-1] + speed_ms[1:]) / 2.0)
    t_sec = np.concatenate([[0.0], np.cumsum(dt)])
    lap_time = float(t_sec[-1])

    # Проста замкнута конфігурація траси (овал)
    angle = np.linspace(0.0, 2.0 * np.pi, n)
    x = 700.0 * np.cos(angle)
    y = 400.0 * np.sin(angle)

    df = pd.DataFrame(
        {
            "Time": pd.to_timedelta(t_sec, unit="s"),
            "Distance": distance,
            "Speed": speed_kmh,
            "Throttle": throttle,
            "Brake": brake,
            "X": x,
            "Y": y,
        }
    )
    return df, lap_time


def test_prepare_telemetry_and_lap_time_accuracy() -> None:
    """Перевіряє інтерполяцію на сітку 1 м та точність розрахунку часу кола (< 0.5%)."""
    raw, official_time = _make_synthetic_lap()
    proc = prepare_telemetry_grid(raw, official_lap_time_sec=official_time)

    assert len(proc.distance) == 3000
    assert np.allclose(np.diff(proc.distance), 1.0)
    assert proc.lap_time_error_pct < 0.5
    assert proc.max_brake_g > 1.5
    assert proc.max_speed_kmh == pytest.approx(310.0, abs=2.0)


def test_braking_point_physics_formula() -> None:
    """Перевіряє відповідність розрахунку формулі s_гальм = (v1² - v2²) / (2 * μ * g)."""
    raw, official_time = _make_synthetic_lap()
    proc = prepare_telemetry_grid(raw, official_lap_time_sec=official_time)

    res_low_mu = analyze_braking_point(proc, mu=3.0)
    res_high_mu = analyze_braking_point(proc, mu=4.2)

    expected_s_brake = (
        (res_low_mu.v_start_kmh / 3.6) ** 2 - (res_low_mu.v_apex_kmh / 3.6) ** 2
    ) / (2.0 * 3.0 * 9.8)
    assert res_low_mu.s_brake == pytest.approx(expected_s_brake, rel=1e-4)
    # За більшого зчеплення μ гальмівний шлях менший, а оптимальна точка гальмування пізніша
    assert res_high_mu.s_brake < res_low_mu.s_brake
    assert res_high_mu.s_opt > res_low_mu.s_opt


def test_two_driver_comparison_and_plots() -> None:
    """Перевіряє порівняння двох пілотів і генерацію всіх PNG-графіків у io.BytesIO."""
    raw1, t1 = _make_synthetic_lap(speed_offset_kmh=5.0)
    raw2, t2 = _make_synthetic_lap(speed_offset_kmh=0.0)

    comp = compare_two_drivers(raw1, t1, (None, None, None), raw2, t2, (None, None, None))
    # Перший пілот швидший, тому t2 - t1 на фініші додатне
    assert comp.delta_t[-1] > 0.0
    assert len(comp.segment_dominance) >= 50

    braking = analyze_braking_point(comp.driver1, mu=3.5)

    for buf in [
        plot_single_driver_telemetry(comp.driver1, braking, "NOR", "#FF8000", "2024 Monza (Q)"),
        plot_drivers_comparison(comp, braking, "NOR", "LEC", "#FF8000", "#DC0000", "2024 Monza (Q)"),
        plot_track_dominance_map(comp, braking, "NOR", "LEC", "#FF8000", "#DC0000", "2024 Monza (Q)"),
        plot_braking_zone_detail(comp.driver1, braking, "NOR", "#FF8000", "2024 Monza (Q)"),
    ]:
        data = buf.getvalue()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 10_000


def test_sqlite_storage_and_keyboards(tmp_path: Path) -> None:
    """Перевіряє збереження налаштувань/історії в SQLite та побудову клавіатур."""
    async def _run() -> None:
        storage = BotStorage(db_path=tmp_path / "test.db")
        await storage.init()
        assert await storage.get_user_mu(101) == pytest.approx(3.5)

        await storage.save_user_settings(101, default_mu=4.1, last_season=2024)
        assert await storage.get_user_mu(101) == pytest.approx(4.1)

        await storage.add_history(101, 2024, "Italian Grand Prix", "Q", "compare", ["NOR", "LEC"], 4.1)
        hist = await storage.get_recent_history(101, limit=5)
        assert len(hist) == 1
        assert hist[0]["drivers"] == "NOR,LEC"

    asyncio.run(_run())

    assert seasons_keyboard().inline_keyboard
    assert grand_prix_keyboard([{"round": 1, "name": "Bahrain Grand Prix"}]).inline_keyboard
    assert sessions_keyboard().inline_keyboard
    assert modes_keyboard().inline_keyboard
    assert drivers_keyboard([{"code": "NOR", "last_name": "Norris"}]).inline_keyboard
    assert mu_presets_keyboard(3.5).inline_keyboard
    assert format_lap_time(79.327) == "1:19.327"
