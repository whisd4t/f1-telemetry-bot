"""Математичні та фізичні розрахунки параметрів руху боліда Формули-1.

Реалізує:
- приведення телеметрії на єдину шкалу дистанції з кроком Δs = 1 м (numpy.interp);
- згладжування шуму датчиків ковзним середнім за 5 точками (Pandas rolling);
- переведення швидкості в СІ: v = v_км/год / 3.6;
- обчислення прискорення a = (v2 - v1) / (t2 - t1) та перевантажень n = |a| / g;
- обчислення різниці часу між двома пілотами Δt(s) = t2(s) - t1(s);
- визначення гальмівного шляху та оптимальної точки початку гальмування.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DEFAULT_MU, DISTANCE_STEP_M, GRAVITY, TRACK_SEGMENT_M


@dataclass
class ProcessedTelemetry:
    """Оброблена телеметрія одного кола на рівномірній сітці дистанції (крок 1 м)."""

    distance: np.ndarray  # s, м
    time: np.ndarray  # t(s), с
    speed_kmh: np.ndarray  # v_км/год (згладжена)
    speed_ms: np.ndarray  # v, м/с
    throttle: np.ndarray  # %, [0..100]
    brake: np.ndarray  # [0..1]
    accel_ms2: np.ndarray  # a, м/с²
    g_force: np.ndarray  # n = a / g (зі знаком: >0 розгін, <0 гальмування)
    x: np.ndarray  # координата X траси
    y: np.ndarray  # координата Y траси
    lap_length_m: float
    official_lap_time_sec: float
    computed_lap_time_sec: float
    lap_time_error_pct: float
    max_speed_kmh: float
    max_brake_g: float
    max_accel_g: float
    sector_boundaries_m: tuple[float, float]


@dataclass
class BrakingAnalysis:
    """Результати моделювання точки початку гальмування перед ключовим повільним поворотом."""

    s_fact: float  # Фактична координата натискання гальма, м
    s_apex: float  # Координата апексу (найменшої швидкості в повороті), м
    v_start_kmh: float  # Швидкість перед гальмуванням, км/год
    v_apex_kmh: float  # Швидкість в апексі, км/год
    mu: float  # Коефіцієнт зчеплення шин μ
    a_brake: float  # Максимальне сповільнення a_гальм = μ * g, м/с²
    s_brake: float  # Мінімально необхідний гальмівний шлях, м
    s_opt: float  # Розрахункова оптимальна координата початку гальмування, м
    delta_s: float  # Запас для пізнішого гальмування Δs = s_опт - s_факт, м
    apex_x: float  # Координата X точки апексу
    apex_y: float  # Координата Y точки апексу
    opt_x: float  # Координата X оптимальної точки гальмування
    opt_y: float  # Координата Y оптимальної точки гальмування


@dataclass
class ComparisonResult:
    """Результати порівняння двох пілотів на спільній шкалі дистанції."""

    distance: np.ndarray
    driver1: ProcessedTelemetry
    driver2: ProcessedTelemetry
    delta_t: np.ndarray  # Δt(s) = t2(s) - t1(s), с
    segment_dominance: list[dict[str, float | int]]


def prepare_telemetry_grid(
    raw_telemetry: pd.DataFrame,
    official_lap_time_sec: float,
    sector_times: tuple[float | None, float | None, float | None] = (None, None, None),
    common_distance: np.ndarray | None = None,
    step_m: float = DISTANCE_STEP_M,
) -> ProcessedTelemetry:
    """Очищує сирі дані датчиків, згладжує швидкість та інтерполює на сітку з кроком 1 м."""
    df = raw_telemetry.copy()

    # Переведення часу у секунди від початку кола
    if pd.api.types.is_timedelta64_dtype(df["Time"]):
        time_sec = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        time_sec = df["Time"].to_numpy(dtype=float)
        time_sec = time_sec - time_sec[0]

    df["_t_sec"] = time_sec
    df = df.dropna(subset=["Distance", "Speed", "_t_sec"])
    df = df.sort_values("Distance").drop_duplicates(subset=["Distance"], keep="first")

    raw_dist = df["Distance"].to_numpy(dtype=float)
    raw_dist = raw_dist - min(0.0, float(raw_dist[0]))
    lap_length = float(raw_dist[-1])

    if common_distance is not None:
        s_grid = common_distance
    else:
        s_grid = np.arange(0.0, lap_length, step_m, dtype=float)

    # Згладжування шуму датчика швидкості ковзним середнім за 5 сусідніми точками
    smoothed_speed_series = (
        df["Speed"].astype(float).rolling(window=5, center=True, min_periods=1).mean()
    )

    raw_time = df["_t_sec"].to_numpy(dtype=float)
    raw_throttle = np.clip(df["Throttle"].astype(float).to_numpy(), 0.0, 100.0)
    raw_brake = df["Brake"].astype(float).to_numpy()
    if raw_brake.max() > 1.0:
        raw_brake = raw_brake / 100.0
    raw_brake = np.clip(raw_brake, 0.0, 1.0)

    # Лінійна інтерполяція всіх каналів на єдину шкалу дистанції s
    t_grid = np.interp(s_grid, raw_dist, raw_time)
    speed_kmh = np.interp(s_grid, raw_dist, smoothed_speed_series.to_numpy(dtype=float))
    throttle = np.interp(s_grid, raw_dist, raw_throttle)
    brake = (np.interp(s_grid, raw_dist, raw_brake) >= 0.5).astype(float)

    if "X" in df.columns and "Y" in df.columns:
        x_grid = np.interp(s_grid, raw_dist, df["X"].astype(float).to_numpy())
        y_grid = np.interp(s_grid, raw_dist, df["Y"].astype(float).to_numpy())
    else:
        x_grid = np.zeros_like(s_grid)
        y_grid = np.zeros_like(s_grid)

    # 1. Переведення швидкості в систему СІ: v = v_км/год / 3.6
    speed_ms = speed_kmh / 3.6

    # 2. Обчислення прискорення a = (v2 - v1) / (t2 - t1) та перевантажень n = |a| / g
    dv = np.diff(speed_ms)
    dt = np.diff(t_grid)
    safe_dt = np.where(dt > 1e-4, dt, np.nan)
    raw_accel = np.where(np.isnan(safe_dt), 0.0, dv / safe_dt)
    accel_full = np.concatenate([[raw_accel[0] if len(raw_accel) else 0.0], raw_accel])

    # Додаткове м'яке згладжування похідної прискорення (вікно 5 точок) для усунення піків
    accel_smooth = (
        pd.Series(accel_full).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
    )
    # Обмеження фізично реалістичними межами датчика боліда Ф-1
    accel_smooth = np.clip(accel_smooth, -7.0 * GRAVITY, 3.5 * GRAVITY)
    g_force = accel_smooth / GRAVITY

    # Перевірка точності обчислення часу кола: T_бот = sum(Δs / v_i)
    safe_v = np.maximum(speed_ms, 1.0)
    ds = float(s_grid[1] - s_grid[0]) if len(s_grid) > 1 else step_m
    computed_lap_time = float(np.sum(ds / safe_v))
    error_pct = (
        abs(computed_lap_time - official_lap_time_sec) / official_lap_time_sec * 100.0
        if official_lap_time_sec > 0
        else 0.0
    )

    # Визначення меж секторів траси S1 / S2 / S3
    s1_t, s2_t, _ = sector_times
    if s1_t is not None and s2_t is not None and t_grid[-1] > 0:
        s1_dist = float(np.interp(s1_t, t_grid, s_grid))
        s2_dist = float(np.interp(s1_t + s2_t, t_grid, s_grid))
    else:
        s1_dist = lap_length / 3.0
        s2_dist = 2.0 * lap_length / 3.0

    max_brake_g = float(np.max(np.abs(np.minimum(g_force, 0.0))))
    max_accel_g = float(np.max(np.maximum(g_force, 0.0)))

    return ProcessedTelemetry(
        distance=s_grid,
        time=t_grid,
        speed_kmh=speed_kmh,
        speed_ms=speed_ms,
        throttle=throttle,
        brake=brake,
        accel_ms2=accel_smooth,
        g_force=g_force,
        x=x_grid,
        y=y_grid,
        lap_length_m=lap_length,
        official_lap_time_sec=official_lap_time_sec,
        computed_lap_time_sec=computed_lap_time,
        lap_time_error_pct=error_pct,
        max_speed_kmh=float(np.max(speed_kmh)),
        max_brake_g=max_brake_g,
        max_accel_g=max_accel_g,
        sector_boundaries_m=(s1_dist, s2_dist),
    )


def analyze_braking_point(
    telemetry: ProcessedTelemetry,
    mu: float = DEFAULT_MU,
    gravity: float = GRAVITY,
) -> BrakingAnalysis:
    """Знаходить найінтенсивнішу зону гальмування та обчислює оптимальну точку початку гальмування.

    Формули:
        a_гальм = μ * g
        s_гальм = (v_поч² - v_апекс²) / (2 * a_гальм)
        s_опт = s_апекс - s_гальм
        Δs = s_опт - s_факт
    """
    brake_mask = telemetry.brake >= 0.5
    n = len(telemetry.distance)

    # Пошук усіх неперервних відрізків натискання педалі гальма
    zones: list[tuple[int, int]] = []
    in_zone = False
    start_idx = 0
    for idx in range(n):
        if brake_mask[idx] and not in_zone:
            start_idx = idx
            in_zone = True
        elif not brake_mask[idx] and in_zone:
            if idx - start_idx >= 15:  # ігноруємо короткі дотики менш ніж 15 м
                zones.append((start_idx, idx - 1))
            in_zone = False
    if in_zone and n - start_idx >= 15:
        zones.append((start_idx, n - 1))

    best_zone: tuple[int, int, float] | None = None
    for z_start, z_end in zones:
        # Шукаємо мінімальну швидкість (апекс) у межах гальмування та до +120 м після нього
        search_end = min(n - 1, z_end + 120)
        apex_rel = int(np.argmin(telemetry.speed_kmh[z_start : search_end + 1]))
        apex_idx = z_start + apex_rel
        speed_drop = float(telemetry.speed_kmh[z_start] - telemetry.speed_kmh[apex_idx])
        if best_zone is None or speed_drop > best_zone[2]:
            best_zone = (z_start, apex_idx, speed_drop)

    # Якщо канал гальм порожній, визначаємо зону за максимальним падінням швидкості
    if best_zone is None:
        apex_idx = int(np.argmin(telemetry.speed_kmh))
        z_start = max(0, apex_idx - 150)
    else:
        z_start, apex_idx, _ = best_zone

    s_fact = float(telemetry.distance[z_start])
    s_apex = float(telemetry.distance[apex_idx])
    v_start_kmh = float(telemetry.speed_kmh[z_start])
    v_apex_kmh = float(telemetry.speed_kmh[apex_idx])

    v_start = v_start_kmh / 3.6
    v_apex = v_apex_kmh / 3.6

    a_brake = mu * gravity
    s_brake = max(0.0, (v_start**2 - v_apex**2) / (2.0 * a_brake))
    s_opt = s_apex - s_brake
    delta_s = s_opt - s_fact

    opt_x = float(np.interp(s_opt, telemetry.distance, telemetry.x))
    opt_y = float(np.interp(s_opt, telemetry.distance, telemetry.y))

    return BrakingAnalysis(
        s_fact=s_fact,
        s_apex=s_apex,
        v_start_kmh=v_start_kmh,
        v_apex_kmh=v_apex_kmh,
        mu=mu,
        a_brake=a_brake,
        s_brake=s_brake,
        s_opt=s_opt,
        delta_s=delta_s,
        apex_x=float(telemetry.x[apex_idx]),
        apex_y=float(telemetry.y[apex_idx]),
        opt_x=opt_x,
        opt_y=opt_y,
    )


def compare_two_drivers(
    raw_tel_1: pd.DataFrame,
    lap_time_1: float,
    sectors_1: tuple[float | None, float | None, float | None],
    raw_tel_2: pd.DataFrame,
    lap_time_2: float,
    sectors_2: tuple[float | None, float | None, float | None],
    step_m: float = DISTANCE_STEP_M,
    segment_length_m: float = TRACK_SEGMENT_M,
) -> ComparisonResult:
    """Вирівнює телеметрію двох пілотів на спільній дистанції, обчислює Δt(s) та домінування."""
    len_1 = float(raw_tel_1["Distance"].max() - min(0.0, float(raw_tel_1["Distance"].min())))
    len_2 = float(raw_tel_2["Distance"].max() - min(0.0, float(raw_tel_2["Distance"].min())))
    common_len = min(len_1, len_2)
    common_dist = np.arange(0.0, common_len, step_m, dtype=float)

    proc_1 = prepare_telemetry_grid(
        raw_tel_1, lap_time_1, sectors_1, common_distance=common_dist, step_m=step_m
    )
    proc_2 = prepare_telemetry_grid(
        raw_tel_2, lap_time_2, sectors_2, common_distance=common_dist, step_m=step_m
    )

    # 3. Різниця часу між двома пілотами: Δt(s) = t2(s) - t1(s)
    # Якщо Δt(s) зростає — перший пілот швидший на цій ділянці траси.
    delta_t = proc_2.time - proc_1.time

    # Розбиття траси на відрізки по 50 м для визначення переваги за середньою швидкістю
    segments: list[dict[str, float | int]] = []
    seg_step = max(1, int(round(segment_length_m / step_m)))
    for start in range(0, len(common_dist) - 1, seg_step):
        end = min(len(common_dist), start + seg_step + 1)
        avg_v1 = float(np.mean(proc_1.speed_kmh[start:end]))
        avg_v2 = float(np.mean(proc_2.speed_kmh[start:end]))
        segments.append(
            {
                "start_idx": start,
                "end_idx": end,
                "faster_driver": 1 if avg_v1 >= avg_v2 else 2,
                "avg_speed_1": avg_v1,
                "avg_speed_2": avg_v2,
            }
        )

    return ComparisonResult(
        distance=common_dist,
        driver1=proc_1,
        driver2=proc_2,
        delta_t=delta_t,
        segment_dominance=segments,
    )


def format_lap_time(seconds: float) -> str:
    """Форматує час кола у вигляд ХВ:СС.ммм (наприклад, 1:19.327)."""
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}:{rem:06.3f}"
