"""Модуль завантаження та перевірки телеметрії Формули-1 (FastF1 + резервний OpenF1 API)."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd
import requests

from config import CACHE_DIR

logger = logging.getLogger(__name__)

_LOADER_LOCK = threading.Lock()
_SESSION_MEMORY_CACHE: dict[tuple[int, int | str, str], Any] = {}

OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPENF1_SESSION_MAP = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q": "Qualifying",
    "R": "Race",
}

FALLBACK_COLORS: list[str] = [
    "#FF8000",
    "#DC0000",
    "#3671C6",
    "#00D2BE",
    "#22C55E",
    "#A855F7",
]


class TelemetryDataError(Exception):
    """Помилка отримання або валідації телеметричних даних."""


@dataclass
class DriverLapBundle:
    """Набір перевірених даних найшвидшого кола одного пілота."""

    driver_code: str
    driver_name: str
    team_name: str
    team_color: str
    lap_time_sec: float
    compound: str
    lap_number: int
    sector_times: tuple[float | None, float | None, float | None]
    telemetry: pd.DataFrame


@dataclass
class OpenF1SessionProxy:
    """Легкий контейнер сесії OpenF1, сумісний з інтерфейсом завантажувача FastF1."""

    season: int
    grand_prix: str
    session_type: str
    session_key: int
    drivers: list[dict[str, str]]
    laps_raw: list[dict[str, Any]]
    cache_dir: Path


def enable_local_cache(cache_dir: Path = CACHE_DIR) -> Path:
    """Увімкнення локального дискового кешу бібліотеки FastF1."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(fastf1.Cache, "configure"):
        fastf1.Cache.configure(cache_dir=str(cache_dir))
    else:
        fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir


def _openf1_get(endpoint: str, params: dict[str, Any], cache_file: Path | None = None) -> list[dict[str, Any]]:
    """Виконує кешований HTTP-запит до відкритого API OpenF1 із захистом від ліміту запитів (429)."""
    if cache_file is not None and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    last_exc: Exception | None = None
    for attempt in range(4):
        if attempt > 0:
            time.sleep(1.2 * attempt)
        else:
            time.sleep(0.45)
        try:
            resp = requests.get(f"{OPENF1_BASE_URL}/{endpoint}", params=params, timeout=25)
            if resp.status_code == 429:
                last_exc = requests.HTTPError("429 Too Many Requests", response=resp)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        except requests.RequestException as exc:
            last_exc = exc

    assert last_exc is not None
    raise last_exc


def get_season_events(season: int) -> list[dict[str, str | int]]:
    """Повертає список етапів Гран-прі для обраного сезону (без передсезонних тестів)."""
    enable_local_cache()
    try:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        events: list[dict[str, str | int]] = []
        for _, row in schedule.iterrows():
            round_num = int(row["RoundNumber"])
            if round_num <= 0:
                continue
            events.append(
                {
                    "round": round_num,
                    "name": str(row["EventName"]),
                    "country": str(row.get("Country", "")),
                    "location": str(row.get("Location", "")),
                }
            )
        if events:
            return events
    except Exception:
        logger.warning("FastF1 не зміг отримати календар %s, використовую OpenF1.", season)

    try:
        cache_path = CACHE_DIR / "openf1" / f"meetings_{season}.json"
        meetings = _openf1_get("meetings", {"year": season}, cache_file=cache_path)
        events = []
        round_idx = 0
        for m in meetings:
            name = str(m.get("meeting_name", ""))
            if "testing" in name.lower():
                continue
            round_idx += 1
            events.append(
                {
                    "round": round_idx,
                    "name": name,
                    "country": str(m.get("country_name", "")),
                    "location": str(m.get("location", "")),
                }
            )
        if events:
            return events
    except Exception as exc:
        raise TelemetryDataError(
            f"Не вдалося завантажити календар сезону {season}. Спробуйте пізніше."
        ) from exc

    raise TelemetryDataError(f"Для сезону {season} не знайдено етапів Гран-прі.")


def _load_openf1_session(season: int, grand_prix: int | str, session_type: str) -> OpenF1SessionProxy:
    """Резервне завантаження сесії через відкритий OpenF1 API із локальним кешем."""
    session_name = OPENF1_SESSION_MAP.get(session_type, session_type)
    openf1_dir = CACHE_DIR / "openf1"
    openf1_dir.mkdir(parents=True, exist_ok=True)

    # Визначаємо назву або локацію етапу
    target_query = str(grand_prix).lower()
    if isinstance(grand_prix, int):
        events = get_season_events(season)
        matched = next((e for e in events if e["round"] == grand_prix), None)
        if matched:
            target_query = str(matched["location"] or matched["name"]).lower()

    sessions = _openf1_get(
        "sessions",
        {"year": season, "session_name": session_name},
        cache_file=openf1_dir / f"sessions_{season}_{session_type}.json",
    )
    matched_session = next(
        (
            s
            for s in sessions
            if target_query in str(s.get("location", "")).lower()
            or target_query in str(s.get("circuit_short_name", "")).lower()
            or target_query in str(s.get("country_name", "")).lower()
        ),
        None,
    )
    if not matched_session:
        raise TelemetryDataError(
            f"Не знайдено сесію {season} {grand_prix} ({session_type}) у відкритих архівах."
        )

    session_key = int(matched_session["session_key"])
    drivers_raw = _openf1_get(
        "drivers",
        {"session_key": session_key},
        cache_file=openf1_dir / f"drivers_{session_key}.json",
    )
    laps_raw = _openf1_get(
        "laps",
        {"session_key": session_key},
        cache_file=openf1_dir / f"laps_{session_key}.json",
    )

    drivers_map: dict[str, dict[str, str]] = {}
    for d in drivers_raw:
        code = str(d.get("name_acronym") or "").strip()
        if not code:
            continue
        raw_col = str(d.get("team_colour") or "").strip().lstrip("#")
        drivers_map[code] = {
            "code": code,
            "driver_number": str(d.get("driver_number", "")),
            "last_name": str(d.get("last_name") or code),
            "full_name": str(d.get("full_name") or code),
            "team": str(d.get("team_name") or ""),
            "color": f"#{raw_col}" if len(raw_col) == 6 else FALLBACK_COLORS[len(drivers_map) % len(FALLBACK_COLORS)],
        }

    drivers_list = sorted(drivers_map.values(), key=lambda x: x["code"])
    if not drivers_list or not laps_raw:
        raise TelemetryDataError("У вибраній сесії відсутні дані про кола або пілотів.")

    return OpenF1SessionProxy(
        season=season,
        grand_prix=str(matched_session.get("location") or grand_prix),
        session_type=session_type,
        session_key=session_key,
        drivers=drivers_list,
        laps_raw=laps_raw,
        cache_dir=openf1_dir,
    )


def load_session(season: int, grand_prix: int | str, session_type: str) -> Any:
    """Завантажує сесію через FastF1 (із кешем), а за потреби — через OpenF1 API."""
    enable_local_cache()
    cache_key = (season, grand_prix, session_type)

    with _LOADER_LOCK:
        if cache_key in _SESSION_MEMORY_CACHE:
            return _SESSION_MEMORY_CACHE[cache_key]

        # 1. Основне джерело — бібліотека FastF1
        try:
            session = fastf1.get_session(season, grand_prix, session_type)
            session.load(telemetry=True, weather=False, messages=True)
            laps = session.laps
            if laps is not None and not laps.empty:
                _SESSION_MEMORY_CACHE[cache_key] = session
                return session
        except Exception as exc:
            logger.warning(
                "FastF1 не зміг завантажити %s %s %s (%s). Перехід на OpenF1 API.",
                season,
                grand_prix,
                session_type,
                exc,
            )

        # 2. Резервне джерело — відкритий OpenF1 API
        try:
            proxy = _load_openf1_session(season, grand_prix, session_type)
        except TelemetryDataError:
            raise
        except Exception as exc:
            raise TelemetryDataError(
                "Не вдалося завантажити дані обраної сесії. "
                "Перевірте інтернет-з'єднання або спробуйте інший етап."
            ) from exc

        _SESSION_MEMORY_CACHE[cache_key] = proxy
        return proxy


def get_session_drivers(session: Any) -> list[dict[str, str]]:
    """Повертає перелік пілотів, які мають принаймні одне коректне коло в сесії."""
    if isinstance(session, OpenF1SessionProxy):
        valid_nums = {
            str(lap["driver_number"])
            for lap in session.laps_raw
            if lap.get("lap_duration") is not None
        }
        return [d for d in session.drivers if d["driver_number"] in valid_nums]

    valid_laps = session.laps.dropna(subset=["LapTime"])
    driver_codes = list(dict.fromkeys(valid_laps["Driver"].dropna().astype(str).tolist()))
    if not driver_codes:
        raise TelemetryDataError("У цій сесії немає пілотів із завершеними колами.")

    drivers_info: list[dict[str, str]] = []
    results = getattr(session, "results", None)

    for code in driver_codes:
        last_name = code
        team_name = ""
        if results is not None and not results.empty and "Abbreviation" in results.columns:
            matched = results[results["Abbreviation"] == code]
            if not matched.empty:
                row = matched.iloc[0]
                last_name = str(row.get("LastName") or code)
                team_name = str(row.get("TeamName") or "")
        drivers_info.append({"code": code, "last_name": last_name, "team": team_name})

    drivers_info.sort(key=lambda item: item["code"])
    return drivers_info


def _resolve_driver_color(session: Any, driver_code: str, fallback_idx: int = 0) -> str:
    """Визначає офіційний колір команди пілота або повертає резервний контрастний колір."""
    results = getattr(session, "results", None)
    if results is not None and not results.empty and "Abbreviation" in results.columns:
        matched = results[results["Abbreviation"] == driver_code]
        if not matched.empty:
            raw_color = str(matched.iloc[0].get("TeamColor") or "").strip().lstrip("#")
            if len(raw_color) == 6:
                return f"#{raw_color}"
    return FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]


def _get_openf1_fastest_lap(
    session: OpenF1SessionProxy,
    driver_code: str,
    fallback_color_idx: int = 0,
) -> DriverLapBundle:
    """Завантажує телеметрію найшвидшого кола пілота з OpenF1 API."""
    drv = next((d for d in session.drivers if d["code"] == driver_code), None)
    if not drv:
        raise TelemetryDataError(f"Пілот {driver_code} не знайдений у сесії.")

    drv_num = int(drv["driver_number"])
    valid_laps = [
        lap
        for lap in session.laps_raw
        if lap.get("driver_number") == drv_num
        and lap.get("lap_duration") is not None
        and lap.get("date_start") is not None
        and not lap.get("is_pit_out_lap", False)
    ]
    if not valid_laps:
        raise TelemetryDataError(f"Пілот {driver_code} не має завершеного швидкого кола.")

    fastest = min(valid_laps, key=lambda item: float(item["lap_duration"]))
    lap_dur = float(fastest["lap_duration"])
    lap_num = int(fastest["lap_number"])

    start_ts = pd.to_datetime(fastest["date_start"], utc=True)
    end_ts = start_ts + pd.to_timedelta(lap_dur + 1.0, unit="s")
    date_gt = (start_ts - pd.to_timedelta(0.5, unit="s")).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    date_lt = end_ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    car_cache = session.cache_dir / f"car_{session.session_key}_{drv_num}_L{lap_num}.json"
    loc_cache = session.cache_dir / f"loc_{session.session_key}_{drv_num}_L{lap_num}.json"

    car_rows = _openf1_get(
        "car_data",
        {
            "session_key": session.session_key,
            "driver_number": drv_num,
            "date>": date_gt,
            "date<": date_lt,
        },
        cache_file=car_cache,
    )
    loc_rows = _openf1_get(
        "location",
        {
            "session_key": session.session_key,
            "driver_number": drv_num,
            "date>": date_gt,
            "date<": date_lt,
        },
        cache_file=loc_cache,
    )
    if len(car_rows) < 30:
        raise TelemetryDataError(f"Телеметрія кола пілота {driver_code} недоступна.")

    car_df = pd.DataFrame(car_rows)
    car_df["dt"] = pd.to_datetime(car_df["date"], utc=True)
    car_df = car_df[(car_df["dt"] >= start_ts) & (car_df["dt"] <= start_ts + pd.to_timedelta(lap_dur, unit="s"))]
    car_df = car_df.sort_values("dt").reset_index(drop=True)

    t_sec = (car_df["dt"] - start_ts).dt.total_seconds().to_numpy(dtype=float)
    speed_kmh = car_df["speed"].astype(float).to_numpy()
    speed_ms = speed_kmh / 3.6
    dt_step = np.diff(t_sec, prepend=t_sec[0])
    distance = np.cumsum(speed_ms * dt_step)

    x_arr = np.zeros_like(t_sec)
    y_arr = np.zeros_like(t_sec)
    if loc_rows:
        loc_df = pd.DataFrame(loc_rows)
        loc_df["dt"] = pd.to_datetime(loc_df["date"], utc=True)
        loc_df = loc_df.sort_values("dt")
        loc_t = (loc_df["dt"] - start_ts).dt.total_seconds().to_numpy(dtype=float)
        x_arr = np.interp(t_sec, loc_t, loc_df["x"].astype(float).to_numpy())
        y_arr = np.interp(t_sec, loc_t, loc_df["y"].astype(float).to_numpy())

    tel_df = pd.DataFrame(
        {
            "Time": pd.to_timedelta(t_sec, unit="s"),
            "Distance": distance,
            "Speed": speed_kmh,
            "Throttle": car_df["throttle"].astype(float).to_numpy(),
            "Brake": (car_df["brake"].astype(float).to_numpy() > 0).astype(float),
            "X": x_arr,
            "Y": y_arr,
        }
    )

    s1 = fastest.get("duration_sector_1")
    s2 = fastest.get("duration_sector_2")
    s3 = fastest.get("duration_sector_3")
    sector_times = (
        float(s1) if s1 is not None else None,
        float(s2) if s2 is not None else None,
        float(s3) if s3 is not None else None,
    )

    return DriverLapBundle(
        driver_code=driver_code,
        driver_name=drv["full_name"],
        team_name=drv["team"],
        team_color=drv.get("color") or FALLBACK_COLORS[fallback_color_idx % len(FALLBACK_COLORS)],
        lap_time_sec=lap_dur,
        compound="SOFT",
        lap_number=lap_num,
        sector_times=sector_times,
        telemetry=tel_df,
    )


def get_fastest_valid_lap(
    session: Any,
    driver_code: str,
    fallback_color_idx: int = 0,
) -> DriverLapBundle:
    """Отримує та перевіряє телеметрію найшвидшого коректного кола пілота.

    Відкидає неповні кола, кола під червоними прапорами та пошкоджені записи.
    """
    if isinstance(session, OpenF1SessionProxy):
        return _get_openf1_fastest_lap(session, driver_code, fallback_color_idx)

    driver_laps = session.laps.pick_drivers(driver_code)
    if driver_laps.empty:
        raise TelemetryDataError(
            f"Пілот {driver_code} не брав участі в сесії або зійшов на першому колі."
        )

    clean_laps = driver_laps.dropna(subset=["LapTime"])
    if "Deleted" in clean_laps.columns:
        non_deleted = clean_laps[clean_laps["Deleted"] != True]  # noqa: E712
        if not non_deleted.empty:
            clean_laps = non_deleted
    if "IsAccurate" in clean_laps.columns:
        accurate = clean_laps[clean_laps["IsAccurate"] == True]  # noqa: E712
        if not accurate.empty:
            clean_laps = accurate
    if "TrackStatus" in clean_laps.columns:
        no_red_flag = clean_laps[~clean_laps["TrackStatus"].astype(str).str.contains("5")]
        if not no_red_flag.empty:
            clean_laps = no_red_flag

    if clean_laps.empty:
        raise TelemetryDataError(
            f"Пілот {driver_code} не має жодного завершеного швидкого кола без порушень."
        )

    fastest_lap = clean_laps.pick_fastest()
    if fastest_lap is None or pd.isna(fastest_lap["LapTime"]):
        raise TelemetryDataError(
            f"Не вдалося визначити найшвидше коло для пілота {driver_code}."
        )

    try:
        telemetry = fastest_lap.get_telemetry()
    except Exception as exc:
        raise TelemetryDataError(
            f"Телеметрія найшвидшого кола пілота {driver_code} пошкоджена або недоступна."
        ) from exc

    required_cols = {"Time", "Distance", "Speed", "Throttle", "Brake"}
    if telemetry is None or telemetry.empty or not required_cols.issubset(telemetry.columns):
        raise TelemetryDataError(f"Неповний набір каналів датчиків для пілота {driver_code}.")
    if len(telemetry) < 30:
        raise TelemetryDataError(f"Занадто мало точок телеметрії для пілота {driver_code}.")

    driver_name = driver_code
    team_name = str(fastest_lap.get("Team") or "")
    results = getattr(session, "results", None)
    if results is not None and not results.empty and "Abbreviation" in results.columns:
        matched = results[results["Abbreviation"] == driver_code]
        if not matched.empty:
            row = matched.iloc[0]
            full_name = str(row.get("FullName") or "").strip()
            if full_name:
                driver_name = full_name
            if not team_name:
                team_name = str(row.get("TeamName") or "")

    def _td_to_sec(val: Any) -> float | None:
        return None if pd.isna(val) else float(val.total_seconds())

    sector_times = (
        _td_to_sec(fastest_lap.get("Sector1Time")),
        _td_to_sec(fastest_lap.get("Sector2Time")),
        _td_to_sec(fastest_lap.get("Sector3Time")),
    )

    return DriverLapBundle(
        driver_code=driver_code,
        driver_name=driver_name,
        team_name=team_name,
        team_color=_resolve_driver_color(session, driver_code, fallback_color_idx),
        lap_time_sec=float(fastest_lap["LapTime"].total_seconds()),
        compound=str(fastest_lap.get("Compound") or "N/A"),
        lap_number=int(fastest_lap["LapNumber"]) if not pd.isna(fastest_lap.get("LapNumber")) else 0,
        sector_times=sector_times,
        telemetry=telemetry.copy(),
    )
