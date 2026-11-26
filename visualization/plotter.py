"""Візуалізація телеметрії Формули-1 та побудова схем трас у буфер io.BytesIO."""

from __future__ import annotations

import io
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

from core.physics import (
    BrakingAnalysis,
    ComparisonResult,
    ProcessedTelemetry,
    format_lap_time,
)

# Темна контрастна тема, оптимізована для перегляду на екрані смартфона
BG_FIG = "#0E1117"
BG_AX = "#161B26"
GRID_COLOR = "#2A3244"
TEXT_MAIN = "#F3F4F6"
TEXT_MUTED = "#9CA3AF"
COLOR_BRAKE_OPT = "#FACC15"
COLOR_BRAKE_FACT = "#F97316"
COLOR_SECTOR = "#64748B"


def _ensure_distinct_colors(color1: str, color2: str) -> tuple[str, str]:
    """Гарантує контрастність кольорів, якщо обрано двох пілотів з однієї команди."""
    c1 = color1.upper()
    c2 = color2.upper()
    if c1 == c2:
        return c1, "#FACC15" if c1 != "#FACC15" else "#00E5FF"
    return c1, c2


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    """Застосовує єдиний читабельний стиль оформлення до графіка."""
    ax.set_facecolor(BG_AX)
    ax.set_ylabel(ylabel, color=TEXT_MAIN, fontsize=12, fontweight="bold")
    ax.tick_params(colors=TEXT_MAIN, labelsize=10)
    ax.grid(True, linestyle="--", linewidth=0.6, color=GRID_COLOR, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color("#334155")


def _draw_sector_lines(
    axes: list[plt.Axes],
    sector_boundaries: tuple[float, float],
) -> None:
    """Наносить вертикальні межі секторів S1/S2/S3 на всі підграфіки."""
    s1, s2 = sector_boundaries
    for idx, ax in enumerate(axes):
        ax.axvline(s1, color=COLOR_SECTOR, linestyle=":", linewidth=1.2, alpha=0.9)
        ax.axvline(s2, color=COLOR_SECTOR, linestyle=":", linewidth=1.2, alpha=0.9)
        if idx == 0:
            ymin, ymax = ax.get_ylim()
            y_label = ymax - (ymax - ymin) * 0.08
            ax.text(s1 + 25, y_label, "S2", color=TEXT_MUTED, fontsize=10, fontweight="bold")
            ax.text(s2 + 25, y_label, "S3", color=TEXT_MUTED, fontsize=10, fontweight="bold")


def _fig_to_bytes(fig: plt.Figure, dpi: int = 180) -> io.BytesIO:
    """Зберігає графік Matplotlib у PNG-буфер оперативної пам'яті без запису на диск."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_single_driver_telemetry(
    telemetry: ProcessedTelemetry,
    braking: BrakingAnalysis,
    driver_code: str,
    driver_color: str,
    title: str,
) -> io.BytesIO:
    """Будує 4-рівневий графік телеметрії для одного пілота."""
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.1, 0.85, 1.35]},
    )
    fig.patch.set_facecolor(BG_FIG)

    s = telemetry.distance

    # 1. Швидкість (км/год)
    axes[0].plot(s, telemetry.speed_kmh, color=driver_color, linewidth=2.2, label=f"{driver_code} (швидкість)")
    axes[0].axvline(
        braking.s_opt,
        color=COLOR_BRAKE_OPT,
        linestyle="--",
        linewidth=1.6,
        label=f"Опт. гальмування ({braking.s_opt:.0f} м, μ={braking.mu:.1f})",
    )
    axes[0].axvline(
        braking.s_fact,
        color=COLOR_BRAKE_FACT,
        linestyle="-.",
        linewidth=1.3,
        alpha=0.9,
        label=f"Факт. гальмування ({braking.s_fact:.0f} м)",
    )
    _style_axis(axes[0], "Швидкість,\nкм/год")
    axes[0].legend(
        loc="lower right",
        facecolor=BG_FIG,
        edgecolor="#334155",
        labelcolor=TEXT_MAIN,
        fontsize=10,
    )

    # 2. Педаль газу (%)
    axes[1].plot(s, telemetry.throttle, color="#22C55E", linewidth=1.8)
    axes[1].fill_between(s, 0, telemetry.throttle, color="#22C55E", alpha=0.15)
    axes[1].set_ylim(-5, 108)
    _style_axis(axes[1], "Газ, %")

    # 3. Індикатор гальма
    axes[2].step(s, telemetry.brake, where="mid", color="#EF4444", linewidth=1.8)
    axes[2].fill_between(s, 0, telemetry.brake, step="mid", color="#EF4444", alpha=0.25)
    axes[2].set_ylim(-0.1, 1.2)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Відпущено", "Натиснуто"])
    _style_axis(axes[2], "Гальмо")

    # 4. Поздовжнє перевантаження (g)
    axes[3].axhline(0, color=TEXT_MUTED, linewidth=0.9, alpha=0.7)
    axes[3].plot(s, telemetry.g_force, color="#38BDF8", linewidth=1.6)
    axes[3].fill_between(
        s, 0, telemetry.g_force, where=telemetry.g_force < 0, color="#EF4444", alpha=0.22
    )
    axes[3].fill_between(
        s, 0, telemetry.g_force, where=telemetry.g_force >= 0, color="#22C55E", alpha=0.22
    )
    _style_axis(axes[3], "Перевант., g")
    axes[3].set_xlabel("Пройдена дистанція s, м", color=TEXT_MAIN, fontsize=12, fontweight="bold")

    _draw_sector_lines(list(axes), telemetry.sector_boundaries_m)

    fig.suptitle(
        f"{title}\n{driver_code} — час кола {format_lap_time(telemetry.official_lap_time_sec)} "
        f"| Vmax: {telemetry.max_speed_kmh:.1f} км/год | Макс. гальмування: {telemetry.max_brake_g:.2f} g",
        color=TEXT_MAIN,
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _fig_to_bytes(fig)


def plot_drivers_comparison(
    comparison: ComparisonResult,
    braking: BrakingAnalysis,
    driver1_code: str,
    driver2_code: str,
    color1: str,
    color2: str,
    title: str,
) -> io.BytesIO:
    """Будує 4-рівневий графік порівняння телеметрії двох пілотів."""
    c1, c2 = _ensure_distinct_colors(color1, color2)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.5, 11.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.1, 0.9, 1.4]},
    )
    fig.patch.set_facecolor(BG_FIG)

    s = comparison.distance
    d1 = comparison.driver1
    d2 = comparison.driver2

    # 1. Швидкість (км/год)
    axes[0].plot(
        s,
        d1.speed_kmh,
        color=c1,
        linewidth=2.2,
        label=f"{driver1_code} ({format_lap_time(d1.official_lap_time_sec)})",
    )
    axes[0].plot(
        s,
        d2.speed_kmh,
        color=c2,
        linewidth=2.0,
        linestyle="--",
        label=f"{driver2_code} ({format_lap_time(d2.official_lap_time_sec)})",
    )
    axes[0].axvline(
        braking.s_opt,
        color=COLOR_BRAKE_OPT,
        linestyle=":",
        linewidth=1.8,
        label=f"Опт. точка гальмування ({braking.s_opt:.0f} м)",
    )
    _style_axis(axes[0], "Швидкість,\nкм/год")
    axes[0].legend(
        loc="lower right",
        facecolor=BG_FIG,
        edgecolor="#334155",
        labelcolor=TEXT_MAIN,
        fontsize=10.5,
    )

    # 2. Педаль газу (%)
    axes[1].plot(s, d1.throttle, color=c1, linewidth=1.8, label=driver1_code)
    axes[1].plot(s, d2.throttle, color=c2, linewidth=1.6, linestyle="--", label=driver2_code)
    axes[1].set_ylim(-5, 108)
    _style_axis(axes[1], "Газ, %")

    # 3. Індикатор гальмування
    axes[2].step(s, d1.brake, where="mid", color=c1, linewidth=1.8, label=driver1_code)
    axes[2].step(s, d2.brake * 0.92, where="mid", color=c2, linewidth=1.6, linestyle="--", label=driver2_code)
    axes[2].set_ylim(-0.1, 1.2)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["0", "1"])
    _style_axis(axes[2], "Гальмо")

    # 4. Дельта часу Δt(s) = t2(s) - t1(s)
    axes[3].axhline(0.0, color=TEXT_MUTED, linewidth=1.0, linestyle="-")
    axes[3].plot(s, comparison.delta_t, color="#E2E8F0", linewidth=2.0)
    axes[3].fill_between(
        s, 0, comparison.delta_t, where=comparison.delta_t >= 0, color=c1, alpha=0.25
    )
    axes[3].fill_between(
        s, 0, comparison.delta_t, where=comparison.delta_t < 0, color=c2, alpha=0.25
    )
    _style_axis(axes[3], f"Δt = t({driver2_code})−t({driver1_code}),\nс")
    axes[3].set_xlabel("Пройдена дистанція s, м", color=TEXT_MAIN, fontsize=12, fontweight="bold")

    _draw_sector_lines(list(axes), d1.sector_boundaries_m)

    fig.suptitle(
        f"{title}\nПорівняння телеметрії: {driver1_code} vs {driver2_code} (крок Δs = 1 м)",
        color=TEXT_MAIN,
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    return _fig_to_bytes(fig)


def plot_track_dominance_map(
    comparison: ComparisonResult,
    braking: BrakingAnalysis,
    driver1_code: str,
    driver2_code: str,
    color1: str,
    color2: str,
    title: str,
) -> io.BytesIO:
    """Малює схему траси за координатами (X, Y) з 50-метровими ділянками переваги пілотів."""
    c1, c2 = _ensure_distinct_colors(color1, color2)

    x = comparison.driver1.x
    y = comparison.driver1.y

    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_FIG)

    segments_xy = []
    segment_colors = []
    d1_wins = 0
    d2_wins = 0

    for seg in comparison.segment_dominance:
        s_idx = int(seg["start_idx"])
        e_idx = int(seg["end_idx"])
        pts = np.column_stack([x[s_idx:e_idx], y[s_idx:e_idx]])
        if len(pts) < 2:
            continue
        segments_xy.append(pts)
        if seg["faster_driver"] == 1:
            segment_colors.append(c1)
            d1_wins += 1
        else:
            segment_colors.append(c2)
            d2_wins += 1

    # Тонка підкладка траси для візуальної цілісності
    ax.plot(x, y, color="#1F2937", linewidth=9.5, solid_capstyle="round", zorder=1)
    lc = LineCollection(
        segments_xy,
        colors=segment_colors,
        linewidths=5.5,
        capstyle="round",
        joinstyle="round",
        zorder=2,
    )
    ax.add_collection(lc)

    # Позначка лінії старту/фінішу
    ax.scatter(
        [x[0]],
        [y[0]],
        color="#FFFFFF",
        edgecolors="#000000",
        s=110,
        zorder=5,
        label="Старт / Фініш",
    )

    # Позначка розрахованої оптимальної точки початку гальмування
    ax.scatter(
        [braking.opt_x],
        [braking.opt_y],
        color=COLOR_BRAKE_OPT,
        edgecolors="#000000",
        marker="D",
        s=125,
        zorder=6,
        label=f"Опт. точка гальмування (s = {braking.s_opt:.0f} м, μ = {braking.mu:.1f})",
    )
    ax.scatter(
        [braking.apex_x],
        [braking.apex_y],
        color="#EF4444",
        edgecolors="#FFFFFF",
        marker="o",
        s=90,
        zorder=6,
        label=f"Апекс ключового повороту (v = {braking.v_apex_kmh:.0f} км/год)",
    )

    total_seg = max(1, d1_wins + d2_wins)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=c1,
            lw=5,
            label=f"{driver1_code} швидший ({d1_wins / total_seg * 100:.1f}% траси)",
        ),
        Line2D(
            [0],
            [0],
            color=c2,
            lw=5,
            label=f"{driver2_code} швидший ({d2_wins / total_seg * 100:.1f}% траси)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=COLOR_BRAKE_OPT,
            markeredgecolor="#000000",
            markersize=9,
            linestyle="None",
            label=f"Опт. гальмування ({braking.s_opt:.0f} м, μ={braking.mu:.1f})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#EF4444",
            markersize=8,
            linestyle="None",
            label=f"Апекс повороту ({braking.v_apex_kmh:.0f} км/год)",
        ),
    ]

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.axis("off")
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        facecolor=BG_AX,
        edgecolor="#334155",
        labelcolor=TEXT_MAIN,
        fontsize=10.5,
        frameon=True,
    )
    ax.set_title(
        f"{title}\nСхема траси: ділянки переваги пілотів (відрізки по 50 м)",
        color=TEXT_MAIN,
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_braking_zone_detail(
    telemetry: ProcessedTelemetry,
    braking: BrakingAnalysis,
    driver_code: str,
    driver_color: str,
    title: str,
) -> io.BytesIO:
    """Будує деталізований графік зони гальмування з фактичною та оптимальною точками."""
    margin_before = 120.0
    margin_after = 80.0
    s_min = max(0.0, min(braking.s_fact, braking.s_opt) - margin_before)
    s_max = min(telemetry.lap_length_m, braking.s_apex + margin_after)
    mask = (telemetry.distance >= s_min) & (telemetry.distance <= s_max)

    s = telemetry.distance[mask]
    v = telemetry.speed_kmh[mask]
    g = telemetry.g_force[mask]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.2]}
    )
    fig.patch.set_facecolor(BG_FIG)

    ax1.plot(s, v, color=driver_color, linewidth=2.5, label=f"Фактична швидкість ({driver_code})")

    # Теоретична крива рівносповільненого гальмування від s_opt до s_apex
    s_model = np.linspace(braking.s_opt, braking.s_apex, 100)
    v0_ms = braking.v_start_kmh / 3.6
    v_model_ms = np.sqrt(np.maximum(0.0, v0_ms**2 - 2.0 * braking.a_brake * (s_model - braking.s_opt)))
    ax1.plot(
        s_model,
        v_model_ms * 3.6,
        color=COLOR_BRAKE_OPT,
        linestyle="--",
        linewidth=2.2,
        label=f"Модельне гальмування (μ = {braking.mu:.2f}, a = {braking.a_brake:.1f} м/с²)",
    )

    ax1.axvline(braking.s_fact, color=COLOR_BRAKE_FACT, linestyle="-.", linewidth=1.8, label=f"s_факт = {braking.s_fact:.1f} м")
    ax1.axvline(braking.s_opt, color=COLOR_BRAKE_OPT, linestyle="--", linewidth=1.8, label=f"s_опт = {braking.s_opt:.1f} м")
    ax1.axvline(braking.s_apex, color="#EF4444", linestyle=":", linewidth=1.8, label=f"Апекс = {braking.s_apex:.1f} м")
    _style_axis(ax1, "Швидкість,\nкм/год")
    ax1.legend(
        loc="lower left",
        facecolor=BG_FIG,
        edgecolor="#334155",
        labelcolor=TEXT_MAIN,
        fontsize=9.8,
    )

    ax2.axhline(-braking.mu, color=COLOR_BRAKE_OPT, linestyle="--", linewidth=1.4, label=f"Розрах. сповільнення (−{braking.mu:.2f} g)")
    ax2.plot(s, g, color="#38BDF8", linewidth=2.0, label="Фактичне перевантаження (g)")
    ax2.fill_between(s, 0, g, where=g < 0, color="#EF4444", alpha=0.22)
    _style_axis(ax2, "Перевант., g")
    ax2.set_xlabel("Дистанція на колі s, м", color=TEXT_MAIN, fontsize=12, fontweight="bold")
    ax2.legend(
        loc="lower left",
        facecolor=BG_FIG,
        edgecolor="#334155",
        labelcolor=TEXT_MAIN,
        fontsize=9.8,
    )

    delta_sign = "+" if braking.delta_s >= 0 else ""
    fig.suptitle(
        f"{title} — Розрахунок точки гальмування ({driver_code})\n"
        f"s_гальм = {braking.s_brake:.1f} м | Запас Δs = {delta_sign}{braking.delta_s:.1f} м",
        color=TEXT_MAIN,
        fontsize=13.2,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _fig_to_bytes(fig)
