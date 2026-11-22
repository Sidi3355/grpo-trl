"""Inline SVG charts for the HTML report: colours come from CSS tokens, so they follow the theme.

Every mark carries a ``<title>`` (the hover tooltip), every series is named in a
legend and on an axis or a direct label, and all positions come from one scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import numpy as np
import pandas as pd

from . import stats

SERIES = {  # arm -> CSS class; the class sets the colour from a token
    "grpo_nostd": "s1",
    "grpo": "s2",
    "ppo": "s3",
    "mc_tuned": "s4",
    "mc_default": "s5",
}
NAMES = {
    "mc_default": "TRL shipped (defaults)",
    "mc_tuned": "TRL shipped (tuned)",
    "ppo": "PPO",
    "grpo": "GRPO (canonical)",
    "grpo_nostd": "GRPO (mean-centred)",
}
DATASETS = {
    "synth-route": "synthetic routing",
    "founders": "founders",
    "kickstarter": "kickstarter",
    "fakejobs": "job postings",
    "credit": "credit",
    "adult": "census income",
}


class Scale:
    """A linear map from data to pixels."""

    def __init__(self, low: float, high: float, start: float, end: float) -> None:
        self.low, self.high, self.start, self.end = low, high, start, end

    def __call__(self, value: float) -> float:
        share = (value - self.low) / (self.high - self.low)
        return self.start + share * (self.end - self.start)


def _ticks(low: float, high: float, step: float) -> list[float]:
    first = np.ceil(low / step - 1e-9) * step
    return [round(float(v), 6) for v in np.arange(first, high + 1e-9, step)]


def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) >= 1e-9 else "0"


def _legend(arms: Sequence[str], x: float, y: float, width: float) -> tuple[str, float]:
    """Legend rows that wrap within ``width``; returns the markup and the height used."""
    out, cx, cy = [], x, y
    for arm in arms:
        label = NAMES[arm]
        needed = 22 + 6.4 * len(label) + 14
        if cx + needed > x + width and cx > x:
            cx, cy = x, cy + 18
        out.append(f'<circle class="{SERIES[arm]} mark" cx="{cx + 6:.1f}" cy="{cy:.1f}" r="5"/>')
        out.append(f'<text class="lbl" x="{cx + 16:.1f}" y="{cy + 4:.1f}">{escape(label)}</text>')
        cx += needed
    return "".join(out), cy - y + 18


def dot_plot(
    frame: pd.DataFrame, arms: Sequence[str], regime: str, title: str, width: int = 720
) -> str:
    """One row per dataset, one dot per arm: mean normalised test return with a 95% CI."""
    mine = frame[frame["regime"] == regime]
    datasets = list(dict.fromkeys(mine["dataset"]))
    rows = []
    for dataset in datasets:
        for arm in arms:
            values = mine.loc[
                (mine["dataset"] == dataset) & (mine["arm"] == arm), "norm"
            ].to_numpy()
            if len(values):
                rows.append((dataset, arm, *stats.mean_ci(values)))
    low = min(min(r[3] for r in rows), 0.0) - 0.1
    high = max(max(r[4] for r in rows), 1.0) + 0.1
    left, right, band = 118, 16, 74
    legend, legend_height = _legend(arms, left, 34, width - left - right)
    top = 34 + legend_height + 6
    height = top + band * len(datasets) + 34
    x = Scale(low, high, left, width - right)
    step = 0.5 if high - low <= 4 else 1.0
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<text class="ttl" x="0" y="14">{escape(title)}</text>',
        legend,
    ]
    for tick in _ticks(low, high, step):
        cls = "ref" if tick in (0.0, 1.0) else "grid"
        parts.append(
            f'<line class="{cls}" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top}" y2="{height - 30}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x(tick):.1f}" y="{height - 16}" text-anchor="middle">{_fmt(tick)}</text>'
        )
    parts.append(
        f'<text class="tick" x="{x(0):.1f}" y="{height - 3}" text-anchor="middle">random</text>'
    )
    parts.append(
        f'<text class="tick" x="{x(1):.1f}" y="{height - 3}" text-anchor="middle">best fixed set</text>'
    )
    for i, dataset in enumerate(datasets):
        y0 = top + band * i
        degenerate = bool(mine.loc[mine["dataset"] == dataset, "degenerate"].any())
        parts.append(
            f'<text class="lbl" x="0" y="{y0 + band / 2 + 4:.1f}">{escape(DATASETS.get(dataset, dataset))}</text>'
        )
        if degenerate:
            parts.append(f'<text class="tick" x="0" y="{y0 + band / 2 + 18:.1f}">not pooled</text>')
        if i:
            parts.append(f'<line class="grid" x1="0" x2="{width}" y1="{y0}" y2="{y0}"/>')
        for k, arm in enumerate(arms):
            found = [r for r in rows if r[0] == dataset and r[1] == arm]
            if not found:
                continue
            _, _, mean, lo, hi = found[0]
            y = y0 + 12 + k * (band - 24) / max(len(arms) - 1, 1)
            tip = f"{NAMES[arm]}, {DATASETS.get(dataset, dataset)}: {mean:.2f} (95% CI {lo:.2f} to {hi:.2f})"
            parts.append(
                f"<g><title>{escape(tip)}</title>"
                f'<line class="{SERIES[arm]} bar" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
                f'<circle class="{SERIES[arm]} mark" cx="{x(mean):.1f}" cy="{y:.1f}" r="5"/>'
                f'<rect class="hit" x="{x(lo) - 6:.1f}" y="{y - 7:.1f}" width="{x(hi) - x(lo) + 12:.1f}" height="14"/></g>'
            )
    parts.append("</svg>")
    return "".join(parts)


def forest(
    table: pd.DataFrame,
    title: str,
    labels: Sequence[str],
    margin: float | None = None,
    width: int = 720,
    unit: str = "difference in normalised test return",
) -> str:
    """Rows of estimates with 95% CIs around zero. ``table`` has diff, ci_low, ci_high, kind."""
    left, right, band = 300, 20, 30
    low = min(float(table["ci_low"].min()), -0.15) - 0.05
    high = max(float(table["ci_high"].max()), 0.15) + 0.05
    top, height = 30, 30 + band * len(table) + 40
    x = Scale(low, high, left, width - right)
    span = high - low
    step = 0.1 if span <= 0.8 else 0.25 if span <= 2 else 0.5
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<text class="ttl" x="0" y="14">{escape(title)}</text>',
    ]
    for tick in _ticks(low, high, step):
        cls = "ref" if tick == 0.0 else "grid"
        parts.append(
            f'<line class="{cls}" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top}" y2="{height - 36}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x(tick):.1f}" y="{height - 22}" text-anchor="middle">{_fmt(tick)}</text>'
        )
    parts.append(
        f'<text class="tick" x="{(left + width - right) / 2:.1f}" y="{height - 6}" text-anchor="middle">{escape(unit)}</text>'
    )
    for i, (label, (_, row)) in enumerate(zip(labels, table.iterrows(), strict=True)):
        y = top + band * i + band / 2
        tip = f"{label}: {row['diff']:+.3f} (95% CI {row['ci_low']:+.3f} to {row['ci_high']:+.3f})"
        parts.append(f'<text class="lbl" x="0" y="{y + 4:.1f}">{escape(label)}</text>')
        if margin is not None and row.get("kind") == "noninferiority":
            parts.append(
                f'<line class="mar" x1="{x(-margin):.1f}" x2="{x(-margin):.1f}" y1="{y - 9:.1f}" y2="{y + 9:.1f}"/>'
            )
        parts.append(
            f"<g><title>{escape(tip)}</title>"
            f'<line class="s1 bar" x1="{x(row["ci_low"]):.1f}" x2="{x(row["ci_high"]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<circle class="s1 mark" cx="{x(row["diff"]):.1f}" cy="{y:.1f}" r="5"/>'
            f'<rect class="hit" x="{x(row["ci_low"]) - 6:.1f}" y="{y - 9:.1f}" width="{x(row["ci_high"]) - x(row["ci_low"]) + 12:.1f}" height="18"/></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def levels(
    rows: Sequence[tuple[str, float, float, float, bool]],
    title: str,
    width: int = 720,
    unit: str = "normalised test return, pooled over conditions",
) -> str:
    """Rows of (label, mean, low, high, emphasised): a ladder of pooled scores."""
    left, right, band = 330, 44, 30
    low = min(min(r[2] for r in rows), 0.0) - 0.1
    high = max(max(r[3] for r in rows), 1.0) + 0.1
    top, height = 30, 30 + band * len(rows) + 52
    x = Scale(low, high, left, width - right)
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<text class="ttl" x="0" y="14">{escape(title)}</text>',
    ]
    for tick in _ticks(low, high, 0.5):
        cls = "ref" if tick in (0.0, 1.0) else "grid"
        parts.append(
            f'<line class="{cls}" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top}" y2="{height - 48}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x(tick):.1f}" y="{height - 34}" text-anchor="middle">{_fmt(tick)}</text>'
        )
    parts.append(
        f'<text class="tick" x="{x(0):.1f}" y="{height - 21}" text-anchor="middle">random</text>'
    )
    parts.append(
        f'<text class="tick" x="{x(1):.1f}" y="{height - 21}" text-anchor="middle">best fixed set</text>'
    )
    parts.append(
        f'<text class="tick" x="{(left + width - right) / 2:.1f}" y="{height - 4}" text-anchor="middle">{escape(unit)}</text>'
    )
    for i, (label, mean, lo, hi, strong) in enumerate(rows):
        y = top + band * i + band / 2
        series = "s1" if strong else "s6"
        tip = f"{label}: {mean:.2f} (95% CI {lo:.2f} to {hi:.2f})"
        parts.append(f'<text class="lbl" x="0" y="{y + 4:.1f}">{escape(label)}</text>')
        parts.append(
            f"<g><title>{escape(tip)}</title>"
            f'<line class="{series} bar" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<circle class="{series} mark" cx="{x(mean):.1f}" cy="{y:.1f}" r="5"/>'
            f'<rect class="hit" x="{x(lo) - 6:.1f}" y="{y - 9:.1f}" width="{x(hi) - x(lo) + 12:.1f}" height="18"/></g>'
        )
        parts.append(f'<text class="val" x="{x(hi) + 8:.1f}" y="{y + 4:.1f}">{mean:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def curves(
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    arms: Sequence[str],
    title: str,
    width: int = 720,
    height: int = 300,
    x_label: str = "share of the compute budget",
) -> str:
    """Mean normalised test return against budget share, one line per arm, labelled at its end."""
    left, right, top, bottom = 40, 150, 30, 46
    values = [v for xs, ys in series.values() for v in ys]
    low, high = min(min(values), 0.0) - 0.05, max(max(values), 1.0) + 0.1
    x, y = Scale(0.0, 1.0, left, width - right), Scale(low, high, height - bottom, top)
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<text class="ttl" x="0" y="14">{escape(title)}</text>',
    ]
    for tick in _ticks(low, high, 0.5):
        cls = "ref" if tick in (0.0, 1.0) else "grid"
        parts.append(
            f'<line class="{cls}" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 6}" y="{y(tick) + 3.5:.1f}" text-anchor="end">{_fmt(tick)}</text>'
        )
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<text class="tick" x="{x(tick):.1f}" y="{height - bottom + 16}" text-anchor="middle">{_fmt(tick)}</text>'
        )
    parts.append(
        f'<text class="tick" x="{(left + width - right) / 2:.1f}" y="{height - 8}" text-anchor="middle">{escape(x_label)}</text>'
    )
    ends: list[tuple[float, str]] = []
    for arm in arms:
        if arm not in series:
            continue
        xs, ys = series[arm]
        points = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs, ys, strict=True))
        parts.append(
            f'<polyline class="{SERIES[arm]} line" points="{points}"><title>{escape(NAMES[arm])}</title></polyline>'
        )
        for a, b in zip(xs, ys, strict=True):
            parts.append(
                f'<circle class="hit" cx="{x(a):.1f}" cy="{y(b):.1f}" r="7"><title>{escape(NAMES[arm])} at {a:.0%} of the budget: {b:.2f}</title></circle>'
            )
        ends.append((y(ys[-1]), arm))
    placed: list[float] = []
    for position, arm in sorted(ends):
        label_y = position
        while any(abs(label_y - other) < 13 for other in placed):
            label_y += 4
        placed.append(label_y)
        parts.append(
            f'<line class="{SERIES[arm]} lead" x1="{x(1.0):.1f}" x2="{x(1.0) + 8:.1f}" y1="{position:.1f}" y2="{label_y:.1f}"/>'
        )
        parts.append(
            f'<circle class="{SERIES[arm]} mark" cx="{x(1.0):.1f}" cy="{position:.1f}" r="4"/>'
        )
        parts.append(
            f'<text class="lbl" x="{x(1.0) + 12:.1f}" y="{label_y + 4:.1f}">{escape(NAMES[arm])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


CSS = """
.chart{width:100%;height:auto;display:block;font-family:var(--font-data)}
.chart .ttl{font:600 13px var(--font-text);fill:var(--ink)}
.chart .lbl{font-size:11.5px;fill:var(--ink)}
.chart .val{font-size:11px;fill:var(--ink-2);font-variant-numeric:tabular-nums}
.chart .tick{font-size:10.5px;fill:var(--ink-3);font-variant-numeric:tabular-nums}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .ref{stroke:var(--ink-3);stroke-width:1}
.chart .mar{stroke:var(--ink);stroke-width:1.5}
.chart .bar{stroke-width:2;stroke-linecap:round}
.chart .line{fill:none;stroke-width:2;stroke-linejoin:round}
.chart .lead{stroke-width:1}
.chart .mark{stroke:var(--surface);stroke-width:2}
.chart .hit{fill:transparent;stroke:none}
.chart .s1{stroke:var(--s1)}.chart circle.s1{fill:var(--s1)}
.chart .s2{stroke:var(--s2)}.chart circle.s2{fill:var(--s2)}
.chart .s3{stroke:var(--s3)}.chart circle.s3{fill:var(--s3)}
.chart .s4{stroke:var(--s4)}.chart circle.s4{fill:var(--s4)}
.chart .s5{stroke:var(--s5)}.chart circle.s5{fill:var(--s5)}
.chart .s6{stroke:var(--s6)}.chart circle.s6{fill:var(--s6)}
.chart circle.mark{stroke:var(--surface)}
"""
