"""Figures for the report. Colour follows the arm, in a fixed order, everywhere.

Palette: the first four categorical slots of a colour-blind-validated palette
(adjacent CVD and normal-vision checks pass on a light surface; two slots are
below 3:1 contrast, so every arm is also named on an axis or at its line's end),
plus a grey for the shipped default, which is the reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats
from .analysis import ARM_LABELS, RESULTS

FIGURES = RESULTS / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
COLOURS = {
    "grpo_nostd": "#2a78d6",  # slot 1, blue
    "grpo": "#eb6834",  # slot 2, orange
    "ppo": "#1baf7a",  # slot 3, aqua
    "mc_tuned": "#eda100",  # slot 4, yellow
    "mc_default": "#8a8983",  # reference grey
}
MAIN_ARMS = ("mc_default", "mc_tuned", "ppo", "grpo", "grpo_nostd")
SHORT = {
    "mc_default": "TRL", "mc_tuned": "TRL tuned", "ppo": "PPO",
    "grpo": "GRPO", "grpo_nostd": "GRPO mean-centred",
}
DATASET_TITLES = {
    "synth-route": "synthetic routing", "founders": "founders", "kickstarter": "kickstarter",
    "fakejobs": "job postings", "credit": "credit", "adult": "census income",
}
REGIME_TITLES = {"R1": "R1: TRL default rewards", "R2": "R2: cost-aware"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlesize": 9.5,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "lines.linewidth": 2.0, "legend.frameon": False,
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _reference_lines(ax: plt.Axes, horizontal: bool = True) -> None:
    line = ax.axhline if horizontal else ax.axvline
    line(0.0, color=MUTED, linewidth=0.8, zorder=1)
    line(1.0, color=MUTED, linewidth=0.8, zorder=1)


def main_results(frame: pd.DataFrame) -> Path:
    """Normalised test return of every main arm in every condition (mean and 95% CI)."""
    _style()
    datasets = list(dict.fromkeys(frame["dataset"]))
    fig, axes = plt.subplots(2, len(datasets), figsize=(2.15 * len(datasets) + 0.6, 5.6), sharey="row")
    for r, regime in enumerate(("R1", "R2")):
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]
            mine = frame[(frame["dataset"] == dataset) & (frame["regime"] == regime)]
            _reference_lines(ax)
            for x, arm in enumerate(MAIN_ARMS):
                values = mine.loc[mine["arm"] == arm, "norm"].to_numpy()
                if not len(values):
                    continue
                mean, low, high = stats.mean_ci(values)
                ax.plot([x, x], [low, high], color=COLOURS[arm], linewidth=2, solid_capstyle="round")
                ax.plot(x, mean, "o", color=COLOURS[arm], markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.set_xticks(range(len(MAIN_ARMS)))
            ax.set_xticklabels([SHORT[a] for a in MAIN_ARMS] if r == 1 else [], rotation=55, ha="right")
            ax.set_xlim(-0.6, len(MAIN_ARMS) - 0.4)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(DATASET_TITLES.get(dataset, dataset))
            if c == 0:
                ax.set_ylabel(f"{REGIME_TITLES[regime]}\nnormalised test return")
            if mine["degenerate"].any():
                ax.text(0.5, 0.04, "anchors too close:\nnot pooled", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7, color=MUTED)
    handles = [plt.Line2D([], [], marker="o", linestyle="-", color=COLOURS[a], label=ARM_LABELS[a],
                          markersize=7) for a in MAIN_ARMS]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.text(0.995, 0.005, "0 = random policy, 1 = best fixed set of slots; dot = mean over seeds, bar = 95% CI",
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return _save(fig, "fig1_main_results")


def hypothesis_forest(table: pd.DataFrame) -> Path:
    """Each confirmatory test: pooled paired difference with its 95% CI."""
    _style()
    table = table.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 0.48 * len(table) + 1.3))
    ax.axvline(0.0, color=MUTED, linewidth=0.8)
    for y, row in table.iterrows():
        ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=COLOURS["grpo_nostd"],
                linewidth=2, solid_capstyle="round")
        ax.plot(row["diff"], y, "o", color=COLOURS["grpo_nostd"], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        if row["kind"] == "noninferiority":
            ax.plot([-0.10, -0.10], [y - 0.32, y + 0.32], color=INK, linewidth=1.2)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([f"{r['hypothesis']}  {r['claim']}" for _, r in table.iterrows()], color=INK)
    ax.set_xlabel("pooled paired difference in normalised test return (first arm minus second)")
    ax.grid(axis="y", visible=False)
    span = ax.get_xlim()
    for y, row in table.iterrows():
        ax.text(span[1] + 0.02 * (span[1] - span[0]), y, row["verdict"], va="center", fontsize=8, color=INK)
    ax.set_title("Confirmatory tests (95% stratified bootstrap CI; short tick = non-inferiority margin)")
    return _save(fig, "fig2_hypotheses")


def learning_curves(curves: pd.DataFrame, anchors: pd.DataFrame, x: str = "fraction") -> Path:
    """Normalised test return against budget spent, pooled over conditions per regime."""
    _style()
    table = anchors.set_index("condition")
    curves = curves[curves["condition"].isin(table.index[~table["degenerate"]])].copy()
    curves["norm"] = (curves["test_ret"] - curves["condition"].map(table["random"])) / curves["condition"].map(table["gap"])
    grid = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, regime in zip(axes, ("R1", "R2"), strict=True):
        _reference_lines(ax)
        ends: list[tuple[float, str]] = []
        for arm in MAIN_ARMS:
            mine = curves[(curves["regime"] == regime) & (curves["arm"] == arm)]
            if mine.empty:
                continue
            per_run = []
            for _, run in mine.groupby(["condition", "seed"]):
                run = run.sort_values("compute")
                per_run.append(np.interp(grid, run["compute"] / run["budget"].iloc[0], run["norm"]))
            mean = np.mean(per_run, axis=0)
            ax.plot(grid, mean, color=COLOURS[arm], label=ARM_LABELS[arm])
            ends.append((float(mean[-1]), arm))
        placed: list[float] = []
        for value, arm in sorted(ends):
            y = value
            while any(abs(y - other) < 0.07 for other in placed):
                y += 0.035
            placed.append(y)
            ax.text(1.02, y, SHORT[arm], color=INK, fontsize=8, va="center")
            ax.plot([1.0, 1.015], [value, y], color=COLOURS[arm], linewidth=1)
        ax.set_xlim(0, 1.32)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("share of the compute budget (1 = one default TRL pass)")
        ax.set_title(REGIME_TITLES[regime])
    axes[0].set_ylabel("normalised test return (mean over conditions and seeds)")
    axes[0].legend(loc="lower right", fontsize=7.5)
    return _save(fig, "fig3_learning_curves")


def ladder(frame: pd.DataFrame, order: list[str], labels: dict[str, str], name: str, title: str,
           highlight: tuple[str, ...] = ("grpo", "grpo_nostd")) -> Path:
    """Pooled normalised return of each arm of a ladder (mean over conditions, 95% CI)."""
    _style()
    usable = frame[~frame["degenerate"]]
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(order) + 1.2))
    _reference_lines(ax, horizontal=False)
    for y, arm in enumerate(reversed(order)):
        mine = usable[usable["arm"] == arm]
        if mine.empty:
            continue
        by_condition = {str(c): g["norm"].to_numpy() for c, g in mine.groupby("condition")}
        mean = float(np.mean([v.mean() for v in by_condition.values()]))
        low, high = stats.stratified_bootstrap(by_condition)
        colour = COLOURS["grpo_nostd"] if arm in highlight else "#86b6ef"
        ax.plot([low, high], [y, y], color=colour, linewidth=2, solid_capstyle="round")
        ax.plot(mean, y, "o", color=colour, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(high + 0.02, y, f"{mean:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[a] for a in reversed(order)], color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("normalised test return, pooled over conditions (0 = random, 1 = best fixed set)")
    ax.set_title(title)
    return _save(fig, name)


def group_size(frame: pd.DataFrame) -> Path:
    """Pooled normalised return against group size, for both GRPO variants."""
    _style()
    usable = frame[~frame["degenerate"]].copy()
    usable["base"] = usable["arm"].str.rsplit("_G", n=1).str[0]
    usable["G"] = usable["arm"].str.rsplit("_G", n=1).str[1].astype(int)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _reference_lines(ax)
    for base in ("grpo", "grpo_nostd"):
        sizes, means, lows, highs = [], [], [], []
        for size, mine in usable[usable["base"] == base].groupby("G"):
            by_condition = {str(c): g["norm"].to_numpy() for c, g in mine.groupby("condition")}
            low, high = stats.stratified_bootstrap(by_condition)
            sizes.append(size)
            means.append(float(np.mean([v.mean() for v in by_condition.values()])))
            lows.append(low)
            highs.append(high)
        if not sizes:
            continue
        positions = np.log2(sizes)
        ax.fill_between(positions, lows, highs, color=COLOURS[base], alpha=0.15, linewidth=0)
        ax.plot(positions, means, "o-", color=COLOURS[base], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=ARM_LABELS[base])
        ax.text(positions[-1] + 0.12, means[-1], SHORT[base], va="center", fontsize=8, color=INK)
    ax.set_xticks(np.log2([2, 4, 8, 16, 32]))
    ax.set_xticklabels(["2", "4", "8", "16", "32"])
    ax.set_xlim(0.7, 6.9)
    ax.set_xlabel("group size G (episodes per sample), at a fixed number of episodes per update")
    ax.set_ylabel("normalised test return")
    ax.set_title("Group size")
    ax.legend(loc="center right", fontsize=8)
    return _save(fig, "fig5_group_size")


def behaviour(frame: pd.DataFrame) -> Path:
    """What the learned policies do: slots revealed, and how often they call positive."""
    _style()
    datasets = list(dict.fromkeys(frame["dataset"]))
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 5.6), sharex="col")
    columns = (("test_n_slots", "slots revealed per sample"), ("test_called", "share of samples called positive"))
    width = 0.15
    for r, regime in enumerate(("R1", "R2")):
        for c, (column, label) in enumerate(columns):
            ax = axes[r, c]
            for k, arm in enumerate(MAIN_ARMS):
                means = [
                    frame[(frame["dataset"] == d) & (frame["regime"] == regime) & (frame["arm"] == arm)][column].mean()
                    for d in datasets
                ]
                ax.bar(np.arange(len(datasets)) + (k - 2) * width, means, width=width - 0.02,
                       color=COLOURS[arm], label=ARM_LABELS[arm])
            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels([DATASET_TITLES.get(d, d) for d in datasets], rotation=25, ha="right")
            ax.grid(axis="x", visible=False)
            ax.set_ylabel(label)
            ax.set_title(REGIME_TITLES[regime])
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "fig6_behaviour")
