"""Assemble the shareable HTML report from a template and generated fragments.

    python -m grpo_trl.page

Reads ``docs/report_template.html``, replaces every ``{{name}}`` with a chart or
table built from ``results/``, and writes ``results/report.html``. The prose
lives in the template; every figure and table is generated here.
"""

from __future__ import annotations

import json
import re
from html import escape

import numpy as np
import pandas as pd

from . import analysis, stats, svg
from .analysis import add_normalised, anchor_table, load_curves, load_runs
from .report import LADDER, LADDER_LABELS, MAIN
from .run import RESULTS, ROOT

TEMPLATE = ROOT / "docs" / "report_template.html"
OUTPUT = RESULTS / "report.html"
ARM_ORDER = ["grpo_nostd", "grpo", "ppo", "mc_tuned", "mc_default"]


def table(frame: pd.DataFrame, numeric: tuple[str, ...] = (), digits: int = 2) -> str:
    """An HTML table; ``numeric`` columns are right-aligned with tabular figures."""
    head = "".join(
        f'<th{" class=num" if c in numeric else ""}>{escape(str(c))}</th>' for c in frame.columns
    )
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                text = "" if np.isnan(value) else f"{value:.{digits}f}"
            else:
                text = str(value)
            cells.append(f'<td{" class=num" if column in numeric else ""}>{escape(text)}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def curve_series(curves: pd.DataFrame, anchors: pd.DataFrame, regime: str) -> dict[str, tuple[list[float], list[float]]]:
    usable = anchors.set_index("condition")
    mine = curves[(curves["regime"] == regime) & curves["condition"].isin(usable.index[~usable["degenerate"]])].copy()
    mine["norm"] = (mine["test_ret"] - mine["condition"].map(usable["random"])) / mine["condition"].map(usable["gap"])
    grid = np.linspace(0, 1, 11)
    out = {}
    for arm in MAIN:
        runs = [
            np.interp(grid, run.sort_values("compute")["compute"] / run["budget"].iloc[0], run.sort_values("compute")["norm"])
            for _, run in mine[mine["arm"] == arm].groupby(["condition", "seed"])
        ]
        if runs:
            out[arm] = ([float(g) for g in grid], [float(v) for v in np.mean(runs, axis=0)])
    return out


def fragments() -> dict[str, str]:
    parts: dict[str, str] = {"chart_css": svg.CSS}
    e1 = add_normalised(load_runs("e1"))
    anchors = anchor_table(e1)
    curves = load_curves("e1")
    parts["n_seeds"] = str(e1["seed"].nunique())
    parts["n_runs_e1"] = str(len(e1))
    for regime in ("R1", "R2"):
        title = {"R1": "R1, TRL's default rewards", "R2": "R2, cost-aware rewards"}[regime]
        parts[f"chart_main_{regime}"] = svg.dot_plot(
            e1, ARM_ORDER, regime, f"{title}: normalised test return by dataset (mean and 95% CI over seeds)"
        )
        parts[f"chart_curves_{regime}"] = svg.curves(
            curve_series(curves, anchors, regime), ARM_ORDER,
            f"{title}: mean normalised test return as the budget is spent",
            x_label="share of the compute budget (1 = one default TRL pass over the training set)",
        )
    aggregate = analysis.aggregate(e1, ARM_ORDER)
    pooled = pd.DataFrame({
        "Optimiser": aggregate["arm"].map(svg.NAMES),
        "Normalised return": [f"{m:.2f}" for m in aggregate["mean_norm"]],
        "95% CI": [f"{a:.2f} to {b:.2f}" for a, b in zip(aggregate["low"], aggregate["high"], strict=True)],
        "Runs below random": [f"{v:.0%}" for v in aggregate["below_random"]],
        "Runs above best fixed set": [f"{v:.0%}" for v in aggregate["above_static"]],
        "Slots revealed": [f"{v:.2f}" for v in aggregate["slots"]],
        "Seconds per run": [f"{v:.0f}" for v in aggregate["wall"]],
    })
    parts["table_pooled"] = table(pooled, numeric=tuple(pooled.columns[1:]))
    frames = {"e1": e1}
    if (RESULTS / "raw" / "e2").exists():
        e2 = add_normalised(load_runs("e2"))
        frames["e2"] = e2
        usable = e2[~e2["degenerate"]]
        rows = []
        for arm in LADDER:
            mine = usable[usable["arm"] == arm]
            if mine.empty:
                continue
            by_condition = {str(c): g["norm"].to_numpy() for c, g in mine.groupby("condition")}
            low, high = stats.stratified_bootstrap(by_condition)
            mean = float(np.mean([v.mean() for v in by_condition.values()]))
            rows.append((LADDER_LABELS[arm], mean, low, high, arm in ("grpo", "grpo_nostd")))
        parts["chart_ladder"] = svg.levels(rows, "What the advantage is measured against (same data, same clipped objective)")
    tests = analysis.confirmatory(frames)
    labels = [f"{r['hypothesis']}  {svg.NAMES.get(r['a'], r['a'])} vs {_short(r['b'])}" for _, r in tests.iterrows()]
    parts["chart_tests"] = svg.forest(tests, "Confirmatory tests: pooled paired difference with 95% CI", labels, margin=analysis.MARGIN)
    shown = pd.DataFrame({
        "Test": tests["hypothesis"],
        "Claim": tests["claim"],
        "Difference": [f"{d:+.2f}" for d in tests["diff"]],
        "95% CI": [f"{a:+.2f} to {b:+.2f}" for a, b in zip(tests["ci_low"], tests["ci_high"], strict=True)],
        "Holm p": [f"{p:.4f}" if p >= 0.0001 else "<0.0001" for p in tests["p_holm"]],
        "Verdict": tests["verdict"],
    })
    parts["table_tests"] = table(shown, numeric=("Difference", "95% CI", "Holm p"))
    for name in ("e6_gradient_bias", "e5_summary", "e3_group_size"):
        path = analysis.SUMMARY / f"{name}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            parts[f"table_{name}"] = table(frame, numeric=tuple(c for c in frame.columns if frame[c].dtype.kind == "f"))
    return parts


def _short(arm: str) -> str:
    return {"mc_tuned": "TRL tuned", "ppo": "PPO", "grpo": "GRPO (canonical)",
            "grpo_batch": "batch baseline", "grpo_nostd": "GRPO (mean-centred)"}.get(arm, arm)


def main() -> None:
    template = TEMPLATE.read_text()
    parts = fragments()
    numbers = analysis.SUMMARY / "key_numbers.json"
    if numbers.exists():
        parts["key_numbers"] = escape(json.dumps(json.loads(numbers.read_text())))
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", template)) - set(parts))
    if missing:
        raise SystemExit(f"template placeholders with no fragment: {missing}")
    OUTPUT.write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: parts[m.group(1)], template))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
