"""Load run records and compute the pre-registered tables and tests.

    python -m grpo_trl.analysis          # writes results/summary/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import stats
from .run import RESULTS

SUMMARY = RESULTS / "summary"
MARGIN = 0.10  # non-inferiority margin, normalised units (PLAN.md section 3)
MIN_EFFECT = 0.05  # minimum effect of interest
ARM_LABELS = {
    "mc_default": "TRL shipped (defaults)",
    "mc_tuned": "TRL shipped (tuned)",
    "ppo": "PPO",
    "grpo": "GRPO (canonical)",
    "grpo_nostd": "GRPO (mean-centred)",
    "pg_nobaseline": "no baseline",
    "grpo_batch": "batch baseline",
    "grpo_loo": "GRPO, leave-one-out",
    "ppo_g": "PPO on grouped data",
    "ppo_priv": "PPO, privileged critic",
}


def load_runs(exp: str) -> pd.DataFrame:
    """One row per run of experiment ``exp``: its final checkpoint, anchors and cost."""
    rows: list[dict[str, Any]] = []
    for path in sorted((RESULTS / "raw" / exp).glob("*.json")):
        record = json.loads(path.read_text())
        final, anchors = record["final"], record["info"]["anchors"]
        row: dict[str, Any] = {
            "dataset": record["dataset"],
            "regime": record["regime_name"],
            "arm": record["arm_name"],
            "seed": record["seed"],
            "prevalence": record["info"]["prevalence"],
            "ba_all_val": record["info"]["ba_all_val"],
            "budget": record["budget"],
            "compute": final["compute"],
            "transitions": final["transitions"],
            "episodes": final["episodes"],
            "wall": final["wall"],
            "train_ret": final["train_ret"],
            "initial_test_ret": record["initial"]["test"]["ret"],
            "n_checkpoints": len(record["checkpoints"]),
        }
        row.update({f"test_{k}": v for k, v in final["test"].items()})
        row.update({f"val_{k}": v for k, v in final["val"].items()})
        best = max(record["checkpoints"], key=lambda c: c["val"]["ret"])
        row["test_ret_valbest"] = best["test"]["ret"]
        row.update({f"anchor_{k}": v for k, v in anchors.items() if k != "static_set"})
        row["static_set"] = "-".join(str(j) for j in anchors["static_set"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["condition"] = frame["dataset"] + "/" + frame["regime"]
    return _ordered(frame)


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows in the plan's order: regime, then dataset, then arm and seed."""
    from .configs import DATASETS

    rank = {name: i for i, name in enumerate(DATASETS)}
    keys = pd.DataFrame({
        "r": frame["regime"], "d": frame["dataset"].map(rank), "a": frame["arm"], "s": frame["seed"],
    })
    return frame.loc[keys.sort_values(["r", "d", "a", "s"]).index].reset_index(drop=True)


def load_curves(exp: str) -> pd.DataFrame:
    """One row per checkpoint, plus the untrained policy at compute 0."""
    rows: list[dict[str, Any]] = []
    for path in sorted((RESULTS / "raw" / exp).glob("*.json")):
        record = json.loads(path.read_text())
        key = {
            "dataset": record["dataset"], "regime": record["regime_name"],
            "arm": record["arm_name"], "seed": record["seed"], "budget": record["budget"],
        }
        rows.append({
            **key, "fraction": 0.0, "compute": 0.0, "transitions": 0, "wall": 0.0,
            "test_ret": record["initial"]["test"]["ret"], "val_ret": record["initial"]["val"]["ret"],
            "test_n_slots": record["initial"]["test"]["n_slots"],
        })
        for c in record["checkpoints"]:
            rows.append({
                **key, "fraction": c["fraction"], "compute": c["compute"],
                "transitions": c["transitions"], "wall": c["wall"],
                "test_ret": c["test"]["ret"], "val_ret": c["val"]["ret"],
                "test_n_slots": c["test"]["n_slots"],
            })
    frame = pd.DataFrame(rows)
    frame["condition"] = frame["dataset"] + "/" + frame["regime"]
    rank = {condition: i for i, condition in enumerate(dict.fromkeys(load_runs(exp)["condition"]))}
    order = frame["condition"].map(rank).sort_values(kind="stable").index
    return frame.loc[order].reset_index(drop=True)


DEGENERATE_SHARE = 0.05  # of the headroom (Amendment A2.1)


def anchor_table(frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Per condition: the reference policies' mean returns over seeds, and the usable gap.

    ``gap`` (best fixed set minus random) is the unit of normalised return.
    ``headroom`` (clairvoyant bound minus random) is the most any policy could
    add. A condition is degenerate when its gap is under 5% of its headroom:
    the two anchors are then too close to define a scale.
    """
    one_arm = frame[frame["arm"] == frame["arm"].iloc[0]]
    grouped = one_arm.groupby(["condition", "dataset", "regime"], sort=False)
    table = grouped.agg(
        random=(f"anchor_random_{split}", "mean"),
        static=(f"anchor_static_{split}", "mean"),
        reveal_all=(f"anchor_reveal_all_{split}", "mean"),
        clairvoyant=(f"anchor_clairvoyant_{split}", "mean"),
        prevalence=("prevalence", "mean"),
        ba_all_val=("ba_all_val", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    table["gap"] = table["static"] - table["random"]
    table["headroom"] = table["clairvoyant"] - table["random"]
    table["degenerate"] = table["gap"] < DEGENERATE_SHARE * table["headroom"]
    return table


def add_normalised(frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Add ``norm``: 0 is the random policy, 1 the best fixed slot set.

    The two anchors are the condition's means over seeds, so within a condition
    the normalisation is one affine map and paired differences keep their sign
    and relative size.
    """
    anchors = anchor_table(frame, split).set_index("condition")
    low = frame["condition"].map(anchors["random"])
    gap = frame["condition"].map(anchors["gap"])
    out = frame.copy()
    out["norm"] = (out[f"{split}_ret"] - low) / gap
    out["norm_valbest"] = (out["test_ret_valbest"] - low) / gap if split == "test" else np.nan
    out["degenerate"] = out["condition"].map(anchors["degenerate"])
    # the share of the headroom closed: defined in every condition (sensitivity analysis)
    out["headroom_share"] = (out[f"{split}_ret"] - low) / out["condition"].map(anchors["headroom"])
    return out


def paired(frame: pd.DataFrame, a: str, b: str, column: str = "norm") -> dict[str, np.ndarray]:
    """Per condition, ``a`` minus ``b`` on the seeds both ran."""
    wide = frame[frame["arm"].isin([a, b])].pivot_table(
        index=["condition", "seed"], columns="arm", values=column
    ).dropna()
    return {
        str(condition): (group[a] - group[b]).to_numpy()
        for condition, group in wide.groupby(level="condition", sort=False)
    }


@dataclass(frozen=True)
class Hypothesis:
    """A confirmatory comparison: ``a`` against ``b`` on a set of conditions."""

    name: str
    a: str
    b: str
    exp: str
    regimes: tuple[str, ...]
    kind: str  # "superiority" | "noninferiority" | "two-sided"
    claim: str


HYPOTHESES = (
    Hypothesis("H1", "grpo", "ppo", "e1", ("R1", "R2"), "superiority",
               "GRPO (canonical) beats PPO at equal compute"),
    Hypothesis("H2a", "grpo", "mc_tuned", "e1", ("R2",), "superiority",
               "GRPO (canonical) beats TRL's tuned optimiser in R2"),
    Hypothesis("H2b", "grpo", "mc_tuned", "e1", ("R1",), "noninferiority",
               "GRPO (canonical) is non-inferior to TRL's tuned optimiser in R1"),
    Hypothesis("H3", "grpo", "grpo_batch", "e2", ("R1", "R2"), "superiority",
               "Per-sample groups beat batch groups"),
    Hypothesis("H4", "grpo_nostd", "grpo", "e2", ("R1", "R2"), "two-sided",
               "Mean-centred and z-scored GRPO differ (guess: mean-centred is better)"),
    Hypothesis("H1'", "grpo_nostd", "ppo", "e1", ("R1", "R2"), "superiority",
               "GRPO (mean-centred) beats PPO at equal compute"),
    Hypothesis("H2a'", "grpo_nostd", "mc_tuned", "e1", ("R2",), "superiority",
               "GRPO (mean-centred) beats TRL's tuned optimiser in R2"),
    Hypothesis("H2b'", "grpo_nostd", "mc_tuned", "e1", ("R1",), "noninferiority",
               "GRPO (mean-centred) is non-inferior to TRL's tuned optimiser in R1"),
)


def compare(frame: pd.DataFrame, a: str, b: str, kind: str = "superiority", column: str = "norm") -> dict[str, Any]:
    """The pooled comparison of ``a`` and ``b`` on the non-degenerate conditions of ``frame``."""
    usable = frame[~frame["degenerate"]]
    differences = paired(usable, a, b, column)
    if not differences:
        return {"n_conditions": 0}
    mean = stats.pooled_mean(differences)
    low, high = stats.stratified_bootstrap(differences)
    if kind == "noninferiority":
        p = stats.sign_flip_p(differences, alternative="greater", shift=MARGIN)
    else:
        p = stats.sign_flip_p(differences)
    poi, poi_low, poi_high = stats.probability_of_improvement(differences)
    wins, ties, losses = stats.wins_ties_losses(differences)
    return {
        "a": a, "b": b, "kind": kind,
        "n_conditions": len(differences),
        "n_pairs": int(sum(len(d) for d in differences.values())),
        "diff": mean, "ci_low": low, "ci_high": high, "p": p,
        "poi": poi, "poi_low": poi_low, "poi_high": poi_high,
        "wins": wins, "ties": ties, "losses": losses,
    }


def verdict(result: dict[str, Any], p_holm: float) -> str:
    """The pre-registered reading of one confirmatory test."""
    if result["kind"] == "noninferiority":
        if p_holm < 0.05 and result["ci_low"] > -MARGIN:
            return "supported (non-inferior)"
        if result["ci_high"] < -MARGIN:
            return "rejected (inferior beyond the margin)"
        return "not shown"
    significant = p_holm < 0.05
    if significant and result["diff"] > 0:
        size = "" if abs(result["diff"]) >= MIN_EFFECT else ", below the minimum effect of interest"
        return f"supported{size}"
    if significant and result["diff"] < 0:
        return "reversed (significant in the opposite direction)"
    return "not supported"


def confirmatory(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every confirmatory test with its Holm-adjusted p-value and verdict."""
    rows = []
    for h in HYPOTHESES:
        if h.exp not in frames:
            continue
        frame = frames[h.exp]
        result = compare(frame[frame["regime"].isin(h.regimes)], h.a, h.b, h.kind)
        if result.get("n_conditions"):
            rows.append({"hypothesis": h.name, "claim": h.claim, **result})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = stats.holm(table["p"].tolist())
        table["robust"] = [
            (r["poi_low"] > 0.5) if r["diff"] > 0 else (r["poi_high"] < 0.5)
            for _, r in table.iterrows()
        ]
        table["verdict"] = [verdict(r, r["p_holm"]) for _, r in table.iterrows()]
    return table


def per_condition(frame: pd.DataFrame, column: str = "test_ret") -> pd.DataFrame:
    """Mean and 95% bootstrap CI of ``column`` per condition and arm."""
    rows = []
    for (condition, arm), group in frame.groupby(["condition", "arm"], sort=False):
        mean, low, high = stats.mean_ci(group[column].to_numpy())
        rows.append({"condition": condition, "arm": arm, "mean": mean, "low": low, "high": high, "n": len(group)})
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Per arm, over non-degenerate conditions: mean and IQM of normalised return, and costs."""
    usable = frame[~frame["degenerate"]]
    rows = []
    for arm in arms:
        mine = usable[usable["arm"] == arm]
        if mine.empty:
            continue
        by_condition = {str(c): g["norm"].to_numpy() for c, g in mine.groupby("condition", sort=False)}
        mean = float(np.mean([v.mean() for v in by_condition.values()]))
        low, high = stats.stratified_bootstrap(by_condition)
        iqm, iqm_low, iqm_high = stats.iqm(by_condition)
        spread = float(np.mean([v.std(ddof=1) for v in by_condition.values() if len(v) > 1]))
        rows.append({
            "arm": arm, "mean_norm": mean, "low": low, "high": high,
            "iqm": iqm, "iqm_low": iqm_low, "iqm_high": iqm_high,
            "seed_sd": spread,
            "below_random": float((mine["norm"] < 0).mean()),
            "above_static": float((mine["norm"] > 1).mean()),
            "slots": float(mine["test_n_slots"].mean()),
            "transitions": float(mine["transitions"].mean()),
            "wall": float(mine["wall"].mean()),
            "train_minus_test": float((mine["train_ret"] - mine["test_ret"]).mean()),
        })
    return pd.DataFrame(rows)


def fmt_ci(mean: float, low: float, high: float, digits: int = 3) -> str:
    return f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"


def save_table(table: pd.DataFrame, name: str, directory: Path = SUMMARY) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / f"{name}.csv", index=False)
