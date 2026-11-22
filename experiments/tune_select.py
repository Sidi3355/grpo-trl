"""Pick each method's configuration from the dev-seed search (PLAN.md 6.3, Amendment A2.2).

    python experiments/tune_select.py top3     # after `launch.py tune`: candidates for stage 2
    python experiments/tune_select.py final    # after `launch.py tune2`: the frozen choice

Score of a configuration: its validation return at the end of the budget,
normalised per dev condition (0 = random policy, 1 = best fixed slot set), and
averaged over dev conditions and the seeds run. The two anchors of a condition
are its means over the three dev seeds (``dev_anchors``), as in the confirmatory
analysis; a condition whose gap is under 5% of its headroom would be left out.

Candidates for stage 2 are each method's three best after one seed, under this
score and under the score as first computed (anchors of seed 1000 alone); the
union is re-run on two more seeds and the best mean over all three is frozen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from grpo_trl import configs
from grpo_trl.analysis import DEGENERATE_SHARE, load_runs

HERE = Path(__file__).resolve().parent
METHODS = ("mc", "ppo", "grpo", "grpo_nostd")


def dev_anchors() -> pd.DataFrame:
    """Validation anchors of each dev condition, averaged over the dev seeds."""
    path = HERE / "dev_anchors.json"
    if not path.exists():
        import torch

        from grpo_trl.run import build_problem

        torch.set_num_threads(1)
        rows = []
        for dataset, regime in configs.DEV_CONDITIONS:
            found = [build_problem(dataset, s, regime).info["anchors"] for s in configs.DEV_SEEDS]
            rows.append(
                {
                    "condition": f"{dataset}/{regime}",
                    **{
                        k: float(np.mean([a[f"{k}_val"] for a in found]))
                        for k in ("random", "static", "clairvoyant")
                    },
                    "per_seed_gap": [a["static_val"] - a["random_val"] for a in found],
                }
            )
        path.write_text(json.dumps(rows, indent=1))
    table = pd.DataFrame(json.loads(path.read_text())).set_index("condition")
    table["gap"] = table["static"] - table["random"]
    table["degenerate"] = table["gap"] < DEGENERATE_SHARE * (table["clairvoyant"] - table["random"])
    return table


def scores(seeds: tuple[int, ...]) -> pd.DataFrame:
    anchors = dev_anchors()
    frame = load_runs("tune")
    frame = frame[frame["seed"].isin(seeds)].copy()
    frame = frame[~frame["condition"].map(anchors["degenerate"])]
    frame["score"] = (frame["val_ret"] - frame["condition"].map(anchors["random"])) / frame[
        "condition"
    ].map(anchors["gap"])
    frame["score_seed_anchors"] = (frame["val_ret"] - frame["anchor_random_val"]) / (
        frame["anchor_static_val"] - frame["anchor_random_val"]
    )
    frame["method"] = frame["arm"].str.rsplit("-", n=1).str[0]
    frame["index"] = frame["arm"].str.rsplit("-", n=1).str[1].astype(int)
    return frame


def table(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "index"]
    wide = frame.pivot_table(index=keys, columns="condition", values="score")
    wide["n_runs"] = frame.groupby(keys).size()
    wide["mean"] = frame.groupby(keys)["score"].mean()
    wide["mean_seed_anchors"] = frame.groupby(keys)["score_seed_anchors"].mean()
    return wide.reset_index()


def main(stage: str) -> None:
    pd.set_option("display.width", 240)
    print(dev_anchors().round(3).to_string())
    if stage == "top3":
        ranked = table(scores(configs.DEV_SEEDS[:1]))
        ranked.to_csv(HERE / "tune_search.csv", index=False)
        top = {}
        for method in METHODS:
            mine = ranked[ranked["method"] == method]
            best = set(mine.nlargest(3, "mean")["index"]) | set(
                mine.nlargest(3, "mean_seed_anchors")["index"]
            )
            print(
                f"\n{method}\n{mine.sort_values('mean', ascending=False).head(6).round(3).to_string(index=False)}"
            )
            top[method] = sorted(int(i) for i in best)
        (HERE / "tune_top3.json").write_text(json.dumps(top, indent=1))
        print(top)
        return
    top = json.loads((HERE / "tune_top3.json").read_text())
    frame = scores(configs.DEV_SEEDS)
    chosen, rows = {}, []
    for method in METHODS:
        mine = frame[(frame["method"] == method) & frame["index"].isin(top[method])]
        ranked = table(mine).sort_values("mean", ascending=False)
        print(f"\n{method}\n{ranked.round(3).to_string(index=False)}")
        rows.append(ranked)
        index = int(ranked["index"].iloc[0])
        kind, settings = configs.search_configs(method, 24)[index]
        chosen[method] = {
            "index": index,
            "kind": kind,
            "settings": settings,
            "dev_score": float(ranked["mean"].iloc[0]),
        }
    pd.concat(rows).to_csv(HERE / "tune_finalists.csv", index=False)
    (HERE / "tune_chosen.json").write_text(json.dumps(chosen, indent=1))
    print(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
