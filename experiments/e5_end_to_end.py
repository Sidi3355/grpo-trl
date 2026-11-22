"""E5: end to end through ``VerifiableRL.fit``, classifier co-trained as TRL does.

Runs in an environment where the *patched* library is installed (the proposed
``optimizer="group"``; see ``contrib/trl``). Everything goes through the public
API: construct, ``fit``, ``predict_proba``, ``decision_paths``.

    python experiments/e5_end_to_end.py [--workers 4]

Arms:
    shipped         ``VerifiableRL`` with the library's defaults, three passes of rollouts
    shipped_tuned   the rollout optimiser with the settings tuned in this study, three passes
    group           ``optimizer="group"`` with the settings frozen for mean-centred GRPO,
                    one pass (about the compute of three default rollout passes, or less)
The classifier is pretrained with the dataset's selected ``_pretrain`` settings
and then co-trained with TRL's defaults in every arm.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONDITIONS = (("founders", "R1"), ("founders", "R2"), ("credit", "R1"), ("credit", "R2"))
SEEDS = range(5)


def arms() -> dict[str, dict[str, Any]]:
    frozen = json.loads((HERE / "frozen_configs.json").read_text())
    tuned = {k: v for k, v in frozen["arms"]["mc_tuned"][1].items() if k != "min_queries_frac"}
    passes = {"n_iterations": frozen["e5_shipped_n_iterations"]}
    return {
        "shipped": {**passes},
        "shipped_tuned": {**tuned, **passes, "_frac": frozen["arms"]["mc_tuned"][1].get("min_queries_frac")},
        "group": {"optimizer": "group", **frozen["e5_group"]},
    }


def run(job: tuple[str, str, int, str]) -> str:
    import numpy as np
    import torch
    from think_reason_learn.verifiable_rl import VerifiableRL

    from grpo_trl.problem import returns_of_masks
    from grpo_trl.run import RESULTS, build_problem, classifier_settings

    torch.set_num_threads(1)
    dataset, regime, seed, arm = job
    out = RESULTS / "raw" / "e5" / f"{dataset}__{regime}__{arm}__s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    problem = build_problem(dataset, seed, regime)  # the split, the regime and its costs
    settings = dict(arms()[arm])
    frac = settings.pop("_frac", None)
    kwargs = problem.regime.trl_kwargs(list(problem.slots))
    if frac is not None and regime == "R2":
        floor = min(max(1, round(frac * problem.n_slots)), problem.regime.max_steps)
        kwargs.update(min_queries=floor, predict_min_queries=floor)
    model = VerifiableRL(
        dict(problem.slots), **kwargs, **classifier_settings(dataset), random_state=seed, **settings
    )
    labels = np.where(problem.train.y.numpy() > 0.5, "yes", "no")
    start = time.perf_counter()
    model.fit(problem.train.x.numpy(), labels)
    seconds = time.perf_counter() - start
    masks, paths = model._reveal(problem.test.x.numpy(), None)
    returns, proba = returns_of_masks(model.classifier_, problem.test, torch.from_numpy(masks), problem)
    predicted = model.predict(problem.test.x.numpy()) == "yes"
    truth = problem.test.y.numpy() > 0.5
    tp, fp, fn = (predicted & truth).sum(), (predicted & ~truth).sum(), (~predicted & truth).sum()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f05 = 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0
    record = {
        "dataset": dataset, "regime": regime, "seed": seed, "arm": arm, "settings": settings,
        "test_ret": float(returns.mean()), "precision": precision, "recall": recall, "f05": f05,
        "called": float(predicted.mean()), "n_slots": float(masks.sum(axis=1).mean()),
        "n_paths": len({tuple(p) for p in paths}), "fit_seconds": seconds,
        "anchors": problem.info["anchors"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    return f"{out.stem:44s} test {record['test_ret']:+.3f} F0.5 {f05:.3f} slots {record['n_slots']:.2f} fit {seconds:5.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [(d, r, s, a) for s in SEEDS for d, r in CONDITIONS for a in arms()]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for done, line in enumerate(pool.imap_unordered(run, jobs), 1):
            print(f"[{done}/{len(jobs)}] {line}", flush=True)


if __name__ == "__main__":
    main()
