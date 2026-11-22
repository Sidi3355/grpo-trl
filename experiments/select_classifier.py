"""Choose the frozen classifier's pretraining settings, per dataset, on dev seeds.

The classifier is part of the decision problem, not of any optimiser. It is
trained by TRL's own ``VerifiableRL._pretrain``; this script only chooses that
method's three settings, by a mechanical rule fixed before any arm was tuned:

    the grid setting with the lowest validation log loss under uniformly random
    reveals, averaged over dev seeds 1000 and 1001; settings within 0.005 of the
    best are tied, and a tie goes to the fewest gradient steps, then the lowest loss.

Usage: python experiments/select_classifier.py DATASET   (writes one JSON line)
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from think_reason_learn.verifiable_rl import VerifiableRL
from think_reason_learn.verifiable_rl._verifiable_rl import _random_masks

from grpo_trl.data import load_pool, split_pool

GRID = list(itertools.product((1e-4, 3e-4, 1e-3), (10, 30, 60)))
DEV_SEEDS = (1000, 1001)
BATCH = 64
OUT = Path(__file__).parent / "classifier_selection"


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main(dataset: str) -> None:
    torch.set_num_threads(1)
    pool = load_pool(dataset)
    n_slots = len(pool.slots)
    rows = []
    for lr, epochs in GRID:
        losses, gaps = [], []
        for seed in DEV_SEEDS:
            (xtr, ytr), (xva, yva), _ = split_pool(pool, seed)
            rng = np.random.default_rng(seed)
            model = VerifiableRL(
                dict(pool.slots), random_state=seed,
                pretrain_epochs=epochs, pretrain_lr=lr, pretrain_batch=BATCH,
            )
            features = model._layout(xtr, fitting=True)
            model._build_networks(rng)
            model._pretrain(features, ytr, rng)
            mask_rng = np.random.default_rng(5)
            scores = {}
            for name, (x, y) in (("train", (xtr, ytr)), ("val", (xva, yva))):
                masks = np.vstack([_random_masks(len(x), n_slots, 1, n_slots, mask_rng) for _ in range(4)])
                with torch.no_grad():
                    p = torch.sigmoid(model.classifier_(model._states(np.tile(x, (4, 1)), masks)))
                scores[name] = log_loss(p.numpy().astype(float), np.tile(y, 4))
            losses.append(scores["val"])
            gaps.append(scores["val"] - scores["train"])
        rows.append({
            "pretrain_lr": lr, "pretrain_epochs": epochs, "pretrain_batch": BATCH,
            "val_log_loss": float(np.mean(losses)), "val_minus_train": float(np.mean(gaps)),
        })
        print(dataset, rows[-1], flush=True)
    best = min(r["val_log_loss"] for r in rows)
    tied = [r for r in rows if r["val_log_loss"] <= best + 0.005]
    chosen = min(tied, key=lambda r: (r["pretrain_epochs"], r["val_log_loss"]))
    OUT.mkdir(exist_ok=True)
    (OUT / f"{dataset}.json").write_text(json.dumps({"dataset": dataset, "chosen": chosen, "grid": rows}, indent=1))
    print("CHOSEN", dataset, chosen, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
