"""One run: build the problem, train one arm under a compute budget, record checkpoints.

A problem (data split, frozen classifier, initial policy, anchors) depends only
on the dataset and the seed, so every arm of a condition gets the identical one.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from think_reason_learn.verifiable_rl import VerifiableRL

from . import TRL_COMMIT
from .data import CACHE, ROOT, load_pool, split_pool
from .mc import train_mc
from .meter import Meter
from .pg import PGSettings, train_pg
from .problem import (
    Problem,
    Regime,
    Split,
    anchors,
    evaluate,
    regime_r1,
    regime_r2,
    returns_of_masks,
)

SELECTION = ROOT / "experiments" / "classifier_selection"


def classifier_settings(dataset: str) -> dict[str, Any]:
    """The frozen classifier's ``_pretrain`` settings, chosen on dev seeds.

    Written by ``experiments/select_classifier.py`` (its rule is in that file).
    """
    chosen = json.loads((SELECTION / f"{dataset}.json").read_text())["chosen"]
    return {k: chosen[k] for k in ("pretrain_epochs", "pretrain_lr", "pretrain_batch")}


N_CHECKPOINTS = 10


def build_problem(
    dataset: str, seed: int, regime: str, pretrain: dict[str, Any] | None = None
) -> Problem:
    """The problem for (dataset, seed, regime); the classifier is cached on disk.

    The classifier is trained by TRL's random-reveal curriculum
    (``VerifiableRL._pretrain``), as ``VerifiableRL._train`` does, then frozen.
    """
    pool = load_pool(dataset)
    (xtr, ytr), (xva, yva), (xte, yte) = split_pool(pool, seed)
    train, val, test = (
        Split(torch.from_numpy(x), torch.from_numpy(y))
        for x, y in ((xtr, ytr), (xva, yva), (xte, yte))
    )
    settings = pretrain or classifier_settings(dataset)
    tag = "_".join(f"{v:g}" for v in settings.values())
    cached = CACHE / "problems" / f"{dataset}_s{seed}_{tag}.pt"
    n_slots = len(pool.slots)
    if cached.exists():
        stored = torch.load(cached, weights_only=True)
    else:
        rng = np.random.default_rng(seed)
        shell = VerifiableRL(dict(pool.slots), random_state=seed, **settings)
        features = shell._layout(xtr, fitting=True)
        shell._build_networks(rng)  # as VerifiableRL._train does
        policy_state = {k: v.clone() for k, v in shell.policy_.state_dict().items()}
        shell._pretrain(features, ytr, rng)
        stored = {
            "policy": policy_state,
            "classifier": {k: v.clone() for k, v in shell.classifier_.state_dict().items()},
        }
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stored, cached)
    prevalence = float(ytr.mean())
    problem = Problem(
        dataset, seed, dict(pool.slots), regime_r1(n_slots), train, val, test,
        stored["classifier"], stored["policy"],
    )
    classifier = problem.model().classifier_
    everything = torch.ones(len(val), n_slots, dtype=torch.bool)
    with torch.no_grad():
        proba = torch.sigmoid(classifier(val.x * everything[:, problem.columns]))
    called, positive = (proba >= prevalence).numpy(), (val.y > 0.5).numpy()
    ba_all = 0.5 * (called[positive].mean() + (~called[~positive]).mean())
    if regime == "R2":
        problem.regime = regime_r2(n_slots, prevalence, float(ba_all))
    elif regime != "R1":
        raise ValueError(regime)
    problem.info = {"prevalence": prevalence, "ba_all_val": float(ba_all)}
    problem.info["anchors"] = anchors(problem, classifier)
    return problem


def run_arm(
    problem: Problem,
    arm: str,
    settings: dict[str, Any],
    budget: float,
    *,
    n_checkpoints: int = N_CHECKPOINTS,
) -> dict[str, Any]:
    """Train ``arm`` on ``problem`` within ``budget`` MACs; return the run record.

    ``arm`` is ``"mc"`` (TRL's trainer; ``settings`` are ``VerifiableRL``
    arguments) or ``"pg"`` (``settings`` are ``PGSettings`` fields).
    Training stops before the update that would exceed the budget.
    """
    torch.set_num_threads(1)
    meter = Meter()
    checkpoints: list[dict[str, Any]] = []
    start = time.perf_counter()
    state = {"next": 1, "last_compute": 0.0, "largest_update": 0.0, "eval_time": 0.0}

    def record(model: VerifiableRL, fraction: float) -> None:
        began = time.perf_counter()
        with meter.paused():
            entry: dict[str, Any] = {
                "fraction": fraction,
                **meter.snapshot(),
                "val": evaluate(model, problem.val, problem),
                "test": evaluate(model, problem.test, problem),
                "train_ret": evaluate(model, problem.train, problem)["ret"],
            }
        state["eval_time"] += time.perf_counter() - began
        entry["wall"] = time.perf_counter() - start - state["eval_time"]
        checkpoints.append(entry)

    def after_update(model: VerifiableRL) -> bool:
        spent = meter.compute
        state["largest_update"] = max(state["largest_update"], spent - state["last_compute"])
        state["last_compute"] = spent
        stop = spent + state["largest_update"] > budget
        while state["next"] < n_checkpoints and spent >= budget * state["next"] / n_checkpoints:
            if not stop:
                record(model, state["next"] / n_checkpoints)
            state["next"] += 1
        if stop:
            record(model, 1.0)
        return stop

    if arm == "mc":
        model = train_mc(problem, settings, meter, after_update)
    elif arm == "pg":
        model = train_pg(problem, PGSettings(**settings), meter, after_update)
    else:
        raise ValueError(arm)
    meter.detach()
    del model
    return {
        "dataset": problem.dataset,
        "seed": problem.seed,
        "regime": asdict(problem.regime),
        "arm": arm,
        "settings": settings,
        "budget": budget,
        "info": problem.info,
        "checkpoints": checkpoints,
        "final": checkpoints[-1],
        "macs": dict(meter.macs),
        "trl_commit": TRL_COMMIT,
        "torch": torch.__version__,
    }


def initial_record(problem: Problem) -> dict[str, Any]:
    """Scores of the untrained policy (compute 0), for the start of learning curves."""
    model = problem.model()
    return {
        "val": evaluate(model, problem.val, problem),
        "test": evaluate(model, problem.test, problem),
        "train_ret": evaluate(model, problem.train, problem)["ret"],
    }


def save(record: dict[str, Any], path: Path) -> None:
    """Write a run record as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))


RESULTS = ROOT / "results"
__all__ = ["RESULTS", "build_problem", "run_arm", "initial_record", "save", "Regime", "returns_of_masks"]
