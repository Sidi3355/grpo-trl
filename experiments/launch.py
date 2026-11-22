"""Run the study's stages. Every job writes one JSON record; finished jobs are skipped.

    python experiments/launch.py prepare --seeds 1000 1001 1002
    python experiments/launch.py budget
    python experiments/launch.py tune            # random search on dev conditions
    python experiments/launch.py tune2           # top three per method on two more dev seeds
    python experiments/launch.py e1 | e2 | e3 | e4

Results go to results/raw/<stage>/<dataset>__<regime>__<arm>__s<seed>.json.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BUDGETS = HERE / "budgets.json"
FROZEN = HERE / "frozen_configs.json"
N_SEARCH = 24


def _path(job: dict[str, Any]) -> Path:
    from grpo_trl.run import RESULTS

    name = f"{job['dataset']}__{job['regime']}__{job['arm_name']}__s{job['seed']}.json"
    return RESULTS / "raw" / job["exp"] / name


def run_job(job: dict[str, Any]) -> str:
    import torch

    from grpo_trl.run import build_problem, initial_record, run_arm, save

    torch.set_num_threads(1)
    out = _path(job)
    if out.exists():
        return f"skip {out.name}"
    start = time.time()
    problem = build_problem(job["dataset"], job["seed"], job["regime"])
    budget = json.loads(BUDGETS.read_text())[job["dataset"]][job["regime"]]["compute"]
    record = run_arm(problem, job["kind"], job["settings"], budget * job.get("budget_mult", 1.0))
    record.update(
        exp=job["exp"],
        arm_name=job["arm_name"],
        regime_name=job["regime"],
        budget_mult=job.get("budget_mult", 1.0),
        initial=initial_record(problem),
        wall_total=time.time() - start,
    )
    save(record, out)
    final = record["final"]
    return (
        f"{out.stem:60s} test {final['test']['ret']:+.3f} val {final['val']['ret']:+.3f} "
        f"slots {final['test']['n_slots']:.2f} {time.time() - start:5.0f}s"
    )


def prepare_job(key: tuple[str, int]) -> str:
    import torch

    from grpo_trl.run import build_problem

    torch.set_num_threads(1)
    start = time.time()
    problem = build_problem(key[0], key[1], "R2")
    return f"prepared {key[0]} s{key[1]} BA_all(val) {problem.info['ba_all_val']:.3f} {time.time() - start:.0f}s"


def budget_job(key: tuple[str, str, int]) -> dict[str, Any]:
    """Compute TRL's default trainer spends on one pass over the training set."""
    import torch

    from grpo_trl.mc import train_mc
    from grpo_trl.meter import Meter
    from grpo_trl.run import build_problem

    torch.set_num_threads(1)
    dataset, regime, seed = key
    problem = build_problem(dataset, seed, regime)
    meter, count = Meter(), {"n": 0}
    updates = math.ceil(len(problem.train) / 25)  # update_every=25, n_iterations=1

    def after(_model: object) -> bool:
        count["n"] += 1
        return count["n"] >= updates

    start = time.time()
    train_mc(problem, {}, meter, after)
    return {
        "dataset": dataset,
        "regime": regime,
        "seed": seed,
        "compute": meter.compute,
        "transitions": meter.transitions,
        "episodes": meter.episodes,
        "wall": time.time() - start,
    }


def _pool(function: Any, jobs: list[Any], workers: int) -> list[Any]:
    results = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for done, result in enumerate(pool.imap_unordered(function, jobs), 1):
            results.append(result)
            print(f"[{done}/{len(jobs)}] {result}", flush=True)
    return results


def _job(
    exp: str, dataset: str, regime: str, seed: int, arm_name: str, arm: Any, **extra: Any
) -> dict[str, Any]:
    return {
        "exp": exp,
        "dataset": dataset,
        "regime": regime,
        "seed": seed,
        "arm_name": arm_name,
        "kind": arm[0],
        "settings": arm[1],
        **extra,
    }


def jobs_for(stage: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    from grpo_trl import configs

    jobs: list[dict[str, Any]] = []
    if stage == "tune":
        for method in ("mc", "ppo", "grpo", "grpo_nostd"):
            for index, arm in enumerate(configs.search_configs(method, N_SEARCH)):
                for dataset, regime in configs.DEV_CONDITIONS:
                    jobs.append(
                        _job(
                            "tune",
                            dataset,
                            regime,
                            configs.DEV_SEEDS[0],
                            f"{method}-{index:02d}",
                            arm,
                        )
                    )
    elif stage == "tune2":
        top = json.loads((HERE / "tune_top3.json").read_text())
        for method, indices in top.items():
            searched = configs.search_configs(method, N_SEARCH)
            for index in indices:
                for dataset, regime in configs.DEV_CONDITIONS:
                    for seed in configs.DEV_SEEDS[1:]:
                        jobs.append(
                            _job(
                                "tune",
                                dataset,
                                regime,
                                seed,
                                f"{method}-{index:02d}",
                                searched[index],
                            )
                        )
    else:
        frozen = json.loads(FROZEN.read_text())
        arms = {name: (kind, settings) for name, (kind, settings) in frozen["arms"].items()}
        seeds = list(range(frozen["n_seeds"]))
        if stage == "e1":
            main = ("mc_default", "mc_tuned", "ppo", "grpo", "grpo_nostd")
            for dataset in configs.DATASETS:
                for regime in configs.REGIMES:
                    for seed in seeds:
                        for name in main:
                            jobs.append(_job("e1", dataset, regime, seed, name, arms[name]))
        elif stage == "e2":
            ladder = configs.ladder(arms["grpo"][1], arms["ppo"][1])
            for dataset, regime in configs.ABLATION_CONDITIONS:
                for seed in range(10):
                    for name, arm in ladder.items():
                        jobs.append(_job("e2", dataset, regime, seed, name, arm))
        elif stage == "e3":
            for dataset, regime in configs.ABLATION_CONDITIONS:
                for seed in range(5):  # Amendment A2.5
                    for base in ("grpo", "grpo_nostd"):
                        for size in (2, 4, 8, 16, 32):
                            settings = {**arms[base][1], "group_size": size}
                            jobs.append(
                                _job(
                                    "e3", dataset, regime, seed, f"{base}_G{size}", ("pg", settings)
                                )
                            )
        elif stage == "e4":
            for dataset in ("synth-route", "founders"):
                for regime in configs.REGIMES:
                    for seed in range(3):  # Amendment A2.5
                        for name in ("mc_default", "mc_tuned", "ppo", "grpo", "grpo_nostd"):
                            jobs.append(
                                _job("e4", dataset, regime, seed, name, arms[name], budget_mult=4.0)
                            )
        elif stage == "e7":  # post hoc (Amendment A4): a scale that does not shrink as the policy converges
            arm = ("pg", {**arms["grpo_nostd"][1], "advantage": "group_rawscale"})
            for seed in range(3):
                for dataset in ("synth-route", "founders"):
                    for regime in configs.REGIMES:
                        jobs.append(_job("e7x4", dataset, regime, seed, "grpo_rawscale", arm, budget_mult=4.0))
                for dataset in configs.DATASETS:
                    for regime in configs.REGIMES:
                        jobs.append(_job("e7x1", dataset, regime, seed, "grpo_rawscale", arm))
        else:
            raise SystemExit(f"unknown stage {stage}")
    return [job for job in jobs if not _path(job).exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1000, 1001, 1002])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    from grpo_trl import configs

    if args.stage == "prepare":
        _pool(prepare_job, [(d, s) for s in args.seeds for d in configs.DATASETS], args.workers)
        return
    if args.stage == "budget":
        keys = [
            (d, r, s)
            for d in configs.DATASETS
            for r in configs.REGIMES
            for s in configs.DEV_SEEDS[:2]
        ]
        rows = _pool(budget_job, keys, args.workers)
        table: dict[str, dict[str, dict[str, float]]] = {}
        for dataset in configs.DATASETS:
            for regime in configs.REGIMES:
                mine = [r for r in rows if r["dataset"] == dataset and r["regime"] == regime]
                table.setdefault(dataset, {})[regime] = {
                    key: sum(r[key] for r in mine) / len(mine)
                    for key in ("compute", "transitions", "episodes", "wall")
                }
        BUDGETS.write_text(json.dumps(table, indent=1))
        print(json.dumps(table, indent=1))
        return
    jobs = jobs_for(args.stage, args)
    # longest first within a deterministic shuffle keeps the workers evenly loaded
    jobs.sort(key=lambda j: (j["seed"], j["dataset"], j["regime"], j["arm_name"]))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{args.stage}: {len(jobs)} jobs", flush=True)
    start = time.time()
    _pool(run_job, jobs, args.workers)
    print(f"done in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
