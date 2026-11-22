"""Arms, search spaces and the conditions of each experiment (PLAN.md sections 5, 6, 10)."""

from __future__ import annotations

from typing import Any

import numpy as np

DATASETS = ("synth-route", "founders", "kickstarter", "fakejobs", "credit", "adult")
REGIMES = ("R1", "R2")
DEV_CONDITIONS = (("synth-route", "R2"), ("founders", "R1"), ("founders", "R2"), ("credit", "R2"))
ABLATION_CONDITIONS = (("synth-route", "R2"), ("founders", "R2"), ("credit", "R2"), ("founders", "R1"))
DEV_SEEDS = (1000, 1001, 1002)

Arm = tuple[str, dict[str, Any]]  # ("mc" | "pg", settings)


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _pick(rng: np.random.Generator, options: tuple[Any, ...]) -> Any:
    return options[int(rng.integers(len(options)))]


def sample_mc(rng: np.random.Generator) -> dict[str, Any]:
    """One draw from the incumbent's search space (TRL arguments)."""
    return {
        "policy_lr": _log_uniform(rng, 3e-5, 3e-3),
        "n_rollouts": _pick(rng, (2, 5, 10)),
        "max_depth": _pick(rng, (2, 4, 8)),
        "tau_info": _pick(rng, (0.3, 1.0, 3.0)),
        "tau_stop": _pick(rng, (1.0, 4.0)),
        "update_every": _pick(rng, (10, 25, 50)),
        "train_epochs": _pick(rng, (1, 2, 4)),
        "policy_batch": _pick(rng, (64, 128)),
        "eps": _pick(rng, (0.05, 0.1, 0.3)),
        "min_queries_frac": _pick(rng, (0.0, 1 / 3, 1 / 2)),  # Amendment A1
    }


def sample_ppo(rng: np.random.Generator) -> dict[str, Any]:
    """One draw from PPO's search space."""
    return {
        "advantage": "gae",
        "collect": "single",
        "lr": _log_uniform(rng, 3e-5, 3e-3),
        "critic_lr_mult": _pick(rng, (1.0, 3.0)),
        "clip": _pick(rng, (0.1, 0.2, 0.3)),
        "epochs": _pick(rng, (1, 2, 4)),
        "minibatch": _pick(rng, (256, 1024)),
        "gae_lambda": _pick(rng, (0.8, 0.95, 1.0)),
        "ent_coef": _pick(rng, (0.0, 0.003, 0.01, 0.03)),
        "batch_episodes": _pick(rng, (256, 512, 1024, 2048)),
    }


def sample_grpo(rng: np.random.Generator, advantage: str = "group") -> dict[str, Any]:
    """One draw from GRPO's search space; the advantage is fixed by the arm."""
    return {
        "advantage": advantage,
        "collect": "grouped",
        "lr": _log_uniform(rng, 3e-5, 3e-3),
        "clip": _pick(rng, (0.1, 0.2, 0.3)),
        "epochs": _pick(rng, (1, 2, 4)),
        "minibatch": _pick(rng, (256, 1024)),
        "group_size": _pick(rng, (4, 8, 16, 32)),
        "batch_episodes": _pick(rng, (256, 512, 1024, 2048)),
        "kl_coef": _pick(rng, (0.0, 0.01, 0.04)),
        "ent_coef": _pick(rng, (0.0, 0.003, 0.01, 0.03)),
    }


def search_configs(method: str, n: int) -> list[Arm]:
    """The ``n`` configurations random search tries for ``method``.

    GRPO and its mean-centred variant are given the same draws.
    """
    seeds = {"mc": 11, "ppo": 12, "grpo": 13, "grpo_nostd": 13}
    rng = np.random.default_rng(seeds[method])
    if method == "mc":
        return [("mc", sample_mc(rng)) for _ in range(n)]
    if method == "ppo":
        return [("pg", sample_ppo(rng)) for _ in range(n)]
    advantage = "group" if method == "grpo" else "group_nostd"
    return [("pg", sample_grpo(rng, advantage)) for _ in range(n)]


MC_DEFAULT: Arm = ("mc", {})  # TRL as shipped


def ladder(grpo: dict[str, Any], ppo: dict[str, Any]) -> dict[str, Arm]:
    """E2's arms: GRPO's tuned settings with only the advantage changed."""
    base = {k: v for k, v in grpo.items() if k != "advantage"}
    critic = {**base, "gae_lambda": ppo["gae_lambda"], "critic_lr_mult": ppo["critic_lr_mult"]}
    return {
        "pg_nobaseline": ("pg", {**base, "advantage": "none"}),
        "grpo_batch": ("pg", {**base, "advantage": "batch"}),
        "grpo_nostd": ("pg", {**base, "advantage": "group_nostd"}),
        "grpo": ("pg", {**base, "advantage": "group"}),
        "grpo_loo": ("pg", {**base, "advantage": "group_loo"}),
        # Dr. GRPO (Liu et al., 2025): no std and no length normalisation
        "dr_grpo": ("pg", {**base, "advantage": "group_nostd", "length_norm": False}),
        "ppo_g": ("pg", {**critic, "advantage": "gae"}),
        "ppo_priv": ("pg", {**critic, "advantage": "gae_priv"}),
    }
