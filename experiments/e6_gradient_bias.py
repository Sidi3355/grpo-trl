"""E6 (exploratory, added after the pilot): why does dividing by the group's std hurt?

No training. For a problem's *initial* policy, play many episodes per sample and
compare the direction each advantage rule would push the policy with the
direction of the true policy gradient.

The reference is the plain score-function gradient with a per-sample mean
baseline, ``sum_t grad log pi(a_t|s_t) * (R - mean of the group)``, which is the
policy gradient of the expected return (up to the factor ``1 - 1/G``). For each
rule the script reports

* the cosine between its update direction and the reference;
* the mean advantage it gives episodes by how many slots they revealed, each
  rule scaled to unit standard deviation so the columns compare;
* the share of groups whose episodes all end in the same call (right or wrong),
  so that their returns differ by costs alone.

    python experiments/e6_gradient_bias.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

GROUP = 16
SEEDS = range(3)


def run(job: tuple[str, str, int]) -> str:
    import torch

    from grpo_trl.configs import DATASETS  # noqa: F401  (import check)
    from grpo_trl.pg import batch_advantages, group_advantages
    from grpo_trl.problem import allowed_actions, masked_log_probs, sample_episodes
    from grpo_trl.run import RESULTS, build_problem

    torch.set_num_threads(1)
    dataset, regime, seed = job
    out = RESULTS / "raw" / "e6" / f"{dataset}__{regime}__s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    problem = build_problem(dataset, seed, regime)
    model = problem.model()
    policy = model.policy_
    generator = torch.Generator().manual_seed(seed)
    rows = torch.arange(len(problem.train)).repeat_interleave(GROUP)
    episodes = sample_episodes(policy, model.classifier_, problem.train, rows, problem, generator)
    returns, lengths = episodes.returns, episodes.lengths.to(torch.float32)
    grouped = returns.view(-1, GROUP)
    centred = (grouped - grouped.mean(dim=1, keepdim=True)).reshape(-1)
    rules: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        # advantage per episode, weight per step
        "reference": (centred, torch.ones_like(returns)),
        "grpo": (group_advantages(returns, GROUP, "group"), 1.0 / lengths),
        "grpo_no_length_norm": (group_advantages(returns, GROUP, "group"), torch.ones_like(returns)),
        "grpo_nostd": (group_advantages(returns, GROUP, "group_nostd"), 1.0 / lengths),
        "grpo_batch": (batch_advantages(returns, "batch"), 1.0 / lengths),
        "no_baseline": (batch_advantages(returns, "none"), 1.0 / lengths),
    }
    episode, step = torch.nonzero(episodes.valid, as_tuple=True)
    masks, actions = episodes.masks[episode, step], episodes.actions[episode, step]
    allowed = allowed_actions(masks, problem.regime.min_queries)
    columns, x = problem.columns, problem.train.x
    parameters = [p for p in policy.parameters()]
    gradients = {name: [torch.zeros_like(p) for p in parameters] for name in rules}
    for index in torch.arange(len(episode)).split(8192):
        states = x[rows[episode[index]]] * masks[index][:, columns]
        logp = masked_log_probs(policy(states), allowed[index]).gather(1, actions[index, None]).squeeze(1)
        for name, (advantage, weight) in rules.items():
            coefficient = (advantage * weight)[episode[index]]
            grads = torch.autograd.grad((coefficient * logp).sum(), parameters, retain_graph=True)
            for total, g in zip(gradients[name], grads, strict=True):
                total += g
    flat = {name: torch.cat([g.reshape(-1) for g in grads]) for name, grads in gradients.items()}
    reference = flat["reference"]
    cosine = {
        name: float(torch.dot(g, reference) / (g.norm() * reference.norm() + 1e-12))
        for name, g in flat.items()
    }
    reveals = episodes.final_masks.sum(dim=1)
    by_length: dict[str, dict[str, float]] = {}
    for name in ("reference", "grpo"):
        advantage = rules[name][0]
        scaled = advantage / (advantage.std() + 1e-12)
        by_length[name] = {
            str(int(k)): float(scaled[reveals == k].mean()) for k in reveals.unique()
        }
    with torch.no_grad():
        final_states = x[rows] * episodes.final_masks[:, columns]
        called = torch.sigmoid(model.classifier_(final_states)) >= problem.regime.reward_threshold
    same_call = (called.view(-1, GROUP).float().std(dim=1) == 0).float().mean()
    spread = grouped.std(dim=1)
    record: dict[str, Any] = {
        "dataset": dataset, "regime": regime, "seed": seed, "group_size": GROUP,
        "cosine_to_reference": cosine, "mean_advantage_by_reveals": by_length,
        "share_of_episodes_by_reveals": {str(int(k)): float((reveals == k).float().mean()) for k in reveals.unique()},
        "share_groups_same_call": float(same_call),
        "group_std_p10_p50_p90": [float(q) for q in torch.quantile(spread, torch.tensor([0.1, 0.5, 0.9]))],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    return f"{out.stem:32s} cos grpo {cosine['grpo']:+.2f} nostd {cosine['grpo_nostd']:+.2f} batch {cosine['grpo_batch']:+.2f} same-call groups {float(same_call):.2f}"


def main() -> None:
    from grpo_trl.configs import DATASETS, REGIMES

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [(d, r, s) for s in SEEDS for d in DATASETS for r in REGIMES]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for done, line in enumerate(pool.imap_unordered(run, jobs), 1):
            print(f"[{done}/{len(jobs)}] {line}", flush=True)


if __name__ == "__main__":
    main()
