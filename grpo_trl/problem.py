"""The decision problem every optimiser is given: TRL's VerifiableRL, unchanged.

A ``Problem`` holds one dataset split, a reward regime, the frozen classifier
and the initial policy weights. The networks, the action rule, the reward and
the deployment loop are TRL's own; what is added here is a batched sampler of
whole episodes (for policy-gradient methods) and the evaluation records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from think_reason_learn.llm import LLMSession
from think_reason_learn.verifiable_rl import VerifiableRL
from think_reason_learn.verifiable_rl._networks import Classifier, PolicyNet

Array = npt.NDArray[Any]


@dataclass(frozen=True)
class Regime:
    """Reward settings: every field is a ``VerifiableRL`` constructor argument."""

    name: str
    reward_tp: float
    reward_fp: float
    reward_tn: float
    reward_fn: float
    step_penalty: float
    reward_threshold: float
    min_queries: int
    max_steps: int
    costs: tuple[float, ...]

    def trl_kwargs(self, slot_names: list[str]) -> dict[str, Any]:
        """The regime as ``VerifiableRL`` arguments."""
        return {
            "slot_costs": dict(zip(slot_names, self.costs, strict=True)),
            "reward_tp": self.reward_tp,
            "reward_fp": self.reward_fp,
            "reward_tn": self.reward_tn,
            "reward_fn": self.reward_fn,
            "step_penalty": self.step_penalty,
            "reward_threshold": self.reward_threshold,
            "threshold": self.reward_threshold,
            "min_queries": self.min_queries,
            "predict_min_queries": self.min_queries,
            "max_steps": self.max_steps,
        }


def regime_r1(n_slots: int) -> Regime:
    """R1: the library's defaults, exactly."""
    return Regime(
        "R1",
        reward_tp=4.0,
        reward_fp=-16.0,
        reward_tn=0.0,
        reward_fn=-0.25,
        step_penalty=-0.1,
        reward_threshold=0.3,
        min_queries=min(3, n_slots),
        max_steps=min(5, n_slots),
        costs=(0.0,) * n_slots,
    )


def regime_r2(n_slots: int, prevalence: float, ba_all: float) -> Regime:
    """R2: return = Youden's J (2 * balanced accuracy - 1) - cost, Bayes call threshold.

    A right call earns ``1 / (2 * prevalence of its class)`` and a wrong call
    loses the same, so the expected terminal reward is ``TPR + TNR - 1`` and the
    Bayes-optimal call threshold is the prevalence. Revealing every slot costs
    half of what full information is worth (``2 * ba_all - 1``); costs rise
    linearly in slot order, from half to one and a half times the mean.
    """
    weights = np.linspace(0.5, 1.5, n_slots) / n_slots
    total = 0.5 * max(2.0 * ba_all - 1.0, 0.04)
    positive, negative = 1.0 / (2.0 * prevalence), 1.0 / (2.0 * (1.0 - prevalence))
    return Regime(
        "R2",
        reward_tp=positive,
        reward_fp=-negative,
        reward_tn=negative,
        reward_fn=-positive,
        step_penalty=0.0,
        reward_threshold=prevalence,
        min_queries=1,
        max_steps=n_slots,
        costs=tuple(float(c) for c in total * weights),
    )


@dataclass
class Split:
    """One part of the data, as tensors."""

    x: torch.Tensor  # (n, d) float32
    y: torch.Tensor  # (n,) float32, 1 = positive

    def __len__(self) -> int:
        return int(self.x.shape[0])


@dataclass
class Problem:
    """One (dataset, seed, regime): data, regime, frozen classifier, initial policy."""

    dataset: str
    seed: int
    slots: dict[str, int]
    regime: Regime
    train: Split
    val: Split
    test: Split
    classifier_state: dict[str, torch.Tensor]
    policy_state: dict[str, torch.Tensor]
    info: dict[str, Any] = field(default_factory=dict)
    policy_hidden: int = 512  # TRL's defaults
    clf_hidden: int = 256

    @property
    def n_slots(self) -> int:
        return len(self.slots)

    @property
    def state_dim(self) -> int:
        return sum(self.slots.values())

    @property
    def columns(self) -> torch.Tensor:
        """The slot of each feature column."""
        return torch.repeat_interleave(
            torch.arange(self.n_slots), torch.tensor(list(self.slots.values()))
        )

    @property
    def costs(self) -> torch.Tensor:
        return torch.tensor(self.regime.costs, dtype=torch.float32)

    def model(self, **optimiser_settings: Any) -> VerifiableRL:
        """A ``VerifiableRL`` for this problem: frozen classifier, initial policy.

        Built the way ``VerifiableRL._train`` builds one, so TRL's trainer and
        TRL's deployment loop (``_reveal``) run on it unchanged.
        """
        names = list(self.slots)
        model = VerifiableRL(
            dict(self.slots),
            **self.regime.trl_kwargs(names),
            random_state=self.seed,
            policy_hidden=self.policy_hidden,
            clf_hidden=self.clf_hidden,
            **optimiser_settings,
        )
        model._layout(self.train.x.numpy(), fitting=True)  # slot_names_, columns_, costs_
        model._build_networks(np.random.default_rng(0))  # replaced just below
        model.policy_.load_state_dict(self.policy_state)
        model.classifier_.load_state_dict(self.classifier_state)
        model.classifier_.requires_grad_(requires_grad=False).eval()
        model.classes_ = np.array([0, 1])
        model.threshold_ = self.regime.reward_threshold
        model.usage_ = LLMSession().usage
        return model


def masked_log_probs(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Log-probabilities of the policy over the allowed actions.

    ``pi(a|s)`` is proportional to TRL's ``action_probs(logits)[a]`` on the
    allowed actions, so its argmax is TRL's ``greedy(probs, allowed)``. Computed
    in log space, where a saturated STOP cannot produce 0/0.
    """
    log_info = torch.log_softmax(logits[:, :-1], dim=1) + F.logsigmoid(-logits[:, -1:])
    log_stop = F.logsigmoid(logits[:, -1:])
    logp = torch.cat([log_info, log_stop], dim=1).masked_fill(~allowed, -torch.inf)
    return logp - torch.logsumexp(logp, dim=1, keepdim=True)


def allowed_actions(masks: torch.Tensor, min_queries: int) -> torch.Tensor:
    """TRL's ``allowed_actions`` for tensors: unseen slots, STOP after ``min_queries``."""
    return torch.cat([~masks, (masks.sum(dim=1) >= min_queries)[:, None]], dim=1)


def terminal_rewards(
    logits: torch.Tensor, labels: torch.Tensor, regime: Regime
) -> torch.Tensor:
    """TRL's ``rewards``: the classifier's call scored against the label."""
    called = torch.sigmoid(logits) >= regime.reward_threshold
    positive = labels > 0.5
    return torch.where(
        positive,
        torch.where(called, regime.reward_tp, regime.reward_fn),
        torch.where(called, regime.reward_fp, regime.reward_tn),
    ).to(torch.float32)


@dataclass
class Episodes:
    """A batch of sampled episodes, padded to ``max_steps`` decisions."""

    rows: torch.Tensor  # (b,) index of each episode's sample in the split
    masks: torch.Tensor  # (b, t, s) slots revealed before each decision
    actions: torch.Tensor  # (b, t)
    logp: torch.Tensor  # (b, t) log-probability of the action taken
    logp_all: torch.Tensor  # (b, t, s + 1) of every action (for the KL term)
    valid: torch.Tensor  # (b, t) a decision was made
    rewards: torch.Tensor  # (b, t) reward received after each decision
    returns: torch.Tensor  # (b,) undiscounted episode return
    final_masks: torch.Tensor  # (b, s)

    @property
    def lengths(self) -> torch.Tensor:
        return self.valid.sum(dim=1)


@torch.no_grad()
def sample_episodes(
    policy: PolicyNet,
    classifier: Classifier,
    split: Split,
    rows: torch.Tensor,
    problem: Problem,
    generator: torch.Generator,
    *,
    uniform: bool = False,
) -> Episodes:
    """Play one episode per entry of ``rows`` with the stochastic policy.

    ``uniform=True`` ignores the policy and picks uniformly among the allowed
    actions (the random reference policy).
    """
    regime, n_slots, horizon = problem.regime, problem.n_slots, problem.regime.max_steps
    columns, costs = problem.columns, problem.costs
    b = len(rows)
    x, y = split.x[rows], split.y[rows]
    mask = torch.zeros(b, n_slots, dtype=torch.bool)
    active = torch.ones(b, dtype=torch.bool)
    masks = torch.zeros(b, horizon, n_slots, dtype=torch.bool)
    actions = torch.zeros(b, horizon, dtype=torch.long)
    logp = torch.zeros(b, horizon)
    logp_all = torch.zeros(b, horizon, n_slots + 1)
    valid = torch.zeros(b, horizon, dtype=torch.bool)
    rewards = torch.zeros(b, horizon)
    last = torch.zeros(b, dtype=torch.long)
    for t in range(horizon):
        live = torch.nonzero(active).squeeze(1)
        if not len(live):
            break
        seen = mask[live]
        allowed = allowed_actions(seen, regime.min_queries)
        if uniform:
            log_pi = torch.log(allowed / allowed.sum(dim=1, keepdim=True))
        else:
            log_pi = masked_log_probs(policy(x[live] * seen[:, columns]), allowed)
        action = torch.multinomial(log_pi.exp(), 1, generator=generator).squeeze(1)
        masks[live, t], actions[live, t], valid[live, t] = seen, action, True
        logp[live, t] = log_pi.gather(1, action[:, None]).squeeze(1)
        logp_all[live, t] = log_pi
        reveal = action < n_slots
        mask[live[reveal], action[reveal]] = True
        rewards[live[reveal], t] = regime.step_penalty - costs[action[reveal]]
        revealed = mask[live].sum(dim=1)
        done = ~reveal | (revealed >= n_slots) | (revealed >= horizon)
        last[live[done]] = t
        active[live[done]] = False
    final = terminal_rewards(classifier(x * mask[:, columns]), y, regime)
    rewards[torch.arange(b), last] += final
    return Episodes(
        rows, masks, actions, logp, logp_all, valid, rewards, rewards.sum(dim=1), mask
    )


def returns_of_masks(
    classifier: Classifier,
    split: Split,
    masks: torch.Tensor,
    problem: Problem,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return and positive probability of ending with ``masks`` revealed."""
    regime = problem.regime
    with torch.no_grad():
        logits = classifier(split.x * masks[:, problem.columns])
    reveals = masks.sum(dim=1).to(torch.float32)
    returns = (
        terminal_rewards(logits, split.y, regime)
        - masks.to(torch.float32) @ problem.costs
        + regime.step_penalty * reveals
    )
    return returns, torch.sigmoid(logits)


def evaluate(model: VerifiableRL, split: Split, problem: Problem) -> dict[str, float]:
    """Deploy the policy greedily with TRL's own ``_reveal`` and score it."""
    masks_np, paths = model._reveal(split.x.numpy(), None)
    masks = torch.from_numpy(masks_np)
    returns, proba = returns_of_masks(model.classifier_, split, masks, problem)
    record = _prediction_metrics(proba, split.y, problem.regime.reward_threshold)
    record.update(
        ret=float(returns.mean()),
        n_slots=float(masks.sum(dim=1).float().mean()),
        cost=float((masks.float() @ problem.costs).mean()),
        n_paths=float(len({tuple(p) for p in paths})),
    )
    record.update(_first_slots(paths, problem.n_slots))
    return record


def _first_slots(paths: list[list[int]], n_slots: int) -> dict[str, float]:
    first = np.bincount([p[0] for p in paths if p], minlength=n_slots) / max(len(paths), 1)
    return {f"first_{j}": float(share) for j, share in enumerate(first)}


def _prediction_metrics(
    proba: torch.Tensor, labels: torch.Tensor, threshold: float
) -> dict[str, float]:
    called, positive = (proba >= threshold).numpy(), (labels > 0.5).numpy()
    tp = float((called & positive).sum())
    fp = float((called & ~positive).sum())
    fn = float((~called & positive).sum())
    tn = float((~called & ~positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f05 = 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "bal_acc": 0.5 * (recall + specificity),
        "called": (tp + fp) / len(called),
    }


def subsets(n_slots: int, regime: Regime) -> torch.Tensor:
    """Every set of slots a non-adaptive policy may end with."""
    low, high = max(regime.min_queries, 1), min(regime.max_steps, n_slots)
    rows = [
        [j in chosen for j in range(n_slots)]
        for size in range(low, high + 1)
        for chosen in combinations(range(n_slots), size)
    ]
    return torch.tensor(rows, dtype=torch.bool)


def anchors(problem: Problem, classifier: Classifier, n_random: int = 20) -> dict[str, Any]:
    """Reference policies: random, the best fixed slot set, reveal-all, and a bound.

    The static oracle is chosen on train and scored on each split. The
    clairvoyant bound lets every sample pick its own best set knowing its
    label: no policy can reach it; it only bounds the scale.
    """
    candidates = subsets(problem.n_slots, problem.regime)
    out: dict[str, Any] = {}
    per_split: dict[str, torch.Tensor] = {}
    for name, split in (("train", problem.train), ("val", problem.val), ("test", problem.test)):
        table = torch.stack(
            [
                returns_of_masks(classifier, split, m.expand(len(split), -1), problem)[0]
                for m in candidates
            ]
        )  # (n_subsets, n)
        per_split[name] = table
    best = int(per_split["train"].mean(dim=1).argmax())
    out["static_set"] = [int(j) for j in torch.nonzero(candidates[best]).squeeze(1)]
    full = int(candidates.sum(dim=1).argmax())  # a largest allowed set (the first, in index order)
    generator = torch.Generator().manual_seed(problem.seed + 7919)
    for name, split in (("train", problem.train), ("val", problem.val), ("test", problem.test)):
        table = per_split[name]
        out[f"static_{name}"] = float(table[best].mean())
        out[f"reveal_all_{name}"] = float(table[full].mean())
        out[f"clairvoyant_{name}"] = float(table.max(dim=0).values.mean())
        rows = torch.arange(len(split)).repeat(n_random)
        episodes = sample_episodes(
            None, classifier, split, rows, problem, generator, uniform=True  # type: ignore[arg-type]
        )
        out[f"random_{name}"] = float(episodes.returns.mean())
    return out
