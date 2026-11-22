"""The test bed is right before any result is read.

These tests pin the pieces every conclusion rests on: the sampled policy is the
one TRL deploys, returns are TRL's returns, advantages are what the papers
define, the compute meter counts what it says, and each optimiser can solve a
problem whose answer is known.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from think_reason_learn.verifiable_rl._episodes import (
    Sample,
    Search,
    greedy,
    policy_probs,
    rollouts,
)
from think_reason_learn.verifiable_rl._episodes import (
    allowed_actions as trl_allowed,
)
from think_reason_learn.verifiable_rl._episodes import (
    rewards as trl_rewards,
)
from think_reason_learn.verifiable_rl._networks import Classifier, PolicyNet

from grpo_trl.meter import Meter, macs
from grpo_trl.pg import PGSettings, batch_advantages, gae, group_advantages, train_pg
from grpo_trl.problem import (
    Problem,
    Regime,
    Split,
    allowed_actions,
    anchors,
    evaluate,
    masked_log_probs,
    regime_r1,
    regime_r2,
    returns_of_masks,
    sample_episodes,
    subsets,
    terminal_rewards,
)

SLOTS = {"signal": 2, "noise": 2, "more_noise": 2}


def toy_problem(regime: Regime | None = None, n: int = 240, seed: int = 0) -> Problem:
    """Three slots; the label is the sign of the first column of ``signal``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3 * n, 6)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    torch.manual_seed(seed)
    classifier = Classifier(6, 16)
    with torch.no_grad():  # a classifier that reads the signal slot exactly
        for layer in classifier.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        first, second, third, last = (m for m in classifier.net if isinstance(m, torch.nn.Linear))
        first.weight[0, 0], first.weight[1, 0] = 1.0, -1.0
        second.weight[0, 0], second.weight[1, 1] = 1.0, 1.0
        third.weight[0, 0], third.weight[1, 1] = 1.0, 1.0
        last.weight[0, 0], last.weight[0, 1] = 8.0, -8.0
    parts = [
        Split(torch.from_numpy(x[i * n : (i + 1) * n]), torch.from_numpy(y[i * n : (i + 1) * n]))
        for i in range(3)
    ]
    regime = regime or Regime(
        "toy",
        reward_tp=1.0,
        reward_fp=-1.0,
        reward_tn=1.0,
        reward_fn=-1.0,
        step_penalty=0.0,
        reward_threshold=0.5,
        min_queries=1,
        max_steps=3,
        costs=(0.2, 0.1, 0.1),
    )
    return Problem(
        "toy",
        seed,
        dict(SLOTS),
        regime,
        *parts,
        classifier_state=classifier.state_dict(),
        policy_state=PolicyNet(6, 3, 16).state_dict(),
        policy_hidden=16,
        clf_hidden=16,
    )


# ------------------------------------------------------------------ the policy and the reward


def test_masked_policy_matches_trl_action_rule() -> None:
    """argmax of the sampled policy is TRL's ``greedy(probs, allowed)``, state by state."""
    torch.manual_seed(1)
    policy = PolicyNet(6, 3, 16)
    states = torch.randn(200, 6)
    masks = torch.rand(200, 3) < 0.4
    masks[masks.all(dim=1), 0] = False  # an episode with every slot revealed has ended
    for min_queries in (0, 1, 2):
        allowed = allowed_actions(masks, min_queries)
        assert (allowed.numpy() == trl_allowed(masks.numpy(), min_queries)).all()
        with torch.no_grad():
            ours = masked_log_probs(policy(states), allowed)
        theirs = greedy(policy_probs(policy, states), allowed.numpy())
        assert (ours.argmax(dim=1).numpy() == theirs).all()
        assert torch.allclose(ours.exp().sum(dim=1), torch.ones(200), atol=1e-5)
        assert (ours[~allowed] == -torch.inf).all()


def test_masked_policy_survives_a_saturated_stop() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 80.0], [0.0, 1.0, 2.0, -80.0]])
    allowed = torch.tensor([[True, True, False, False], [True, False, True, True]])
    logp = masked_log_probs(logits, allowed)
    assert torch.isfinite(logp[allowed]).all()
    assert torch.allclose(logp[0, :2].exp(), torch.softmax(logits[0, :2], dim=0), atol=1e-6)


def test_terminal_rewards_match_trl() -> None:
    torch.manual_seed(2)
    classifier = Classifier(6, 16)
    states = torch.randn(64, 6)
    regime = regime_r1(3)
    search = Search(
        n_rollouts=1,
        max_depth=1,
        min_queries=1,
        tau_info=1.0,
        tau_stop=1.0,
        reward_tp=regime.reward_tp,
        reward_fp=regime.reward_fp,
        reward_tn=regime.reward_tn,
        reward_fn=regime.reward_fn,
        step_penalty=regime.step_penalty,
        repeat_penalty=-5.0,
        reward_threshold=regime.reward_threshold,
    )
    with torch.no_grad():
        logits = classifier(states)
    for label in (0, 1):
        ours = terminal_rewards(logits, torch.full((64,), float(label)), regime)
        assert np.allclose(ours.numpy(), trl_rewards(classifier, states, label, search))


def test_episode_returns_match_trl_rollout_returns() -> None:
    """A sampled episode's return equals what TRL's ``rollouts`` pays for the same path."""
    problem = toy_problem(regime_r1(3))
    model = problem.model()
    generator = torch.Generator().manual_seed(3)
    rows = torch.arange(40)
    episodes = sample_episodes(
        model.policy_, model.classifier_, problem.train, rows, problem, generator
    )
    regime = problem.regime
    search = Search(
        n_rollouts=1,
        max_depth=1,
        min_queries=regime.min_queries,
        tau_info=1.0,
        tau_stop=1.0,
        reward_tp=regime.reward_tp,
        reward_fp=regime.reward_fp,
        reward_tn=regime.reward_tn,
        reward_fn=regime.reward_fn,
        step_penalty=regime.step_penalty,
        repeat_penalty=-5.0,
        reward_threshold=regime.reward_threshold,
    )
    for i in range(40):
        sample = Sample(problem.train.x[i], problem.columns.numpy(), np.array(regime.costs))
        final = episodes.final_masks[i].numpy()[None, :]
        depth = int(final.sum())  # rollouts at depth == max_depth stop at once and pay depth steps
        paid, _ = rollouts(
            sample,
            final,
            depth,
            (model.policy_, model.classifier_),
            int(problem.train.y[i]),
            _with_depth(search, depth),
            np.random.default_rng(0),
        )
        assert episodes.returns[i].item() == pytest.approx(float(paid[0]), abs=1e-5)


def _with_depth(search: Search, depth: int) -> Search:
    fields = {name: getattr(search, name) for name in search.__dataclass_fields__}
    fields["max_depth"] = depth
    return Search(**fields)


def test_episodes_obey_the_rules() -> None:
    problem = toy_problem()
    problem.regime = Regime(
        "toy",
        1.0,
        -1.0,
        1.0,
        -1.0,
        step_penalty=-0.1,
        reward_threshold=0.5,
        min_queries=2,
        max_steps=3,
        costs=(0.2, 0.1, 0.1),
    )
    model = problem.model()
    generator = torch.Generator().manual_seed(4)
    rows = torch.arange(len(problem.train)).repeat(4)
    episodes = sample_episodes(
        model.policy_, model.classifier_, problem.train, rows, problem, generator
    )
    reveals = episodes.final_masks.sum(dim=1)
    assert (reveals >= 2).all() and (reveals <= 3).all()
    stops = (episodes.actions == 3) & episodes.valid
    assert (stops.sum(dim=1) <= 1).all()
    assert (episodes.masks.sum(dim=2)[stops] >= 2).all()  # STOP only after min_queries
    taken = episodes.actions.clone()
    taken[~episodes.valid | (taken == 3)] = -1
    for row in taken.tolist():  # never the same slot twice
        slots = [a for a in row if a >= 0]
        assert len(slots) == len(set(slots))
    assert torch.allclose(episodes.rewards.sum(dim=1), episodes.returns)
    expected, _ = returns_of_masks(
        model.classifier_,
        Split(problem.train.x[rows], problem.train.y[rows]),
        episodes.final_masks,
        problem,
    )
    assert torch.allclose(episodes.returns, expected, atol=1e-5)


def test_evaluate_uses_trl_deployment_and_scores_it() -> None:
    problem = toy_problem()
    model = problem.model()
    record = evaluate(model, problem.test, problem)
    masks, paths = model._reveal(problem.test.x.numpy(), None)
    expected, _ = returns_of_masks(
        model.classifier_, problem.test, torch.from_numpy(masks), problem
    )
    assert record["ret"] == pytest.approx(float(expected.mean()))
    assert record["n_slots"] == pytest.approx(np.mean([len(p) for p in paths]))


# ------------------------------------------------------------------ regimes and anchors


def test_r2_reward_is_youden_j_and_threshold_is_bayes() -> None:
    regime = regime_r2(4, prevalence=0.2, ba_all=0.8)
    # expected terminal reward of a classifier with TPR t and TNR s is t + s - 1
    for tpr, tnr in ((1.0, 1.0), (0.5, 0.5), (0.9, 0.7)):
        expected = 0.2 * (tpr * regime.reward_tp + (1 - tpr) * regime.reward_fn) + 0.8 * (
            tnr * regime.reward_tn + (1 - tnr) * regime.reward_fp
        )
        assert expected == pytest.approx(tpr + tnr - 1)
    # calling positive beats calling negative exactly when p > prevalence
    for p in (0.1, 0.19, 0.21, 0.6):
        call = p * regime.reward_tp + (1 - p) * regime.reward_fp
        hold = p * regime.reward_fn + (1 - p) * regime.reward_tn
        assert (call > hold) == (p > regime.reward_threshold)
    assert sum(regime.costs) == pytest.approx(0.5 * (2 * 0.8 - 1))
    assert regime.costs == tuple(sorted(regime.costs))


def test_r1_is_the_library_default() -> None:
    from think_reason_learn.verifiable_rl import VerifiableRL

    default, regime = VerifiableRL(), regime_r1(6)
    for name in (
        "reward_tp",
        "reward_fp",
        "reward_tn",
        "reward_fn",
        "step_penalty",
        "reward_threshold",
        "min_queries",
        "max_steps",
    ):
        assert getattr(regime, name) == getattr(default, name), name
    assert default.slot_costs is None and set(regime.costs) == {0.0}


def test_static_oracle_is_the_best_fixed_set() -> None:
    problem = toy_problem()
    classifier = problem.model().classifier_
    found = anchors(problem, classifier, n_random=5)
    assert found["static_set"] == [0]  # the signal slot alone: right every time, cost 0.2
    assert found["static_test"] == pytest.approx(1.0 - 0.2)
    assert found["clairvoyant_test"] >= found["static_test"] >= found["random_test"]
    assert len(subsets(3, problem.regime)) == 7


# ------------------------------------------------------------------ advantages


def test_group_advantage_is_the_deepseekmath_formula() -> None:
    returns = torch.tensor([1.0, 2.0, 3.0, 6.0, 10.0, 10.0, 10.0, 10.0])
    advantage = group_advantages(returns, 4, "group")
    first = returns[:4]
    assert torch.allclose(advantage[:4], (first - first.mean()) / (first.std(correction=0) + 1e-6))
    assert torch.allclose(advantage[4:], torch.zeros(4))  # a group that agrees teaches nothing
    assert advantage[:4].sum().abs() < 1e-5


def test_group_advantage_ignores_what_differs_between_samples() -> None:
    """Shifting or rescaling one sample's returns leaves its z-scored advantages unchanged."""
    torch.manual_seed(5)
    returns = torch.randn(3, 8)
    moved = returns.clone()
    moved[1] = moved[1] * 20 - 16
    assert torch.allclose(
        group_advantages(returns.reshape(-1), 8, "group"),
        group_advantages(moved.reshape(-1), 8, "group"),
        atol=1e-4,
    )
    # the mean-only variant keeps the scale: the rescaled sample now dominates
    plain = group_advantages(moved.reshape(-1), 8, "group_nostd").view(3, 8)
    assert plain[1].abs().mean() > 5 * plain[0].abs().mean()
    assert torch.allclose(plain.mean(dim=1), torch.zeros(3), atol=1e-5)


def test_leave_one_out_baseline_excludes_the_episode() -> None:
    returns = torch.tensor([0.0, 0.0, 0.0, 4.0])
    advantage = group_advantages(returns, 4, "group_loo")
    std = returns.std(correction=0)
    assert advantage[3].item() == pytest.approx(4.0 / (std.item() + 1e-6), rel=1e-4)
    assert advantage[0].item() == pytest.approx(-(4.0 / 3) / (std.item() + 1e-6), rel=1e-4)


def test_batch_advantage_uses_one_baseline_for_all() -> None:
    returns = torch.tensor([1.0, 2.0, 3.0, 6.0])
    assert torch.allclose(
        batch_advantages(returns, "batch"), (returns - 3.0) / (returns.std(correction=0) + 1e-6)
    )
    assert torch.allclose(
        batch_advantages(returns, "none"), returns / (returns.std(correction=0) + 1e-6)
    )


def test_gae_matches_hand_computation() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0], [5.0, 0.0, 0.0]])
    values = torch.tensor([[0.5, 1.0, 2.0], [4.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, True], [True, False, False]])
    advantage, targets = gae(rewards, values, valid, lam=1.0)  # lam 1: return-to-go minus value
    assert torch.allclose(advantage[0], torch.tensor([6.0 - 0.5, 5.0 - 1.0, 3.0 - 2.0]))
    assert advantage[1, 0].item() == pytest.approx(1.0)
    assert torch.allclose(targets[0], torch.tensor([6.0, 5.0, 3.0]))
    advantage, _ = gae(rewards, values, valid, lam=0.0)  # lam 0: one-step TD error
    assert torch.allclose(advantage[0], torch.tensor([1 + 1.0 - 0.5, 2 + 2.0 - 1.0, 3 - 2.0]))
    half, _ = gae(rewards, values, valid, lam=0.5)
    assert half[0, 0].item() == pytest.approx(1.5 + 0.5 * (3.0 + 0.5 * 1.0))


# ------------------------------------------------------------------ compute accounting


def test_meter_counts_rows_and_charges_training_three_times() -> None:
    network = Classifier(6, 16)
    meter = Meter()
    meter.attach("net", network)
    assert meter.macs["net"] == macs(network) == 6 * 16 + 16 * 8 + 8 * 8 + 8 * 1
    with torch.no_grad():
        network(torch.zeros(10, 6))
    network(torch.zeros(4, 6))
    with meter.paused(), torch.no_grad():
        network(torch.zeros(1000, 6))
    assert meter.forward_rows["net"] == 10 and meter.train_rows["net"] == 4
    assert meter.compute == macs(network) * (10 + 3 * 4)
    meter.detach()
    network(torch.zeros(4, 6))
    assert meter.train_rows["net"] == 4


# ------------------------------------------------------------------ the optimisers learn


def _train(problem: Problem, settings: PGSettings, updates: int) -> dict[str, float]:
    meter, count = Meter(), {"n": 0}

    def after(_model: object) -> bool:
        count["n"] += 1
        return count["n"] >= updates

    model = train_pg(problem, settings, meter, after)
    with meter.paused():
        return evaluate(model, problem.test, problem)


@pytest.mark.parametrize(
    ("advantage", "collect"),
    [
        ("group", "grouped"),
        ("group_nostd", "grouped"),
        ("group_loo", "grouped"),
        ("batch", "grouped"),
        ("gae", "single"),
        ("gae", "grouped"),
        ("gae_priv", "grouped"),
    ],
)
def test_every_arm_finds_the_known_optimum(advantage: str, collect: str) -> None:
    """The best policy reveals ``signal`` and stops: return 0.8. Each arm gets close."""
    problem = toy_problem()
    settings = PGSettings(
        advantage=advantage,
        collect=collect,
        lr=3e-3,
        epochs=2,
        minibatch=128,
        batch_episodes=256,
        group_size=8,
    )
    record = _train(problem, settings, updates=60)
    assert record["ret"] > 0.7, record
    assert record["first_0"] > 0.9


def test_trl_trainer_runs_under_the_budget_harness() -> None:
    from grpo_trl.run import run_arm

    problem = toy_problem()
    problem.info = {}
    record = run_arm(
        problem,
        "mc",
        {"n_rollouts": 2, "max_depth": 2, "update_every": 10, "policy_lr": 1e-2},
        budget=4e7,
        n_checkpoints=4,
    )
    final = record["final"]
    assert final["compute"] <= 4e7
    assert final["transitions"] > 0 and final["train_classifier"] == 0  # the classifier is frozen
    assert [c["fraction"] for c in record["checkpoints"]][-1] == 1.0
    assert record["final"]["test"]["ret"] > 0.5


def test_runs_are_reproducible() -> None:
    from grpo_trl.run import run_arm

    settings = {"advantage": "group", "lr": 1e-3, "batch_episodes": 64, "group_size": 4}
    first, second = (
        run_arm(_with_info(toy_problem()), "pg", settings, budget=2e6, n_checkpoints=2)
        for _ in range(2)
    )
    assert first["final"]["test"] == second["final"]["test"]
    assert first["final"]["compute"] == second["final"]["compute"]


def _with_info(problem: Problem) -> Problem:
    problem.info = {}
    return problem
