"""Group-relative policy optimisation: whole episodes, compared within a sample.

For each training sample the policy plays a group of episodes. An episode's
advantage is its return minus the mean return of its group, so whatever the
group shares (the label, the slots nobody revealed) cancels, and no value
network is needed. The policy is then improved on PPO's clipped objective
(Schulman et al., 2017). This is GRPO (Shao et al., 2024, arXiv:2402.03300)
with one change: the advantage is not divided by the group's standard deviation.
With slot costs that division weighs a sample whose episodes differ only by a
cost as much as one whose episodes differ by a right or a wrong call, and the
policy learns to stop at once.

Unlike the rollout targets, the return an episode is scored by includes the slot
costs and the step penalty, so STOP is learned from what continuing is worth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from ._episodes import Masks, Search, allowed_actions, policy_probs, sample_actions
from ._networks import Classifier, PolicyNet, action_log_probs
from ._torch import shuffled_batches, tensor


@dataclass(frozen=True, slots=True)
class Episodes:
    """Episodes the policy played, padded to the longest one allowed."""

    samples: npt.NDArray[np.intp]
    """The sample of each episode."""
    masks: Masks
    """Slots revealed before each decision, shape ``(episodes, steps, slots)``."""
    actions: npt.NDArray[np.intp]
    log_probs: npt.NDArray[np.float64]
    """Log-probability of each action when it was taken."""
    decided: npt.NDArray[np.bool_]
    """Whether a decision was made at each step."""
    returns: npt.NDArray[np.float64]
    finals: Masks
    """Slots revealed when each episode ended."""


def play(
    features: npt.NDArray[np.float32],
    positive: npt.NDArray[np.float32],
    samples: npt.NDArray[np.intp],
    layout: tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]],
    networks: tuple[PolicyNet, Classifier],
    search: Search,
    max_steps: int,
    rng: np.random.Generator,
) -> Episodes:
    """Play one episode per entry of ``samples``, sampling from the policy.

    An episode reveals unseen slots until it picks STOP (allowed after
    ``min_queries``), has revealed every slot, or has revealed ``max_steps``.
    Its return is the reward of the classifier's call on what was revealed,
    minus the revealed slots' costs, plus the step penalty per reveal.

    Args:
        features: Every training sample's features.
        positive: 1 for each positive sample, else 0.
        samples: The sample of each episode.
        layout: The slot of each feature column, and each slot's cost.
        networks: The policy and the classifier that scores the final call.
        search: The reward settings and ``min_queries``.
        max_steps: Most slots revealed per episode.
        rng: Draws the actions.
    """
    (columns, costs), (policy, classifier) = layout, networks
    device = next(policy.parameters()).device
    n, n_slots = len(samples), len(costs)
    seen = np.zeros((n, n_slots), dtype=bool)
    masks = np.zeros((n, max_steps, n_slots), dtype=bool)
    actions = np.zeros((n, max_steps), dtype=np.intp)
    log_probs = np.zeros((n, max_steps))
    decided = np.zeros((n, max_steps), dtype=bool)
    active = np.ones(n, dtype=bool)
    for step in range(max_steps):
        rows = np.flatnonzero(active)
        if not len(rows):
            break
        allowed = allowed_actions(seen[rows], search.min_queries)
        states = tensor(features[samples[rows]] * seen[rows][:, columns], device)
        # the smallest weight keeps an allowed action possible when STOP saturates
        probs = np.where(allowed, policy_probs(policy, states) + 1e-12, 0.0)
        probs /= probs.sum(axis=1, keepdims=True)
        chosen = sample_actions(probs, rng)
        masks[rows, step], actions[rows, step], decided[rows, step] = (
            seen[rows],
            chosen,
            True,
        )
        log_probs[rows, step] = np.log(probs[np.arange(len(rows)), chosen])
        reveal = chosen < n_slots
        seen[rows[reveal], chosen[reveal]] = True
        revealed = seen[rows].sum(axis=1)
        active[rows[~reveal | (revealed >= min(n_slots, max_steps))]] = False
    finals = tensor(features[samples] * seen[:, columns], device)
    labels = positive[samples] > 0.5
    with torch.no_grad():
        called = (
            (torch.sigmoid(classifier(finals)) >= search.reward_threshold).cpu().numpy()
        )
    call_rewards = np.where(
        labels,
        np.where(called, search.reward_tp, search.reward_fn),
        np.where(called, search.reward_fp, search.reward_tn),
    )
    returns = call_rewards - seen @ costs + search.step_penalty * seen.sum(axis=1)
    return Episodes(samples, masks, actions, log_probs, decided, returns, seen)


def advantages(
    returns: npt.NDArray[np.float64], group_size: int
) -> npt.NDArray[np.float64]:
    """Each episode's return minus its group's mean, on one scale for the batch.

    ``returns`` holds the groups one after another. The centred returns are
    divided by their standard deviation over the whole batch, so the step size
    does not depend on the reward's units while samples keep their relative
    weight.
    """
    grouped = returns.reshape(-1, group_size)
    centred = (grouped - grouped.mean(axis=1, keepdims=True)).reshape(-1)
    return centred / (centred.std() + 1e-6)


def improve(
    policy: PolicyNet,
    optimiser: torch.optim.Optimizer,
    states: torch.Tensor,
    episodes: Episodes,
    advantage: npt.NDArray[np.float64],
    settings: tuple[int, float, int, int, float],
    shuffle: torch.Generator,
) -> None:
    """Raise the probability of the better episodes' actions, within a clip range.

    Every decision of an episode gets the episode's advantage, weighted by one
    over the episode's length, as in GRPO.

    Args:
        policy: The policy to improve.
        optimiser: Its optimiser.
        states: The state before each decision, one row per decision.
        episodes: The episodes the decisions come from.
        advantage: One advantage per episode.
        settings: ``min_queries``, the clip range, epochs, batch size and the
            gradient-norm clip.
        shuffle: Orders the batches.
    """
    min_queries, clip_range, epochs, batch_size, grad_clip = settings
    episode, step = np.nonzero(episodes.decided)
    allowed = tensor(
        allowed_actions(episodes.masks[episode, step], min_queries), states.device
    )
    actions = tensor(episodes.actions[episode, step], states.device)
    old = tensor(episodes.log_probs[episode, step], states)
    weight = tensor(1.0 / episodes.decided.sum(axis=1)[episode], states)
    gain = tensor(advantage[episode], states)
    for _ in range(epochs):
        for batch in shuffled_batches(len(episode), batch_size, shuffle):
            log_probs = action_log_probs(policy(states[batch]), allowed[batch])
            taken = log_probs.gather(1, actions[batch, None]).squeeze(1)
            ratio = torch.exp(taken - old[batch])
            clipped = ratio.clamp(1 - clip_range, 1 + clip_range)
            surrogate = torch.minimum(ratio * gain[batch], clipped * gain[batch])
            loss = -(weight[batch] * surrogate).sum() / weight[batch].sum()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()  # pyright: ignore[reportUnknownMemberType]  # partial annotation
            if grad_clip:
                nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimiser.step()
