"""The incumbent: TRL's own rollout-distillation trainer, instrumented.

``BudgetedTrainer`` subclasses the library's ``_Trainer``. Its ``_episode``,
``_explore``, ``_update``, ``_train_policy`` and the ``targets``/``rollouts``
they call are TRL's code, unmodified. Only ``run`` is restated (it is seven
lines in the library) so that it can stop on a compute budget instead of
after ``n_iterations`` passes, and call back after every network update.

The classifier is frozen by TRL's own switches: ``freeze_clf_updates`` is set
beyond reach, so ``_train_classifier`` never runs, and
``clf_target_update_every=0`` makes the rollouts use the live classifier.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from think_reason_learn.verifiable_rl import _verifiable_rl as trl
from think_reason_learn.verifiable_rl._verifiable_rl import VerifiableRL, _Trainer

from .meter import Meter
from .problem import Problem

FROZEN = {"freeze_clf_updates": 10**12, "clf_target_update_every": 0, "pretrain_epochs": 0}


class BudgetedTrainer(_Trainer):
    """TRL's trainer, stopping on a budget and reporting after each update."""

    def run_budgeted(self, meter: Meter, after_update: Callable[[], bool]) -> None:
        """TRL's ``_Trainer.run``, passing over the samples until told to stop.

        ``after_update`` is called after every network update and returns
        whether to stop.
        """
        model = self.model
        self.shuffle.manual_seed(int(self.rng.integers(2**31)))
        optimisers = (
            torch.optim.Adam(model.policy_.parameters(), lr=model.policy_lr),
            torch.optim.Adam(model.classifier_.parameters(), lr=model.clf_lr),
        )
        n_slots, k = len(model.slot_names_), model.n_rollouts
        original = trl.targets

        def counted(*args: Any, **kwargs: Any) -> Any:
            # one call scores one visited state with n_slots * k rollouts, each
            # opened by a forced reveal (the decisions after it are policy rows)
            meter.transitions += n_slots * k
            meter.episodes += n_slots * k
            return original(*args, **kwargs)

        trl.targets = counted
        try:
            while True:
                order = self.rng.permutation(len(self.features))
                for count, i in enumerate(order, 1):
                    before = meter.forward_rows["policy"]
                    self._episode(int(i), model.classifier_)
                    # every policy row in an episode is one simulated decision
                    meter.transitions += meter.forward_rows["policy"] - before
                    meter.episodes += 1
                    if count % model.update_every == 0:
                        self._update(optimisers, None)
                        if after_update():
                            return
                if len(self.features) % model.update_every:
                    self._update(optimisers, None)
                    if after_update():
                        return
        finally:
            trl.targets = original


def train_mc(
    problem: Problem,
    settings: dict[str, Any],
    meter: Meter,
    after_update: Callable[[VerifiableRL], bool],
) -> VerifiableRL:
    """Train the policy with TRL's trainer under ``settings`` (its own arguments).

    ``min_queries_frac``, if given, is the one setting that is not a TRL
    argument: where the regime lets STOP be chosen after one slot (R2), the
    incumbent may instead hold itself to ``round(frac * n_slots)`` slots, in
    training and in deployment, which is how a TRL user would set
    ``min_queries``/``predict_min_queries``. It applies in R2 only; in R1
    ``min_queries`` is the library default for every arm.
    """
    settings = dict(settings)
    frac = settings.pop("min_queries_frac", None)
    if frac is not None and problem.regime.name == "R2":
        floor = max(problem.regime.min_queries, round(frac * problem.n_slots))
        floor = min(floor, problem.regime.max_steps)
        settings.update(min_queries=floor, predict_min_queries=floor)
    model = problem.model(**{**settings, **FROZEN})
    meter.attach("policy", model.policy_)
    meter.attach("classifier", model.classifier_)
    rng = np.random.default_rng(problem.seed + 104729)
    trainer = BudgetedTrainer(
        model, problem.train.x.numpy(), problem.train.y.numpy(), None, rng
    )
    trainer.run_budgeted(meter, lambda: after_update(model))
    return model
