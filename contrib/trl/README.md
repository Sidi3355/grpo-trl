# Proposed change to Think-Reason-Learn: `VerifiableRL(optimizer="group")`

A TRL-native implementation of the critic-free optimiser this study tested end to end, written to TRL's
contribution rules (`CONTRIBUTING.md`): plain constructor parameters stored unchanged, default behaviour
untouched, tests beside the existing ones, a changelog entry.

> **Status: a starting point for discussion, not a change to merge as is.** The study's recommendation is PPO
> (see [`REPORT.md`](../../REPORT.md), section 7). This patch implements the critic-free variant, which is the
> smaller change and did well at the study's standard budget and in the end-to-end check, but which collapsed
> late in training when given four times that budget in 8 of 12 runs. Read "Known limitation" below first.

| File | What |
|---|---|
| [`verifiable-rl-group-optimizer.patch`](verifiable-rl-group-optimizer.patch) | The whole change, against `Vela-Research/think-reason-learn` at `a712645` (v0.4.4): 5 files, 437 added lines, 4 changed |
| [`_group.py`](_group.py) | The new module, for reading without applying the patch |
| [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md) | The pull-request text, in TRL's template |
| [`ISSUE.md`](ISSUE.md) | A short note for TRL's maintainers: the two main findings and two questions. Send this first |

## What it adds

```python
from think_reason_learn.verifiable_rl import VerifiableRL

model = VerifiableRL(slots={"education": 384, "jobs": 384}, optimizer="group")
```

- `optimizer="rollout"` (the default, the paper's method) is exactly what the library does today.
- `optimizer="group"` learns from whole episodes. For each training sample the policy plays `group_size`
  episodes. An episode's advantage is its return minus the mean return of its group; the policy is improved on
  PPO's clipped objective. It is GRPO without the division by the group's standard deviation, and without a
  critic.
- The return an episode is scored by includes slot costs and the step penalty, so the policy learns when to
  stop. The rollout targets cannot: STOP is fitted to the reward of stopping now.
- The classifier is co-trained exactly as before, on the episodes' final states, with the same replay buffer,
  freeze period and target copy.
- New parameters: `optimizer`, `group_size` (32), `group_batch` (2048 episodes per update), `group_epochs` (4),
  `group_lr` (1.5e-3), `clip_range` (0.3). The defaults are the configuration frozen in this study before the
  confirmatory runs. `policy_batch`, `grad_clip`, `n_iterations`, `min_queries`, `max_steps`, the rewards and the
  classifier settings are shared with the rollout optimiser.
- **Two shared settings are worth changing with it.** The study's configuration used `policy_batch=256` and
  `grad_clip=0.5`; the library's defaults for those (128 and 5.0) belong to the rollout optimiser and are left
  alone. With them untouched the new optimiser is a little weaker (report section 4.7). The docstring says so.
- The LLM supervisor advises only at prediction time under `optimizer="group"` (training samples its actions
  from the policy itself).
- No new dependency. Saving, loading, pickling, `clone` and the supervisor are unaffected.

## How it was checked

In a clone of the upstream repository at `a712645`, with the environment built from TRL's own lock file
(`poetry install --with dev --all-extras`), exactly as its CI does:

| Check | Result |
|---|---|
| `poetry run ruff check .` | passes |
| `poetry run ruff format --check .` | passes |
| `poetry run pyright` (strict) | 0 errors |
| `poetry run pytest` (offline suite) | 494 passed, 2 skipped |

Seven tests are added to `tests/verifiable_rl/test_verifiable_rl.py`:

- the scikit-learn contract (constructor, fit, predict, pickle, save and load) under `optimizer="group"`;
- it learns to stop: with one informative slot and costly others, at least 90% of decision paths are that slot
  alone;
- fits are deterministic and leave PyTorch's global random state untouched;
- an unknown `optimizer` is refused before any work;
- played episodes follow the rules (never a slot twice, STOP only after `min_queries`) and their returns equal
  what TRL's own `rollouts` pays for the same path;
- advantages are centred per sample and share one scale;
- `action_log_probs` ranks actions as `greedy` does, and survives a saturated STOP.

## Applying it

In a clone of `Vela-Research/think-reason-learn` (or a fork of it):

```bash
git checkout -b feat/verifiable-rl-group-optimizer a712645
```

```bash
git apply --index /path/to/grpo-trl/contrib/trl/verifiable-rl-group-optimizer.patch
```

```bash
poetry run pytest tests/verifiable_rl
```

The patch was made against `a712645`; `VerifiableRL` had not changed between v0.3.0 and that commit.

## The evidence behind it

[`REPORT.md`](../../REPORT.md). In one paragraph: on six datasets, two reward regimes and ten seeds, at equal
compute, this optimiser scores 1.01 on a scale where 0 is a random policy and 1 the best non-adaptive policy,
against 0.60 for the rollout optimiser tuned with the same effort and 0.49 as shipped. PPO, which needs a critic
network, scores 1.10. Canonical GRPO, which divides by the group's standard deviation, scores 0.60 and should not
be offered. Section 4.7 of the report runs this patch end to end, through `VerifiableRL.fit`, with the classifier
co-trained; sections 4.6 and 4.8 are the four-times-budget result and its follow-up.

## Known limitation

Trained for much longer than the study's standard budget (about 12 policy updates at these defaults), the
optimiser reaches its best policy and then gets worse: at four times the budget, with the classifier frozen, 8
of 12 runs ended more than 0.3 normalised units below their own best checkpoint, and it finished behind PPO
(0.95 against 1.42) and level with the tuned rollout optimiser (0.87). One pass over 2,000 samples at these
defaults is about 31 updates, so the default sits closer to that region than is comfortable. In the end-to-end
check, where the classifier is co-trained, one pass did well on both datasets tried.

One suspected cause was tested after the fact: the batch-wide normaliser of the advantages shrinks as the policy
converges, so the step never does. Dividing by the batch standard deviation of the returns instead cost nothing
at the standard budget and helped a little at four times it (1.12), without curing the collapse (6 of 12 runs).
That variant is `advantage="group_rawscale"` in the study's `grpo_trl/pg.py`; this patch keeps the variant the
end-to-end check ran. An entropy bonus, a lower learning rate or a validation-based stopping rule were not
tried.

## What a maintainer might reasonably want instead

- **PPO.** Best at both budgets and stable. The study's implementation is `grpo_trl/pg.py` (`advantage="gae"`);
  porting it means one more network (a critic the size of the classifier), GAE, and three more settings.
- **A smaller default `group_size`.** Group sizes from 2 to 32 score the same within noise (report section 4.5);
  a smaller group makes a pass over the samples cheaper.
- **The name.** `"group"` describes what it does; `"grpo"` would be more searchable but is not accurate, since the
  standard-deviation step is removed on purpose.
