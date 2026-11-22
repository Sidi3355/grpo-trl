# feat(verifiable_rl): learn the policy from whole episodes, compared within each sample

## Summary

Adds a second way for `VerifiableRL` to train its policy: `VerifiableRL(..., optimizer="group")`. The default,
`optimizer="rollout"`, is the paper's method and is unchanged; nothing changes for existing code.

**What it does.** For each training sample the policy plays `group_size` whole episodes. An episode's advantage
is its return minus the mean return of its group, so whatever the group shares (the label, the slots nobody
revealed) cancels and no value network is needed. The policy is then improved on PPO's clipped objective. This is
GRPO (Shao et al., 2024) without its division by the group's standard deviation.

**Why.** Two properties of the rollout targets limit what the policy can learn:

- STOP is fitted to the reward of stopping *now*, so slot costs and the step penalty shape which slot is asked,
  never when to stop (as the docstring of `targets` says). In a study on six datasets, with a per-slot cost and
  STOP allowed after one slot, the rollout optimiser revealed exactly its minimum number of slots everywhere.
- Each visited state costs `n_slots * n_rollouts` rollouts to label.

Scoring whole episodes by their actual return fixes the first and removes the second.

**Evidence.** A pre-registered comparison (plan, hypotheses and analysis committed before any run; six
datasets, two reward regimes including the library's defaults, ten seeds; identical networks, rewards and
deployment code; equal compute counted on every network; equal tuning effort):

| Optimiser | Normalised test return (0 = random policy, 1 = best non-adaptive policy), 95% CI |
|---|---|
| `rollout`, library defaults | 0.49 [0.39, 0.60] |
| `rollout`, tuned | 0.60 [0.48, 0.71] |
| `group` (this PR) | 1.01 [0.93, 1.09] |
| PPO, for reference (needs a critic; not in this PR) | 1.10 [1.02, 1.17] |
| GRPO as published, for reference (not in this PR) | 0.60 [0.50, 0.70] |

- `group` minus tuned `rollout`: +0.41 [+0.30, +0.53], 9 of 12 conditions won, 1 lost; with the library's default
  rewards alone, +0.46 [+0.35, +0.56].
- At equal compute it finishes in about half the wall-clock time.
- Runs ending below a random policy: 8% against 21%.
- {{E5_SENTENCE}}

Two things the study found that are worth saying plainly:

- **PPO did slightly better** (+0.09, not significant after correction for multiple tests). This PR proposes the
  critic-free method because it is the smaller change. If you would rather have PPO, the study's implementation
  ports directly.
- **GRPO as published should not be offered here.** Dividing by each group's standard deviation gives a sample
  whose episodes differ only by a slot cost the same weight as one whose episodes differ by a right or wrong
  call; its update is close to orthogonal to the policy gradient (mean cosine 0.24 against 0.99 without the
  division) and it learns to stop as early as allowed.

Full write-up, raw run records and the code that produced them: the `grpo-trl` repository (`REPORT.md`).

**What changes in the code.**

- `verifiable_rl/_group.py` (new): `play` (a batch of whole episodes), `advantages`, `improve`.
- `verifiable_rl/_networks.py`: `action_log_probs`, the log-space form of `action_probs` over allowed actions.
- `verifiable_rl/_verifiable_rl.py`: six constructor parameters (`optimizer`, `group_size`, `group_batch`,
  `group_epochs`, `group_lr`, `clip_range`) and a branch in `_Trainer.run`. The classifier is co-trained as
  before, on the episodes' final states.
- With `optimizer="group"` the supervisor advises only when predicting.

## Type of change
- [x] feat
- [ ] fix
- [ ] refactor
- [ ] docs
- [ ] test
- [ ] chore

## Checklist
- [x] Tests added/updated (seven tests in `tests/verifiable_rl/test_verifiable_rl.py`)
- [x] Lint and type-check pass (`ruff check`, `ruff format --check`, `pyright` strict: 0 errors, under the lock file)
- [x] Docs updated (docstrings; changelog entry under "Unreleased")

## Related issues
None.
