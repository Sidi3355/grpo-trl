# grpo-trl

**Is GRPO a better policy optimiser than PPO for Think-Reason-Learn? A pre-registered study.**

[Think-Reason-Learn](https://github.com/Vela-Research/think-reason-learn) (TRL) is Vela Research's open-source
library of scikit-learn classifiers that reason with language models. One of its six models, `VerifiableRL`,
uses reinforcement learning: a policy decides which piece of information about a sample to reveal next, or to
stop, and a classifier predicts from what was revealed. This repository asks whether Group Relative Policy
Optimization (GRPO) would train that policy better than Proximal Policy Optimization (PPO).

> "TRL" here always means Think-Reason-Learn, not Hugging Face's Transformer Reinforcement Learning library.

**Start with [`REPORT.md`](REPORT.md).**

## The result in six lines

1. **TRL does not use PPO.** `VerifiableRL` trains its policy by distilling Monte Carlo rollout targets. So the
   study compares three things: what TRL ships, PPO, and GRPO.
2. **GRPO as published is not better than PPO; it is clearly worse** (-0.50 normalised units, 95% CI -0.59 to
   -0.41), and no better than what TRL ships.
3. **One change repairs GRPO for a short run: do not divide by the group's standard deviation.** That
   "mean-centred" variant is within 0.09 of PPO at the standard budget, **but it collapses when trained four
   times as long**, and a fix aimed at the diagnosed cause only partly helped.
4. **PPO clearly beats TRL's shipped optimiser** at equal compute, at both budgets tried (+0.50 and +0.55), in
   about half the wall-clock time. It is the robust choice.
5. **TRL's optimiser cannot learn when to stop**, and as shipped it does not improve with four times the budget.
6. **For TRL:** offer a whole-episode policy-gradient optimiser, and make it PPO. A tested TRL-native
   implementation of the critic-free variant is in [`contrib/trl`](contrib/trl/README.md) as a starting point.

| Optimiser | One budget: 12 conditions, 10 seeds | Four budgets: 4 conditions, 3 seeds |
|---|---|---|
| PPO | **1.10** [1.02, 1.17] | **1.42** [1.24, 1.60] |
| GRPO, mean-centred | 1.01 [0.93, 1.09] | 0.95 [0.63, 1.23] |
| TRL shipped, tuned | 0.60 [0.48, 0.71] | 0.87 [0.58, 1.18] |
| GRPO, canonical | 0.60 [0.50, 0.70] | 0.75 [0.46, 1.03] |
| TRL shipped, defaults | 0.49 [0.39, 0.60] | 0.37 [0.12, 0.61] |

Normalised test return: 0 = a random policy, 1 = the best policy that does not adapt to what it sees. Mean with
95% CI. One budget = the compute of one default TRL pass over the training set.

![Main results](results/figures/fig1_main_results.png)

## How the study was run

- **Pre-registered.** Hypotheses, metrics, statistics and the decision rule were committed before any
  experiment; every later change is a dated amendment made before the confirmatory seeds were run
  ([`PLAN.md`](PLAN.md), [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)).
- **Only the optimiser varies.** Networks, action rule, rewards and the deployment loop are imported from the
  pinned library; the incumbent arm is TRL's own trainer code.
- **Equal compute, equal tuning.** Every arm gets the compute of one default TRL pass, counted on every network;
  every learner gets the same random search on separate dev seeds.
- **Paired and pooled statistics.** Same seeds for every arm; stratified bootstrap CIs, permutation tests, Holm
  correction.
- **Tested.** 35 tests check equivalence with TRL's action rule, rewards and rollout returns, the advantage
  formulas, the compute meter and the statistics.

## What is where

| Path | What |
|---|---|
| [`REPORT.md`](REPORT.md) | The write-up: question, method, results, limitations, recommendation |
| [`PLAN.md`](PLAN.md) | The pre-registration and its amendments |
| [`docs/METHODS.md`](docs/METHODS.md) | The method in full, with code pointers |
| [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) | Every departure from the plan |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Commands to reproduce everything |
| [`results/summary/TABLES.md`](results/summary/TABLES.md) | Every table, generated from the raw records |
| `results/raw/` | One JSON record per run |
| `results/figures/` | Figures |
| `grpo_trl/` | The test bed: decision problem, compute meter, optimisers, data, analysis |
| `experiments/` | Launchers, budgets, tuning records, frozen configurations |
| [`contrib/trl`](contrib/trl/README.md) | The proposed change to TRL: patch, tests, pull-request text |
| `tests/` | The tests |

## Quick start

```bash
uv venv --python 3.13 .venv
```

```bash
source .venv/bin/activate
```

```bash
uv pip install -e ".[dev]"
```

```bash
python -m pytest tests -q
```

```bash
python -m grpo_trl.report
```

The last command rebuilds every table and figure from the committed run records.

## Scope and honesty notes

- The founder data is generated (from [`trl-for-angels`](https://github.com/Sidi3355/trl-for-angels)); VCBench
  was not available. Nothing here is a claim about real founders.
- Ten seeds resolve pooled differences of about 0.17 and larger. The four-budget, end-to-end and post hoc
  experiments are small (three to five seeds on four conditions).
- The `VerifiableRL` paper could not be retrieved; the incumbent is the library code at commit `a712645`.
