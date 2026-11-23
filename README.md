# grpo-trl

**Can PPO or GRPO train Think-Reason-Learn's `VerifiableRL` better than the optimiser it ships with? A
pre-registered study.**

[Think-Reason-Learn](https://github.com/Vela-Research/think-reason-learn) (TRL) is Vela Research's library of
scikit-learn classifiers that reason with language models. Its `VerifiableRL` model learns which pieces of
information about a sample to reveal before predicting, and trains that policy by distilling Monte Carlo rollout
targets. This repository tests two policy-gradient methods, PPO and GRPO, against that shipped optimiser, on
TRL's own decision problem, with everything but the optimiser held fixed.

> "TRL" here always means Think-Reason-Learn, not Hugging Face's Transformer Reinforcement Learning library.

## Read

| | |
|---|---|
| **[`paper/paper.pdf`](paper/paper.pdf)** | **The paper, 8 pages. Start here.** Source: [`paper.tex`](paper/paper.tex). Word: [`paper.docx`](paper/paper.docx) |
| [`REPORT.md`](REPORT.md) | The long-form write-up, with every table |
| [`contrib/trl/ISSUE.md`](contrib/trl/ISSUE.md) | A short note for TRL's maintainers |
| [`results/report.html`](results/report.html) | The same findings as a web page |

## Result

1. **PPO clearly beats TRL's shipped optimiser** at equal compute, at both budgets tried (+0.50 and +0.55), in
   about half the wall-clock time.
2. **TRL's optimiser cannot learn when to stop**, and as shipped it does not improve with four times the budget.
3. **GRPO as published does not beat TRL's optimiser**: level with it overall, worse where stopping matters, and
   0.50 behind PPO.
4. **One change repairs GRPO for a short run** (do not divide by the group's standard deviation): +0.41 over
   TRL's optimiser and within 0.09 of PPO at the standard budget. **It collapses when trained four times as
   long.**
5. **For TRL:** offer a whole-episode policy-gradient optimiser, and make it PPO.

| Optimiser | One budget: 12 conditions, 10 seeds | Four budgets: 4 conditions, 3 seeds |
|---|---|---|
| TRL shipped, defaults | 0.49 [0.39, 0.60] | 0.37 [0.12, 0.61] |
| TRL shipped, tuned | 0.60 [0.48, 0.71] | 0.87 [0.58, 1.18] |
| PPO | **1.10** [1.02, 1.17] | **1.42** [1.24, 1.60] |
| GRPO, canonical | 0.60 [0.50, 0.70] | 0.75 [0.46, 1.03] |
| GRPO, mean-centred | 1.01 [0.93, 1.09] | 0.95 [0.63, 1.23] |

Normalised test return, mean with 95% CI: 0 = a random policy, 1 = the best policy that does not adapt to what
it sees. One budget = the compute of one default TRL pass over the training set.

![Main results](results/figures/fig1_main_results.png)

## How it was run

- **Only the optimiser varies.** Networks, action rule, rewards and the deployment loop are imported from the
  pinned library (v0.4.4); the incumbent arm is TRL's own trainer code.
- **Equal compute and equal tuning** for every method, on six datasets and two reward regimes.
- **Pre-registered.** Hypotheses, metrics, statistics and a decision rule were committed before any experiment;
  every later change is a dated amendment ([`PLAN.md`](PLAN.md), [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)).
- **Paired, pooled statistics** with bootstrap intervals, permutation tests and Holm correction.
- **Tested.** 35 tests check equivalence with TRL's code, the advantage formulas, the compute meter and the
  statistics.

## Where things are

| Path | What |
|---|---|
| `paper/` | The paper: LaTeX source, PDF, Word, figures, bibliography |
| `REPORT.md`, `PLAN.md` | The long-form write-up; the pre-registration and its amendments |
| `docs/` | Method in full, deviations, how to reproduce, page sources |
| `results/` | One JSON record per run (`raw/`), generated tables (`summary/`), figures |
| `grpo_trl/` | The test bed: decision problem, compute meter, optimisers, data, analysis |
| `experiments/` | Launchers, budgets, tuning records, frozen configurations |
| `contrib/trl/` | For TRL's maintainers: issue text, a TRL-native patch with tests, pull-request text |
| `tests/` | The tests |

## Reproduce

```bash
uv venv --python 3.13 .venv && source .venv/bin/activate && uv pip install -e ".[dev]"
```

```bash
python -m pytest tests -q
```

```bash
python -m grpo_trl.report
```

The last command rebuilds every table and figure from the committed run records. Re-running the experiments
themselves is described in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Read this before citing it

- **Where the question came from.** The study began as "TRL uses PPO; would GRPO be better?", which is where the
  repository's name comes from. A code audit found TRL does not use PPO, so both became challengers to the
  optimiser it does ship. The hypotheses were registered around GRPO and are reported as registered.
- **PPO against TRL's optimiser was not a registered test.** It is the clearest result and is reported as
  descriptive, with unadjusted intervals.
- **It is small.** 2,000 training samples, ten seeds; three to five seeds for the longer-budget, end-to-end and
  follow-up experiments.
- **Data.** The founder profiles are generated (from
  [`trl-for-angels`](https://github.com/Sidi3355/trl-for-angels)); VCBench was not available. Nothing here is a
  claim about real founders.
- **The `VerifiableRL` paper could not be retrieved;** the incumbent is the library code at commit `a712645`.
- **AI assistance.** The experiments, code and drafts were produced with an AI coding assistant under the
  author's direction.
