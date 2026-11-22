# Reproducing the study

Everything was run on one 4-core CPU machine (no GPU), Python 3.13, PyTorch 2.15. Each run is single-threaded and
deterministic given its seed; four runs execute in parallel.

## 1. Environment

```bash
uv venv --python 3.13 .venv
```

```bash
source .venv/bin/activate
```

```bash
uv pip install -e ".[dev]"
```

This installs Think-Reason-Learn from GitHub at the pinned commit `a712645` (v0.4.4), the version every result
here was produced with.

## 2. Data

Raw files go in `data/raw/` (not committed; `data/raw/SHA256SUMS` in this document lists their hashes).

| File | Source |
|---|---|
| `founders.jsonl` | `data/raw/full.jsonl` of `github.com/Sidi3355/trl-for-angels` |
| `kickstarter_train.parquet` | Hugging Face dataset `james-burton/kick_starter_funding`, `data/train-*.parquet` |
| `fakejobs_{train,val,test}.parquet` | Hugging Face dataset `james-burton/fake_job_postings2`, `data/{train,validation,test}-*.parquet` |
| `credit.csv` | Hugging Face dataset `Lancer73/uci-credit-card-default`, `UCI_Credit_Card.csv` |
| `adult.csv` | Hugging Face dataset `scikit-learn/adult-census-income`, `adult.csv` |

```
250e154ed75714ae57a564926d66c6319cd6aac1bcd32774cc76841a88d74e53  adult.csv
9e5c04a4331d1e7b7d11b5ce881f550d55c1932292eacfe3734dd4546888ddec  credit.csv
cc6099cc3ead2d77d22744e46e926bbf3a68a4c33b6d90d861821afccf6ae909  fakejobs_test.parquet
b9eec51aac65105d8c08d8ff444f0d163c4302b02dc30c765a00002993aba5d2  fakejobs_train.parquet
1bdda1f421f052dcf915b632baae74b95bb581b902eff8b99e0946d320237ffb  fakejobs_val.parquet
6861ba4172c6b5b4e8a306366dbece67905dabd917b3eef4296c9a8bc023da6c  founders.jsonl
41883ee66248ecb9ca90ce49a0f7b2c823b8f038c47ef30b6dde31800235e6ee  kickstarter_train.parquet
```

Build the slot-feature pools (embeds the text slots with `all-MiniLM-L6-v2`; about 15 minutes on CPU):

```bash
python -m grpo_trl.data
```

## 3. Tests

```bash
python -m pytest tests -q
```

35 tests: equivalence with TRL's action rule, rewards and rollout returns; hand-computed advantages and GAE; the
compute meter; every optimiser on a toy problem with a known optimum; the statistics on numbers with known
answers.

## 4. The stages, in order

Each stage writes one JSON record per run under `results/raw/<stage>/` and skips runs that already exist, so an
interrupted stage resumes where it stopped. The committed records let you skip straight to step 5.

```bash
python experiments/select_classifier.py synth-route   # once per dataset: the frozen classifier's settings
```

```bash
python experiments/launch.py prepare --seeds 1000 1001 1002
```

```bash
python experiments/launch.py budget     # C1: compute of one default TRL pass, per dataset and regime
```

```bash
python experiments/launch.py tune       # 24 configurations x 4 learners x 4 dev conditions
```

```bash
python experiments/tune_select.py top3
```

```bash
python experiments/launch.py tune2      # finalists on two more dev seeds
```

```bash
python experiments/tune_select.py final
```

`experiments/frozen_configs.json` is then written by hand from `tune_chosen.json` (it is committed). The
confirmatory stages:

```bash
python experiments/launch.py prepare --seeds 0 1 2 3 4 5 6 7 8 9
```

```bash
python experiments/launch.py e1         # main comparison: 5 arms x 12 conditions x seeds
```

```bash
python experiments/launch.py e2         # baseline ladder
```

```bash
python experiments/launch.py e3         # group size
```

```bash
python experiments/launch.py e4         # four times the budget
```

```bash
python experiments/e6_gradient_bias.py  # gradient-direction diagnostic (no training)
```

E5 runs in an environment with the *patched* library (see `contrib/trl/README.md`):

```bash
python experiments/e5_end_to_end.py
```

## 5. Tables and figures

```bash
python -m grpo_trl.report
```

writes `results/summary/TABLES.md`, the CSVs beside it and `results/figures/*.png`. Every number in `REPORT.md`
is read off those files.

## Approximate cost

| Stage | Runs | Wall-clock on 4 cores |
|---|---|---|
| tune + tune2 | 488 | about 90 minutes |
| e1 (10 seeds) | 600 | about 80 minutes |
| e2 | 360 | about 45 minutes |
| e3 | 200 | about 25 minutes |
| e4 | 60 | about 40 minutes |
| e5 | 60 | about 30 minutes |
| e6 | 36 | about 5 minutes |
