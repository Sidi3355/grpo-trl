# Methods

The full method, in the order a run executes it. `PLAN.md` is the pre-registration; this file says exactly
what the code does. Section numbers in brackets point to the code.

## 1. What is held fixed, and what varies

Everything that defines the task is **imported from Think-Reason-Learn** at commit `a712645` (v0.4.4) and is the
same for every arm:

| Fixed | Source |
|---|---|
| Policy network (MLP, hidden 512) and classifier network (MLP, hidden 256) | `think_reason_learn.verifiable_rl._networks` |
| Network initialisation | `_torch.seeded` |
| Action rule: softmax over slots, sigmoid for STOP, restricted to allowed actions | `_networks.action_probs`, `_episodes.allowed_actions` |
| Reward of a final call | `_episodes.rewards` semantics (`reward_tp/fp/tn/fn`, `reward_threshold`) |
| Deployment: greedy reveal, never a slot twice | `VerifiableRL._reveal`, called unchanged |
| Classifier training | `VerifiableRL._pretrain` (random-reveal curriculum), then frozen |

**Only the policy optimiser varies.** Tests in `tests/test_testbed.py` check, state by state, that the policy the
new optimisers sample from has the same argmax as TRL's deployment rule, that the terminal rewards equal TRL's
`rewards`, and that an episode's return equals what TRL's `rollouts` pays for the same path.

## 2. The decision problem [`grpo_trl/problem.py`]

A sample has features in `S` named slots and a binary label. An episode starts with nothing revealed. At each
step the policy picks an unseen slot to reveal, or STOP (once `min_queries` slots are revealed). The episode ends
at STOP, when every slot is revealed, or after `max_steps` reveals. Its return is

    R = r(classifier's call on what was revealed, label) - sum of the revealed slots' costs + step_penalty * reveals

The state is the feature vector with unrevealed slots set to zero, as in TRL.

### Two reward regimes

| | R1: TRL default | R2: cost-aware |
|---|---|---|
| Right call | TP +4, TN 0 | `+1 / (2 * prevalence of the true class)` |
| Wrong call | FP -16, FN -0.25 | `-1 / (2 * prevalence of the true class)` |
| Call threshold | 0.3 | training prevalence (the Bayes threshold for this reward) |
| Step penalty | -0.1 | 0 |
| Slot costs | 0 | `0.5 * (2*BA_all - 1) * w_j`, `w = linspace(0.5, 1.5, S) / S` in slot order |
| `min_queries` / `max_steps` | 3 / 5 | 1 / S |

R1 is the library exactly as shipped. In R2 the expected terminal reward is Youden's J (`TPR + TNR - 1`), so the
return is "balanced accuracy, minus what the information cost"; revealing everything costs half of what full
information is worth (`BA_all` is the frozen classifier's validation balanced accuracy with all slots revealed),
and costs are assigned by slot position, not by informativeness. Both regimes use only existing `VerifiableRL`
arguments.

### The frozen classifier

The reward depends on the classifier, so a classifier that changes during training changes the problem. For each
(dataset, seed) the classifier is trained once with TRL's own `_pretrain` and frozen. For the incumbent this uses
TRL's own switches (`freeze_clf_updates` out of reach, `clf_target_update_every=0`). `_pretrain`'s three settings
were chosen per dataset by a rule fixed in advance (`experiments/select_classifier.py`): lowest validation log
loss under random reveals on dev seeds, ties to the fewest gradient steps. The end-to-end check (E5) removes this
restriction and co-trains the classifier as TRL does.

## 3. The arms

### 3.1 TRL's shipped optimiser [`grpo_trl/mc.py`]

`BudgetedTrainer` subclasses TRL's `_Trainer`. Its `_episode`, `_explore`, `_update`, `_train_policy`, and the
`targets` and `rollouts` they call, are the library's code. Only the outer loop is restated, so it can stop on a
compute budget and report after each update. What that code does: for every state an episode visits, reveal each
slot in turn and roll the current policy out `n_rollouts` times; fit the policy by cross-entropy to
`softmax(Q / tau_info)` over slots and to `sigmoid(Q_stop / tau_stop)` for STOP, where `Q_stop` is the reward of
stopping now.

- **TRL shipped (defaults):** every setting at the library default.
- **TRL shipped (tuned):** the same code with settings from the search in section 6.

### 3.2 Policy-gradient arms [`grpo_trl/pg.py`]

One trainer; arms differ only in how episodes are collected and how the advantage is computed.

Per update: sample training samples in passes (every sample once per pass); play episodes with the current
stochastic policy; compute advantages; take `epochs` passes of minibatch Adam steps on the clipped surrogate

    L = - mean_w[ min(rho * A, clip(rho, 1 - eps, 1 + eps) * A) ] - c_ent * H[pi] + beta * KL(pi || pi_old)

with `rho = pi(a|s) / pi_old(a|s)`. Gradient norm clipped at 0.5; Adam epsilon 1e-5; constant learning rate.

| Arm | Episodes per update | Advantage of step t of episode g of sample i | Step weight |
|---|---|---|---|
| **PPO** | `B` samples, one episode each | GAE(lambda), gamma = 1, from a learned critic `V(s)`; normalised per minibatch | 1 |
| **GRPO (canonical)** | `B/G` samples, `G` episodes each | `(R_ig - mean_g R_i) / (std_g R_i + 1e-6)` | `1 / length` |
| **GRPO (mean-centred)** | same | `(R_ig - mean_g R_i) / (batch std of the centred returns + 1e-6)` | `1 / length` |

- PPO's critic is TRL's `Classifier` architecture (hidden 256) on the same state, with its own Adam optimiser,
  trained by squared error to the GAE value targets. PPO receives rewards step by step (the step penalty and the
  slot's cost at each reveal, the terminal reward at the end).
- GRPO (canonical) is equation 3 of Shao et al. (2024) with outcome supervision: one scalar advantage per
  episode, applied to all its steps, `1/|o|` length normalisation, no critic. The KL term's reference is the
  policy that collected the batch, since there is no pretrained reference policy; it is computed exactly over the
  (at most eight) actions.
- GRPO (mean-centred) changes one thing: the group's standard deviation is not divided out. A single batch-wide
  scale keeps the step size independent of the reward's units while leaving the relative weight of samples equal
  to the true gradient's. Removing the std term is one of the two changes of "Dr. GRPO" (Liu et al., 2025,
  arXiv:2503.20783); the other (removing length normalisation) is an extra arm in the ablation.

### 3.3 Reference policies (no learning)

- **Random:** uniform over allowed actions (20 episodes per sample).
- **Static oracle:** the fixed set of slots with the best mean return on train, by exhaustive search over every
  set the regime allows. It is the best policy that does not adapt to what it sees.
- **Reveal-all**, and a **clairvoyant bound** (every sample gets its own best set, chosen knowing its label;
  unreachable, shown only for scale).

### 3.4 The baseline ladder (E2)

GRPO's tuned settings and data collection, with only the advantage changed: no baseline; batch mean and std (the
grouping of de Oliveira et al., 2025, where groups mix unrelated episodes); per-sample mean; per-sample mean and
std (GRPO); leave-one-out mean; Dr. GRPO (per-sample mean, no length normalisation); a learned critic on the same
data (PPO-G); a privileged critic that also sees every feature and the label (PPO-priv).

## 4. Budget and compute accounting [`grpo_trl/meter.py`]

Every arm gets the same compute budget and stops before the update that would exceed it.

- **Unit: multiply-accumulates (MACs).** A forward row through a network costs the sum of
  `in_features * out_features` over its linear layers; a row that is back-propagated costs three times that.
  Forward hooks on the policy, the classifier and PPO's critic count rows whichever code calls them. Evaluation is
  not charged.
- **Budget `C1`:** what TRL's default trainer spends on one pass over the training set (`n_iterations=1`, the
  default), measured per (dataset, regime) on two dev seeds (`experiments/budgets.json`).
- Simulated decisions ("transitions"), episodes and wall-clock seconds are logged at every checkpoint.
- Ten checkpoints per run, at equal shares of the budget. **The endpoint is the last one**: the policy `fit`
  would return. No checkpoint is selected.

The proxy is conservative for the new arms: at equal MACs they finish in about half the wall-clock time of TRL's
trainer on the same machine, because whole episodes batch better than one state's rollouts.

## 5. Data [`grpo_trl/data.py`]

| Dataset | Rows in pool | Positive | Slots (columns) |
|---|---|---|---|
| `synth-route` | 12,000 | 29% | router (8), four candidates (8 each), weak (8) |
| `founders` | 3,000 | 20% | company, background, education, jobs, prior startups, prior exits (384 each) |
| `kickstarter` | 12,000 | 32% | name, description, keywords (384 each), goal (3), geography (11), timing (5) |
| `fakejobs` | 12,000 | 4.3% | title, description (384 each), salary (4), experience (9), education (10) |
| `credit` | 12,000 | 22% | profile (13), six monthly slots (4 each) |
| `adult` | 12,000 | 24% | demographics (9), education (2), work (26), family (14), capital (3) |

- Text slots are embedded with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, unit norm, 128 tokens).
- Tabular slots: numeric columns standardised on the training split; categorical columns one-hot (most frequent
  values plus "other"); a constant 1 appended so a revealed slot is never all zeros.
- `synth-route`: the label is Bernoulli(0.3). The router names one of four candidate slots; only that candidate
  carries the label (a shift of 1.5 standard deviations along a fixed direction). The weak slot always carries a
  shift of 0.5. The best strategy reads the router, then the named candidate, then stops.
- `founders`: the 3,000 generated profiles of `Sidi3355/trl-for-angels`, cut into the six parts of its template.
  Generated data: nothing here is a claim about real founders.
- Sources larger than 12,000 rows are reduced once to a stratified pool (seed 20261122).
- **Per seed:** a fresh stratified split of the pool into train 2,000, validation 500, test 2,000 (`founders`:
  1,800 / 400 / 800).

## 6. Tuning [`grpo_trl/configs.py`, `experiments/tune_select.py`]

Equal effort for every learner, on dev seeds only (1000, 1001, 1002), judged on validation return at the end of
the budget.

1. Random search: 24 configurations per method on four dev conditions (`synth-route` R2, `founders` R1,
   `founders` R2, `credit` R2), dev seed 1000. GRPO and its mean-centred variant get the same 24 draws.
2. The three best of each method are re-run on dev seeds 1001 and 1002; the best mean over all three seeds is
   frozen.
3. One configuration per method serves every dataset and regime.

Search spaces are in `PLAN.md` section 6.3 (plus `min_queries_frac` for the incumbent, Amendment A1.2).

## 7. Seeds, pairing, endpoints

A seed fixes the split, the frozen classifier, the initial policy weights and all sampling. Every arm of a
condition (dataset x regime) runs on the same seeds, so all comparisons are paired. Dev seeds start at 1000;
confirmatory seeds at 0; the two never mix.

- **Primary endpoint:** mean return per test sample of the greedy policy at the end of the budget.
- **Normalised return:** `(R - R_random) / (R_static - R_random)`, with both anchors averaged over the
  condition's seeds, so 0 is the random policy, 1 is the best fixed slot set, and more than 1 means the policy
  gains by adapting. A condition whose anchor gap is under 2% of its reward range is reported raw and left out of
  pooled statistics.

## 8. Statistics [`grpo_trl/stats.py`, `grpo_trl/analysis.py`]

- **Confirmatory tests:** the mean paired difference in normalised test return, averaged over conditions with
  equal weight. 95% CI: percentile bootstrap, resampling seeds within each condition (10,000 resamples). p-value:
  sign-flip permutation test, flipping within conditions (10,000 permutations), two-sided. Non-inferiority tests
  shift the differences by the margin (0.10) and are one-sided.
- **Multiplicity:** Holm over the eight confirmatory tests, family-wise alpha 0.05.
- **Robustness beside each test:** probability of improvement (share of paired seeds won, averaged over
  conditions, bootstrap CI); wins, ties and losses by condition (a win is a condition whose own 95% CI excludes
  zero); interquartile mean of normalised return per arm.
- **Minimum effect of interest:** 0.05 normalised units.
- The permutation test's false-positive rate, the Holm adjustment, the bootstrap and the normalisation are
  unit-tested on numbers with known answers (`tests/test_stats.py`).
