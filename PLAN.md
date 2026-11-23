# Pre-registered research plan: is GRPO a better policy optimiser than PPO for Think-Reason-Learn?

**Status:** version 1, committed before any experiment was run. The git history of this file is the
pre-registration record. Later changes are appended as dated amendments in section 13; nothing above that
section is rewritten after the confirmatory runs start.

**Reading note (added 2026-11-23, Amendment A5).** This plan was registered around the question in its title.
The finished study is presented around a wider one, which the audit in section 1 already pointed to: do PPO and
GRPO train `VerifiableRL`'s policy better than the optimiser TRL ships? The text below is left as it was
registered. Hypotheses, tests and the decision rule are unchanged, and are reported as written here.

**Naming.** "TRL" here always means Vela Research's *Think-Reason-Learn* library
(`Vela-Research/think-reason-learn`), not Hugging Face's *Transformer Reinforcement Learning* library
(which also ships a `GRPOTrainer`). The two are unrelated.

---

## 1. The question, and a correction to its premise

The question as asked: *TRL uses Proximal Policy Optimization (PPO) in its reinforcement-learning model; would
Group Relative Policy Optimization (GRPO) be a better policy optimiser there?*

Before designing anything we audited the code (section 2). **TRL does not use PPO.** The string "PPO" appears
nowhere in the library, its tests, its docs or its git history (upstream `a712645`, v0.4.4, and the fork at
`3437d64`, v0.3.0; the RL module is byte-identical in both). TRL's one reinforcement-learning model,
`VerifiableRL`, trains its policy by **Monte Carlo rollout distillation**: at every visited state each action is
scored by rollouts, and the policy network is fitted by cross-entropy to a softmax of those scores. No policy
gradient, no critic, no clipped surrogate.

So the study answers three questions instead of one:

1. **GRPO vs PPO** on TRL's own decision problem (the question as asked).
2. **GRPO vs what TRL actually ships** (the question that decides whether TRL should change).
3. **Why**: which ingredient of GRPO is responsible for any difference.

## 2. Background

### 2.1 What `VerifiableRL` does (code audit, upstream v0.4.4)

`VerifiableRL` implements Ye, Alican, Griffin, Yin and Ihlamur, "Learning What to Ask: Cost-Aware Sequential
Founder Evaluation" (MIDAS @ ECML-PKDD 2026). Each sample's features come in named *slots* (education, jobs, ...).
A policy network chooses which slot to reveal next, or to STOP; a classifier network predicts the label from what
has been revealed.

- **State:** the feature vector with unrevealed slots zeroed.
- **Actions:** reveal slot *j*, or STOP (allowed once `min_queries` slots are revealed).
- **Return:** the classifier's call scored against the label (`reward_tp=4`, `reward_fp=-16`, `reward_tn=0`,
  `reward_fn=-0.25`, the call being `sigmoid(classifier) >= reward_threshold=0.3`), minus slot costs, plus
  `step_penalty=-0.1` per reveal.
- **Policy head:** softmax over slots, separate sigmoid for STOP (`_networks.action_probs`).
- **Policy training (`_episodes.targets`, `_Trainer._train_policy`):** for each visited state, each slot is
  revealed and then rolled out `n_rollouts=10` times with the current policy; the info target is
  `softmax(Q_info / tau_info)`; the STOP target is `sigmoid(Q_stop / tau_stop)`, where `Q_stop` is the reward of
  stopping *now*. The policy is fitted to these targets from a replay buffer. The docstring notes the consequence:
  "slot costs and the step penalty shape which slot is revealed, not when to stop".
- **Classifier training:** pretrained on random reveals, then co-trained on the rollouts' final states.

We could not read the paper itself (its host is unreachable from the sandbox and it is not on arXiv); the audit
is of the code, which the changelog says was "checked against its paper".

### 2.2 PPO and GRPO

- **PPO** (Schulman et al., 2017, arXiv:1707.06347): on-policy actor-critic. A learned value function gives
  per-step advantages (GAE); the policy maximises a clipped importance-weighted surrogate over several epochs.
- **GRPO** (Shao et al., 2024, DeepSeekMath, arXiv:2402.03300, section 4.1): drops the critic. For each prompt it
  samples a *group* of G outputs from the old policy and uses the group's own rewards as the baseline:
  `A_i = (r_i - mean(r)) / std(r)`, applied to every step of output *i*, inside PPO's clipped surrogate, with a KL
  penalty to a reference policy added to the loss.
- **Prior evidence outside language models.** de Oliveira et al., "Learning Without Critics? Revisiting GRPO in
  Classical Reinforcement Learning Environments" (arXiv:2511.03527; AKCIT and OATML, Oxford) found GRPO
  *underperforms* PPO on Acrobot, MountainCar, HalfCheetah and Humanoid and matches or beats it only on
  short-horizon CartPole. They name two causes: no state-dependent credit assignment over long horizons, and a
  grouping strategy that "mixes potentially unrelated episodes" because classical RL has no prompt. They list
  better grouping as an open problem.

### 2.3 Why `VerifiableRL` is a principled place to expect GRPO to work

The `VerifiableRL` decision problem has exactly the three properties the prior work says GRPO needs:

1. **A natural prompt.** Each training sample is a fixed initial context from which any number of episodes can be
   sampled. Groups are *per sample*, as in the original GRPO, not per batch.
2. **A short horizon.** At most `max_steps=5` actions.
3. **An outcome reward.** The dominant reward arrives at the end.

It also has one property that should make a per-sample group baseline *stronger* than any learned critic. The
return depends on the label, which the policy never observes. A critic `V(s)` sees only the revealed features, so
it cannot remove the return variance due to the hidden label and the hidden slots. A group of episodes on the
same sample shares the label and the hidden slots, so its mean removes that variance exactly. The group baseline
is, in effect, a privileged-information baseline that costs no parameters.

## 3. Hypotheses

Primary metric: mean test return of the greedy deployment policy (section 7). Directions are stated in advance.

| ID | Hypothesis | Prediction | Test |
|---|---|---|---|
| **H1** | GRPO vs PPO at equal compute | GRPO attains higher test return than PPO | superiority, pooled over all conditions |
| **H2a** | GRPO vs TRL's shipped optimiser (tuned), cost-aware regime R2 | GRPO attains higher test return | superiority, pooled over R2 conditions |
| **H2b** | GRPO vs TRL's shipped optimiser (tuned), TRL-default regime R1 | GRPO is not worse by more than the margin | non-inferiority, margin 0.10 normalised units, pooled over R1 conditions |
| **H3** | Per-sample grouping is the active ingredient | GRPO with per-sample groups beats GRPO with batch groups (the grouping of arXiv:2511.03527) | superiority, pooled over ablation conditions |
| **H4** | Std-normalisation under asymmetric rewards | Two-sided. Directional guess: dividing by the group std reweights samples away from the true return gradient, so mean-only centering ("GRPO-nostd") attains *higher* return than z-scored GRPO | two-sided, pooled over ablation conditions |

Secondary, descriptive (no confirmatory test): compute- and transition-efficiency curves; seed-to-seed stability;
effect on prediction quality (F0.5, precision, recall) and on acquisition cost; adaptivity of the learned paths;
the baseline ladder of section 5.3; sensitivity to group size.

**Sensitivity analysis for H1 (pre-specified):** repeat the H1 comparison at *matched environment transitions*
instead of matched compute, i.e. with PPO's critic free of charge (GRPO read at its last checkpoint at or below
PPO's final transition count).

## 4. The decision problem (identical for every arm)

All arms optimise the same Markov decision process with the same networks; only the policy optimiser differs.

- **Networks:** TRL's own `PolicyNet` (hidden 512) and `Classifier` (hidden 256), imported from the pinned
  library, initialised by TRL's `seeded`.
- **Policy distribution:** `pi(a|s)` proportional to `action_probs(logits)[a] * allowed[a]`, where `allowed` is
  TRL's `allowed_actions` (unseen slots; STOP once `min_queries` are revealed). Its argmax is exactly TRL's
  deployment rule `greedy(probs, allowed)`, so every arm is evaluated by TRL's own inference and a policy trained by
  any arm is a drop-in `policy_` for `VerifiableRL.predict`.
- **Episode:** start with nothing revealed; act until STOP, all slots revealed, or `max_steps` reveals.
- **Return:** `R = r_term(classifier(final state), label) - sum(costs of revealed slots) + step_penalty * (#reveals)`.
- **Frozen classifier (protocol A, primary).** The reward depends on the classifier. To make the comparison one of
  optimisers and not of moving reward functions, the classifier is trained once per (dataset, seed) by TRL's
  random-reveal curriculum (`VerifiableRL._pretrain`) and then frozen for every arm. For the incumbent this is
  TRL's own trainer with classifier updates switched off.
- **Co-trained classifier (protocol B, secondary).** An end-to-end check through the real `VerifiableRL.fit`
  path, classifier co-training included, on a subset (experiment E5).
- `predict_min_queries` is set equal to `min_queries`, so training and deployment allow STOP at the same states.

### Reward regimes

| Regime | Reward | Call threshold | Slot costs | `min_queries` | `max_steps` | Purpose |
|---|---|---|---|---|---|---|
| **R1 TRL-default** | TP 4, FP -16, TN 0, FN -0.25, step -0.1 | 0.3 | 0 | 3 | 5 | Fidelity: the library exactly as shipped |
| **R2 cost-aware** | TP `1/(2*pi1)`, TN `1/(2*pi0)`, FP 0, FN 0, step 0 (return = balanced accuracy - cost) | `pi1` (the Bayes threshold for this reward) | heterogeneous, sized to the dataset (below) | 1 | number of slots | A well-posed costly-information problem in which *when to stop* matters |

`pi1` is the training prevalence. R2 slot costs: `c_j = 0.5 * (BA_all - 0.5) * w_j`, with `BA_all` the frozen
classifier's validation balanced accuracy with every slot revealed and `w = linspace(0.5, 1.5, S) / S` in slot
order. Revealing everything therefore costs half of what full information is worth, on every dataset, and costs
are assigned without reference to which slots are informative. Both regimes use only existing `VerifiableRL`
constructor arguments.

## 5. Arms

### 5.1 Main arms (experiment E1)

| Arm | Policy optimiser | Hyperparameters |
|---|---|---|
| **MC-default** | TRL's rollout distillation, TRL's own code | TRL defaults |
| **MC-tuned** | the same code | tuned (section 6.3) |
| **PPO** | clipped surrogate, learned critic, GAE | tuned |
| **GRPO** | clipped surrogate, per-sample group-relative advantage, no critic | tuned |

- **MC arms** call TRL's `_Trainer` (its `_episode`, `targets`, `rollouts`, `_train_policy`) unmodified, through a
  thin subclass that only adds compute accounting and checkpoints.
- **PPO:** per update, `B` episodes on `B` training samples drawn at random; critic = TRL `Classifier`-shaped MLP
  (hidden 256, the cheaper of TRL's two architectures) on the same state; rewards given per step (step penalty and
  slot cost at each reveal, terminal reward at the end); GAE with gamma = 1; per-minibatch advantage
  normalisation; separate Adam optimisers; max grad norm 0.5; optional entropy bonus.
- **GRPO:** per update, `M = B / G` training samples, `G` episodes each from the old policy;
  `A_ig = (R_ig - mean_g R_i) / (std_g R_i + 1e-6)` applied to every step of episode *(i, g)*; clipped surrogate;
  KL penalty `beta * KL(pi_theta || pi_ref)` with `pi_ref` the policy that collected the batch (there is no
  pretrained reference policy here; `beta = 0` is in the search space); max grad norm 0.5; optional entropy bonus.
  No critic, no value loss.

### 5.2 Reference policies (no learning; anchors for normalisation)

- **Random:** uniform over allowed actions.
- **Static oracle:** the best *non-adaptive* policy, i.e. the fixed set of slots (revealed in a fixed order, then
  STOP) with the highest mean return on the training set, by exhaustive search over all subsets obeying
  `min_queries` and `max_steps`.
- **Reveal-all:** reveal up to `max_steps` slots in index order.

### 5.3 Ablation arms (experiment E2): the baseline ladder

Same clipped surrogate, same data collection as GRPO (`M` samples x `G` episodes), same tuned hyperparameters as
GRPO; only the advantage changes.

| Arm | Advantage |
|---|---|
| PG-nobaseline | return, batch-normalised |
| GRPO-batch | batch mean and std (the grouping of arXiv:2511.03527: groups of unrelated episodes) |
| GRPO-nostd | per-sample group mean only |
| GRPO | per-sample group mean and std |
| GRPO-loo | per-sample leave-one-out mean, group std |
| PPO-G | learned critic + GAE on GRPO's data collection |
| PPO-priv | learned critic that also sees the full features and the label (asymmetric critic) |

Experiment **E3** sweeps the group size `G` in {2, 4, 8, 16, 32} at a fixed episode batch.

## 6. Protocol

### 6.1 Datasets

Six datasets, chosen before any result: one synthetic with a known optimal strategy, one from TRL's own domain,
four public real datasets (two with text slots, two tabular). Slot definitions are fixed here.

| Name | Source | Slots | Label |
|---|---|---|---|
| `synth-route` | generated in this repo | 6 x 8 dims: a *router* slot naming which of four candidate slots carries the label signal, four candidates, one weak always-on slot | Bernoulli(0.3) |
| `founders` | `Sidi3355/trl-for-angels`, `data/raw/full.jsonl` (3,000 generated founder profiles; MIT) | 6 text slots from the profile template: company, background, education, jobs, prior startups, prior exits; each embedded by `sentence-transformers/all-MiniLM-L6-v2` (384 dims) | outcome of $500M or more |
| `kickstarter` | HF `james-burton/kick_starter_funding` | name, description, keywords (MiniLM, 384 each), goal, geography, timing | funded |
| `fakejobs` | HF `james-burton/fake_job_postings2` | title, description (MiniLM), salary, required experience, required education | fraudulent |
| `credit` | HF `Lancer73/uci-credit-card-default` (UCI Taiwan credit) | profile, then one slot per month of history (6 months: status, bill, payment) | default next month |
| `adult` | HF `scikit-learn/adult-census-income` | demographics, education, work, family, capital | income over $50K |

VCBench, the benchmark the TRL papers use, is distributed on request only and was not available; `founders` is
generated data, so nothing here is a claim about real founders.

**Splits.** For each seed, a fresh stratified random split of the dataset's pool: train 2,000 / validation 500 /
test 2,000 (`founders`: 1,800 / 400 / 800). Large sources are first reduced to a fixed stratified pool of 12,000
rows. Numeric features are standardised on train; embeddings are used as they are.

### 6.2 Budget and compute accounting

Every arm gets the same **compute budget** per run and is evaluated at ten equally spaced checkpoints.

- **Unit.** Multiply-accumulates (MACs) of the networks: a forward row costs the network's MACs, a training row
  (forward and backward) costs three times that. Counted by hooks on every network of every arm (policy,
  classifier, critic), during training only. This is deterministic and hardware-independent; wall-clock time is
  logged beside it to check that the proxy tracks real time.
- **Budget `C1`.** For each (dataset, regime): the compute TRL's default trainer uses for one pass over the
  training set (`n_iterations=1`, the library default), measured on the dev seeds and frozen in Amendment A1.
- Environment transitions (simulated actions) and episodes are logged as secondary axes.
- **Endpoint.** The policy at the end of the budget: what `fit` would return. No checkpoint selection. The
  validation-selected checkpoint is reported as a secondary endpoint.

### 6.3 Hyperparameter tuning (equal effort per method)

Tuning uses **dev seeds 1000 and above only**, never a confirmatory seed, and validation return only.

- Random search, the same number of configurations for MC, PPO and GRPO, on four dev conditions
  (`synth-route` R2, `founders` R1, `founders` R2, `credit` R2), one dev seed each. Objective: mean normalised
  validation return at the end of the budget.
- The top three configurations per method are re-run on two further dev seeds; the best mean is frozen.
- One configuration per method is used for every dataset and regime.
- Search spaces:
  - **MC:** `policy_lr` log-uniform [3e-5, 3e-3]; `n_rollouts` {2, 5, 10}; `max_depth` {2, 4, 8};
    `tau_info` {0.3, 1, 3}; `tau_stop` {1, 4}; `update_every` {10, 25, 50}; `train_epochs` {1, 2, 4};
    `policy_batch` {64, 128}; `eps` {0.05, 0.1, 0.3}.
  - **PPO:** lr log-uniform [3e-5, 3e-3]; critic-lr multiplier {1, 3}; clip {0.1, 0.2, 0.3}; epochs {1, 2, 4};
    minibatch {256, 1024}; GAE lambda {0.8, 0.95, 1}; entropy coefficient {0, 0.003, 0.01, 0.03};
    episodes per update {256, 512, 1024, 2048}.
  - **GRPO:** lr log-uniform [3e-5, 3e-3]; clip {0.1, 0.2, 0.3}; epochs {1, 2, 4}; minibatch {256, 1024};
    group size {4, 8, 16, 32}; episodes per update {256, 512, 1024, 2048}; KL coefficient {0, 0.01, 0.04};
    entropy coefficient {0, 0.003, 0.01, 0.03}.

### 6.4 Seeds and pairing

- A seed fixes the data split, the frozen classifier, the initial policy weights and all sampling. Every arm of a
  condition runs on the same seeds, so comparisons are **paired**.
- Confirmatory seeds are 0, 1, 2, ...; their number `n` is fixed in Amendment A1 from the dev-seed variance, to
  give at least 90% power for a pooled difference of 0.05 normalised units, with a floor of 10 and a cap of 30.

## 7. Metrics

- **Primary:** mean return per test sample of the greedy policy, under TRL's deployment rule.
- **Normalised return:** `(R - R_random) / (R_static - R_random)`, anchors computed per (dataset, regime, seed):
  0 is the random policy, 1 is the best non-adaptive policy, above 1 means the policy gains from adapting to what
  it sees. A condition whose anchor gap is below 2% of its reward range is reported raw and left out of pooled
  normalised statistics.
- **Secondary:** precision, recall, F0.5, balanced accuracy of the final calls; mean slots revealed; mean
  acquisition cost; train return (generalisation gap); number of distinct decision paths; on `synth-route`, the
  share of samples routed to the correct candidate slot; compute and transitions to reach normalised return 0.9;
  wall-clock time.

## 8. Analysis

- **Per condition:** mean and 95% bootstrap CI over seeds per arm; paired difference with 95% bootstrap CI;
  Wilcoxon signed-rank p.
- **Pooled (the confirmatory tests):** the mean paired difference in normalised test return across conditions,
  with a 95% CI from a bootstrap stratified by condition (10,000 resamples), and a p-value from a sign-flip
  permutation test stratified by condition. H2b is tested against its margin.
- **Multiplicity:** Holm correction over {H1, H2a, H2b, H3, H4}, family-wise alpha 0.05.
- **Robustness, reported beside every pooled test:** probability of improvement (share of paired seeds won,
  averaged over conditions, stratified bootstrap CI); interquartile mean of normalised return with stratified
  bootstrap CI (Agarwal et al., 2021); wins, ties and losses by condition. A hypothesis is called *supported* only
  if the pooled test passes after Holm; it is called *robust* if the probability of improvement agrees.
- **Minimum effect of interest:** 0.05 normalised units. Smaller differences are reported as practically
  equivalent whatever their p-value.
- All analysis code is written and tested on dev-seed results before the confirmatory results are read.

## 9. Decision rule for the recommendation to TRL

- **Recommend offering GRPO in `VerifiableRL`** if H2a is supported and H2b's non-inferiority holds.
- **Recommend it as the cheaper alternative only** if GRPO is non-inferior in both regimes but not superior, and
  reaches the incumbent's return with less compute.
- **Recommend no change** if GRPO is worse than the tuned incumbent. This is a legitimate outcome and will be
  reported as plainly as a positive one.
- H1 (GRPO vs PPO) answers the question as asked, whatever happens to H2.

## 10. Experiments

| ID | What | Arms | Conditions | Seeds |
|---|---|---|---|---|
| E0 | Validation of the test bed: unit tests, known-optimum checks, anchors, timing | all | toy + dev | dev |
| T | Tuning (6.3) | MC, PPO, GRPO | 4 dev conditions | dev |
| **E1** | **Main comparison (H1, H2a, H2b)** | MC-default, MC-tuned, PPO, GRPO | 6 datasets x 2 regimes | `n` confirmatory |
| **E2** | **Baseline ladder (H3, H4)** | 5.3 | `synth-route`, `founders`, `credit` x R2; `founders` x R1 | 10 confirmatory |
| E3 | Group-size sweep | GRPO, G in {2, 4, 8, 16, 32} | as E2 | 10 |
| E4 | Longer budget (4 x `C1`): do the arms converge to the same place? | main arms | `synth-route`, `founders` x both regimes | 5 |
| E5 | End-to-end through `VerifiableRL.fit` with classifier co-training (protocol B) | shipped vs GRPO-patched | `founders`, `credit` x both regimes | 5 |

E3 to E5 are run if compute allows, in that order; any that is dropped is recorded in section 13.

## 11. Threats to validity, and what is done about each

| Threat | Mitigation |
|---|---|
| The incumbent is library code; PPO and GRPO are new code and could be subtly wrong or weak | Unit tests against hand-computed advantages and returns; a bandit with a known optimum; each arm must beat the random anchor on a toy problem before any tuning; equal tuning effort; standard CleanRL-style PPO choices |
| Hyperparameters tuned for the newcomer but not the incumbent | The incumbent gets the same search budget; it is also reported as shipped |
| A budget that favours one paradigm | Compute is counted for every network including PPO's critic; transitions and wall-clock are reported as well; the H1 sensitivity analysis charges PPO nothing for its critic |
| Frozen classifier differs from how TRL trains | E5 runs the real `fit` path |
| Researcher degrees of freedom | This plan; dev and confirmatory seeds are disjoint; the endpoint has no checkpoint selection; one configuration per method across all conditions |
| Generated founder data; no VCBench | Four real public datasets; no claim about real founders |
| Return can rise while prediction quality falls (the optimiser exploits the reward) | Prediction metrics are reported for every arm and discussed whichever way they move |
| Small numbers of seeds | Paired design, pooled stratified tests, `n` from a power calculation |

## 12. Deliverables

1. `REPORT.md`: the full write-up (method, results, verdicts on H1 to H4, limitations).
2. `results/`: raw per-run records, summary tables, figures; everything regenerable by one command.
3. `grpo_trl/` and `tests/`: the test bed.
4. If section 9 recommends a change: a TRL-native implementation (`VerifiableRL(optimizer="grpo")`) written to
   TRL's contribution rules, with tests, and the text of a pull request.
5. `docs/DEVIATIONS.md`: every departure from this plan, with its reason.

## 13. Amendments

### A1 (2026-11-22, after the pilot on dev seeds, before tuning and before any confirmatory run)

The test bed was built and run on dev seed 1000 only. Four things changed as a result. Each is recorded here
with the evidence; the confirmatory hypotheses H1 to H4 are unchanged.

**A1.1 R2's reward is made symmetric.** As first written, R2 paid `1/(2*pi1)` or `1/(2*pi0)` for a right call and
0 for a wrong one. TRL's STOP target is `sigmoid(Q_stop / tau_stop)` with `Q_stop` the reward of stopping now, so
with no negative rewards the target is at least 0.5 at every state and the incumbent stops at the first chance
whatever it has seen (pilot: one slot revealed on every test sample, for every learning rate tried). That would
have been a strawman. R2 now pays `+1/(2*pi_class)` for a right call and `-1/(2*pi_class)` for a wrong one. The
expected terminal reward becomes Youden's J (`TPR + TNR - 1 = 2*BA - 1`), an affine map of the original; the Bayes
call threshold is still the prevalence; total slot cost is `0.5 * (2*BA_all - 1)`, still half of what full
information is worth. For policy-gradient arms nothing changes but the scale; for the incumbent a right call now
gives a STOP target above 0.5 and a wrong one below.

**A1.2 The incumbent may choose its own `min_queries` in R2.** Even with A1.1, TRL's STOP rule rarely continues
past `min_queries` (its greedy rule compares P(STOP) with P(continue) *times* the best slot's share). A TRL user
would respond by raising `min_queries`. So the incumbent's search space gains `min_queries_frac` in {0, 1/3, 1/2}:
in R2 it may hold itself to `round(frac * S)` slots before STOP, in training and deployment. This only helps the
incumbent. In R1 `min_queries` stays the library default for every arm.

**A1.3 The mean-centred GRPO variant becomes a main arm, with its own hypotheses.** In the pilot canonical GRPO
(z-scored advantage) collapsed on `synth-route` R2 to "reveal one slot, stop" (test return below the random
policy), while the pre-registered ablation arm GRPO-nostd (group mean subtracted, no per-group std) learned the
adaptive optimum (router, then the candidate it names, then STOP; two slots on every test sample). This is the
direction H4 predicted, and larger than expected: per-group z-scoring gives a sample whose episodes differ only
by a slot cost the same gradient weight as a sample whose episodes differ by a right or wrong call. One dev seed
is not evidence, so:

- `GRPO-nostd` is added to E1 as a fifth main arm and tuned with the same effort and the *same* random draws as
  GRPO (only the advantage differs).
- New confirmatory hypotheses, declared now, before tuning and before any confirmatory seed is run:
  **H1'** GRPO-nostd beats PPO (as H1); **H2a'** GRPO-nostd beats the tuned incumbent in R2 (as H2a);
  **H2b'** GRPO-nostd is non-inferior to the tuned incumbent in R1 (as H2b, margin 0.10).
- The Holm family is now {H1, H2a, H2b, H3, H4, H1', H2a', H2b'}.
- Decision rule (section 9) extended: if canonical GRPO fails but GRPO-nostd passes H2a' and H2b', the
  recommendation is to offer the mean-centred variant, and to say plainly that canonical GRPO is not suitable.
- H3 (per-sample vs batch grouping) is tested for both variants; the confirmatory test stays the canonical pair,
  the mean-centred pair (GRPO-nostd vs a batch-mean baseline without std, i.e. `batch` vs `group_nostd` share the
  scale rule) is reported beside it.

**A1.4 Details the plan left open, fixed now.**

- *GRPO-nostd's scale.* The centred returns are divided by one batch-wide standard deviation, so the step size
  does not depend on the reward's units while the relative weight of samples is the true gradient's.
- *KL term.* Computed exactly over the action distribution (there are at most eight actions), not by the
  single-sample estimator the GRPO paper uses for a vocabulary.
- *Length normalisation.* Critic-free arms weight each step by `1/|episode|` (GRPO's objective, equation 3 of
  arXiv:2402.03300); critic arms average over transitions (PPO's usual objective).
- *Frozen classifier settings.* `VerifiableRL._pretrain`'s three settings are chosen per dataset by a mechanical
  rule on dev seeds 1000 and 1001 (lowest validation log loss under random reveals over a 3 x 3 grid; ties within
  0.005 go to the fewest gradient steps): `experiments/select_classifier.py`, results in
  `experiments/classifier_selection/`.
- *Tabular slots carry a constant 1,* so that a revealed slot is never the zero vector TRL uses for "not
  revealed".
- *Stopping on the budget.* Training stops before the update that would exceed the budget, so no arm overspends.

### A2 (2026-11-22, after tuning on dev seeds, before any confirmatory seed was run)

**A2.1 The degeneracy guard gets a sound yardstick.** Version 1 left a condition out of pooled statistics when its
anchor gap (best fixed set minus random) was under 2% of the regime's per-sample reward range. That range is the
wrong yardstick in R2, where the per-sample reward grows as `1/prevalence` (23 for `fakejobs`) while the expected
return lies in [-1, 1]: the rule would have dropped `fakejobs` R2, whose gap (0.22) is the largest of any R2
condition, on a technicality. The guard is now: **a condition is degenerate when its anchor gap, averaged over
seeds, is under 5% of its headroom** (clairvoyant bound minus random). On the dev seeds this drops one condition,
`kickstarter` R2, where the random policy and the best fixed set differ by 0.015 (2% of headroom) and any
normalised score would be noise divided by almost nothing. The anchors involve no arm, so the guard cannot see
any result. Which conditions it drops is decided on the confirmatory seeds' anchors.

**A2.2 Tuning score.** The score is as planned (mean normalised validation return at the end of the budget) with
each dev condition's two anchors averaged over the three dev seeds, as in the confirmatory analysis, instead of
taken from one seed (on seed 1000 `founders` R2's gap was 0.035 and its scores swamped the mean). Stage-2
candidates were the union of each method's three best under both versions of the score; the sets differed by one
configuration. `experiments/tune_search.csv`, `tune_finalists.csv`, `tune_chosen.json`.

**A2.3 Frozen configurations** (`experiments/frozen_configs.json`), one per method for every dataset and regime:

| Arm | Settings | Dev score (validation, 3 seeds x 4 conditions) |
|---|---|---|
| TRL shipped (tuned) | `policy_lr` 8.2e-5, `n_rollouts` 10, `max_depth` 4, `tau_info` 0.3, `tau_stop` 1, `update_every` 10, `train_epochs` 2, `policy_batch` 64, `eps` 0.3, `min_queries_frac` 1/3 | 0.96 |
| PPO | lr 1.16e-3, critic lr x1, clip 0.3, 2 epochs, minibatch 1024, GAE lambda 0.8, entropy 0.03, 2048 episodes per update | 1.24 |
| GRPO (canonical) | lr 1.13e-3, clip 0.1, 2 epochs, minibatch 256, G 8, 1024 episodes per update, KL 0.04, entropy 0.03 | 0.46 |
| GRPO (mean-centred) | lr 1.61e-3, clip 0.3, 4 epochs, minibatch 256, G 32, 2048 episodes per update, KL 0, entropy 0 | 1.32 |

Recorded so that the confirmatory result can be compared with it: on the dev seeds' *test* splits the same four
configurations score 0.93, 1.28, 0.76 and 1.26. Mean-centred GRPO and PPO are tied on dev data.

**A2.4 Number of seeds: a deviation.** The plan asked for the number of seeds that gives 90% power for a pooled
difference of 0.05, capped at 30. The dev seeds show that target cannot be met: paired differences have a pooled
standard deviation of 0.5 to 0.9 normalised units, so it would take more than 100 seeds, and at the measured cost
(about 8 minutes of the 4-core machine per seed for E1's 60 runs) even the cap of 30 is about four hours for E1
alone. **E1 runs seeds 0 to 9** (the plan's floor). With 10 seeds the pooled tests have 90% power for differences
of roughly 0.17 (all conditions) to 0.30 (one regime); smaller differences will be reported as unresolved, not as
equivalence. If compute remains after E2, E5 and E6, seeds 10 to 19 are added to E1; that decision depends on time
only, never on a result, and the confirmatory analysis is run on every seed completed.

**A2.5 Secondary experiments, sized to the compute.**

- E2 as planned (ladder built on canonical GRPO's frozen settings, 10 seeds), plus one arm: Dr. GRPO (per-sample
  mean, no std, no length normalisation; Liu et al., 2025, arXiv:2503.20783).
- E3 (group size) with 5 seeds, for both GRPO variants.
- E4 (four times the budget) with 3 seeds.
- E5 (end to end, classifier co-trained, through `VerifiableRL.fit`): the shipped optimiser gets
  `n_iterations=3`, the group optimiser `n_iterations=1` with its frozen settings; measured compute is then
  roughly equal, slightly in the shipped optimiser's favour. 5 seeds; `founders` and `credit`, both regimes.
- **E6 (new, exploratory, no training):** at the initial policy, the cosine between each advantage rule's update
  direction and the true policy gradient, and the mean advantage each rule gives episodes by length
  (`experiments/e6_gradient_bias.py`). It is there to explain H4's outcome, not to test anything.

**A2.6 One more sensitivity analysis.** Beside the pooled tests, the same comparisons over all twelve conditions
with returns scaled by headroom instead of by the anchor gap (defined everywhere, no condition dropped).

### A3 (2026-11-22, after the confirmatory runs: a closing note, nothing here changed an analysis)

- **E1 was not extended to seeds 10 to 19.** A2.4 allowed the extension if compute remained after E2, E5 and E6.
  It did not: the session's remaining compute went to E3, E5 and E4. The decision was made on time alone. The
  confirmatory analysis is on seeds 0 to 9 as run.
- **No condition was dropped by the degeneracy guard.** On the confirmatory seeds the smallest anchor gap was
  `kickstarter` R2's, at 5.7% of its headroom, just above the 5% line. The all-conditions headroom analysis
  (A2.6) shows what that one noisy scale does to the pooled numbers.
- **E5 gained a fourth arm before it was run:** the patch's own defaults (`optimizer="group"` and nothing else),
  to check what a user gets out of the box. The patch's defaults are the configuration frozen in A2.3.
- **The matched-transitions sensitivity analysis is vacuous for mean-centred GRPO:** at equal compute PPO used
  more transitions than it did, so reading GRPO "at PPO's transition count" reads its final checkpoint.
- **H3 as registered confounds two things** (per-sample grouping and the per-group standard deviation). It is
  reported as registered (reversed), with the pair that isolates grouping beside it, as A1.3 said it would be.

### A4 (2026-11-22, post hoc: written while E4 was still running, after its first seed had been read)

**What prompted it.** The first seed of E4 (four times the budget) showed mean-centred GRPO reaching its best
score and then dropping sharply late in training on three of four conditions (for example synthetic routing, R2:
2.49 at 60% of the budget, 1.35 at 80%), while PPO held steady. Nothing like it appears within the one-times
budget of E1.

**Hypothesis.** The mean-centred variant divides the centred returns by their standard deviation over the batch
(A1.4). As the policy becomes nearly deterministic, the episodes of a group nearly agree, the centred returns go
to zero, and so does that standard deviation. The normalised advantages keep unit variance all the same: the
step size never shrinks, and the last few disagreements are scaled up without limit. It is the per-group
problem of canonical GRPO again, one level up, and it bites only near convergence.

**E7 (post hoc, exploratory; no test from it enters the confirmatory family).** One more arm, `group_rawscale`:
the centred returns divided by the batch standard deviation of the *returns* themselves, which measures the
reward's scale and does not shrink as the policy converges. Everything else is mean-centred GRPO's frozen
configuration; nothing is re-tuned. Run (a) at four times the budget on E4's four conditions and three seeds, to
see whether the late drop goes away, and (b) at one times the budget on all twelve conditions for the same three
seeds, to see what the change costs or gains where E1 was run. Compared with mean-centred GRPO and PPO on the
same seeds.

**How it will be reported.** As a diagnosis and a candidate fix found after the confirmatory runs, on three
seeds. If it removes the drop without losing ground at one times the budget, the proposed patch will use it and
say so; the confirmatory results about mean-centred GRPO stay as they are.

**Outcome of A4 (added after E7).** At one budget the change made no difference (+0.02 [-0.09, +0.14]). At four
budgets it helped a little (0.95 to 1.12, +0.16 [-0.06, +0.43]) and did not cure the collapse: 6 of 12 runs still
ended more than 0.3 below their best, against 8 of 12 without it and 2 of 12 for PPO. The hypothesis explains at
most part of the instability. The proposed patch was therefore left as the variant the end-to-end experiment ran,
with the limitation written into its documentation, and the report's recommendation was changed to PPO.

### A5 (2026-11-23, after all results: a change of presentation only)

The write-up (`README.md`, `REPORT.md`, the shareable page, the package description) is reframed. Its stated aim
is now the comparison this plan's section 1 called "the question that decides whether TRL should change": PPO
and GRPO, as two challengers, against the optimiser TRL ships. The original question (GRPO against PPO) is
reported as the comparison between the two challengers and as the place the study started.

- No number, table, figure, test or verdict changed. H1 to H4 and H1' to H2b' are reported as registered.
- **PPO against TRL's optimiser was never a registered test,** and reframing does not make it one. It is the
  study's clearest result and is labelled descriptive, with unadjusted intervals, wherever it appears.
- The body of this plan above section 13 is not rewritten; a reading note at the top points here.
