# Deviations from the pre-registered plan

Every departure from `PLAN.md` version 1 (commit `39ac496`), newest last. Amendments made before the
confirmatory runs are in `PLAN.md` section 13; this file lists them in one place with what they could affect.

| # | When | What changed | Why | Could it favour a side? |
|---|---|---|---|---|
| 1 | Before tuning (A1.1) | R2 reward made symmetric (right call `+1/(2*pi_class)`, wrong call `-1/(2*pi_class)`) | With no negative rewards TRL's STOP target is never below 0.5, so the incumbent stopped after one slot regardless of what it saw | Favours the incumbent (removes a trivial failure). Affine for the policy-gradient arms |
| 2 | Before tuning (A1.2) | Incumbent's search space gains `min_queries_frac` for R2 | TRL's STOP rule rarely continues past `min_queries`; a user would raise it | Favours the incumbent |
| 3 | Before tuning (A1.3) | GRPO-nostd promoted from ablation arm to main arm; hypotheses H1', H2a', H2b' added; Holm family enlarged | Pilot on one dev seed: canonical GRPO collapsed, the mean-centred variant learned the adaptive optimum | Adds tests, so the multiplicity correction is stricter for all |
| 4 | Before tuning (A1.4) | Details fixed: batch-wide scale for GRPO-nostd, exact KL, length normalisation, classifier settings rule, constant column in tabular slots, never overspend the budget | Left open in version 1 | No |
| 5 | Before confirmatory runs (A2.1) | Degenerate-condition guard: gap under 5% of headroom, instead of under 2% of the per-sample reward range | The old yardstick scales with 1/prevalence in R2 and would have dropped a healthy condition (`fakejobs` R2) | No: anchors involve no arm |
| 6 | Before confirmatory runs (A2.2) | Tuning score normalised by anchors averaged over the three dev seeds; candidate sets are the union under both versions | One seed's tiny gap swamped the mean | No: applied to every method; candidate sets differed by one configuration |
| 7 | Before confirmatory runs (A2.4) | 10 confirmatory seeds, not the 30 the plan's rule gives | The power target needs over 100 seeds; 30 would cost about four hours for E1 alone | Lowers power for every test equally; small differences are reported as unresolved |
| 8 | Before confirmatory runs (A2.5) | E3 with 5 seeds, E4 with 3; Dr. GRPO arm added to E2; E6 added as exploratory | Compute; explanation of H4 | No |
| 9 | After confirmatory runs (A3) | E1 not extended to 20 seeds | Compute went to E3, E5 and E4; decided on time alone, as A2.4 required | No |
| 10 | Before E5 was run (A3) | E5 gained an arm: the patch's own defaults | To check the out-of-the-box behaviour of the proposed change | No |
| 11 | During E4 (A4) | E7 added: a post hoc arm with a reward-scale normaliser | The first seed of E4 showed mean-centred GRPO collapsing late in training | Registered before the rest of E4 was read; exploratory; it did not cure the collapse |
| 12 | After E4 and E7 | The recommendation departs from the pre-registered decision rule: PPO, not mean-centred GRPO | The rule looked at one budget; at four the variant it selects is unstable | Against the study's own starting hypothesis |
| 13 | After all results (A5) | Presentation reframed: the write-up's stated aim is PPO and GRPO against TRL's shipped optimiser, not GRPO against PPO | The code audit showed TRL does not use PPO, and the clearest result is against TRL's optimiser | Registered hypotheses, tests, numbers and verdicts are unchanged. PPO against TRL's optimiser was not a registered test and is labelled descriptive |

## Things that did not go as the plan expected

These are results, not deviations, but a reader checking the plan against the report should find them here.

- **H1 was reversed.** The plan predicted GRPO would beat PPO. PPO beat canonical GRPO by 0.50.
- **H3 was reversed as registered.** The registered pair (canonical per-sample GRPO against a batch baseline)
  confounds grouping with the per-group standard deviation; the batch baseline won by 0.85. With the standard
  deviation removed from both, per-sample grouping is ahead by 0.10, which is not established.
- **The plan's argument for a per-sample baseline (section 2.3) was right about direction and wrong about size.**
  A per-sample mean did score above a critic that sees the label, which scored above an ordinary critic, but the
  gaps are 0.06 and 0.13 and neither is significant.
- **The plan's power target was unreachable** (deviation 7).
- **Tuning made TRL's optimiser worse in R1** (0.72 as shipped, 0.61 tuned), because three of the four dev
  conditions were R2. The report compares against the better TRL arm in each regime as well.
- **Mean-centred GRPO did not hold its gains over a long run** (E4), and the fix suggested by a diagnosis of
  the collapse only partly worked (E7).
