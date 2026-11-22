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
