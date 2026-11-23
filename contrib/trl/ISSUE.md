# VerifiableRL: the STOP target never sees slot costs, and PPO trains the policy better

*Text for an issue or discussion on `Vela-Research/think-reason-learn`. Short on purpose; the paper and the
repository carry the detail.*

---

Hi, and thanks for open-sourcing TRL. I ran a pre-registered comparison of policy optimisers inside
`VerifiableRL` (v0.4.4, `a712645`) and found two things I think are worth your attention. Everything except the
optimiser was your code: networks, action rule, rewards and `_reveal`.

**1. The policy cannot learn when to stop.** In `_episodes.targets`, the STOP target is
`sigmoid(q_stop / tau_stop)`, where `q_stop` is the reward of stopping now. Slot costs and the step penalty never
enter it (the docstring says as much). With a per-slot cost and `min_queries=1`, the trained policy revealed one
slot on all six datasets I tried (1.00 to 1.01 on average), and stayed close to `min_queries` when I raised it to
two (2.04 to 2.33). Four times the training budget did not change the default configuration's score.

**2. PPO, with everything else unchanged, does clearly better at equal compute.** Normalised test return, where
0 is a random policy and 1 is the best fixed set of slots:

| Optimiser | One default pass of compute (12 conditions, 10 seeds) | Four times that (4 conditions, 3 seeds) |
|---|---|---|
| Rollout, library defaults | 0.49 | 0.37 |
| Rollout, tuned | 0.60 | 0.87 |
| PPO | 1.10 | 1.42 |

PPO minus tuned rollout: +0.50 (95% CI 0.40 to 0.60), 9 of 12 conditions won and none lost, including with your
default rewards alone (+0.56). It also ran in about half the wall-clock time.

I also tested GRPO. As published it does not help (0.60); a variant without the per-group standard deviation is
close to PPO at the short budget but collapses when trained four times as long.

**What I am not claiming.** This is small: 2,000 training samples, ten seeds, public datasets plus generated
founder profiles. I did not have VCBench and could not get hold of the paper, so I may have missed how you
evaluate. The classifier was frozen in the main experiment. PPO against your optimiser was not one of my
pre-registered tests, so treat that comparison as descriptive.

**Links.**

- Paper (8 pages): `paper/paper.pdf` in https://github.com/Sidi3355/grpo-trl
- Full write-up, raw run records and the pre-registration: the same repository
- A TRL-native implementation of the critic-free variant, passing your `ruff`, strict `pyright` and full test
  suite, as a concrete starting point: `contrib/trl/` there. I would not merge it as is; its own README explains
  why.

Two questions: does this match what you see on VCBench, and would you be open to a PR adding PPO as an optional
`optimizer`? Happy to do the work.
