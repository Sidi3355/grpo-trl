"""Test bed: policy optimisers for Think-Reason-Learn's VerifiableRL.

Everything that defines the decision problem (networks, action rule, rewards,
deployment) is imported from the pinned ``think_reason_learn`` library; this
package adds only what the study needs around it: a batched episode sampler,
compute accounting, the PPO/GRPO optimisers, datasets and analysis.
"""

TRL_COMMIT = "a71264508d609bbbb15f40b0154bfcaead01aa81"  # Vela-Research/think-reason-learn, v0.4.4
