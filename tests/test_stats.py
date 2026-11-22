"""The analysis does what the plan says, checked on numbers with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grpo_trl import stats
from grpo_trl.analysis import MARGIN, add_normalised, compare, paired, verdict


def test_pooled_mean_weights_conditions_equally() -> None:
    differences = {"a": np.array([1.0, 1.0]), "b": np.array([0.0] * 50)}
    assert stats.pooled_mean(differences) == pytest.approx(0.5)


def test_bootstrap_ci_covers_a_clear_effect_and_excludes_zero() -> None:
    rng = np.random.default_rng(0)
    differences = {c: rng.normal(0.3, 0.1, size=12) for c in "abc"}
    low, high = stats.stratified_bootstrap(differences)
    assert 0 < low < 0.3 < high < 0.6


def test_sign_flip_p_is_small_for_an_effect_and_uniform_without_one() -> None:
    rng = np.random.default_rng(1)
    effect = {c: rng.normal(0.3, 0.1, size=12) for c in "abc"}
    assert stats.sign_flip_p(effect) < 0.001
    nulls = [
        stats.sign_flip_p({c: rng.normal(0, 1, size=10) for c in "ab"}, n=500, seed=i)
        for i in range(200)
    ]
    assert 0.02 < np.mean(np.array(nulls) < 0.05) < 0.10  # about 5% false positives


def test_noninferiority_uses_the_margin() -> None:
    slightly_worse = {c: np.full(10, -0.03) + np.linspace(-0.01, 0.01, 10) for c in "ab"}
    clearly_worse = {c: np.full(10, -0.30) + np.linspace(-0.01, 0.01, 10) for c in "ab"}
    assert stats.sign_flip_p(slightly_worse, alternative="greater", shift=MARGIN) < 0.01
    assert stats.sign_flip_p(clearly_worse, alternative="greater", shift=MARGIN) > 0.9


def test_probability_of_improvement_counts_ties_as_half() -> None:
    point, _, _ = stats.probability_of_improvement({"a": np.array([1.0, -1.0, 0.0, 2.0])})
    assert point == pytest.approx((2 + 0.5) / 4)


def test_iqm_ignores_the_tails() -> None:
    scores = {"a": np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])}
    assert stats.iqm(scores, n=200)[0] == pytest.approx(1.0)


def test_holm_matches_a_worked_example() -> None:
    # sorted p: 0.01, 0.02, 0.03, 0.04 -> x4, x3, x2, x1 with running max
    assert stats.holm([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_wilcoxon_handles_all_zero_differences() -> None:
    assert stats.wilcoxon_p(np.zeros(8)) == 1.0
    assert stats.wilcoxon_p(np.arange(1.0, 11.0)) < 0.01


def _frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(2)
    for condition, (random, static) in {"x/R1": (-2.0, 0.0), "y/R2": (0.2, 0.6)}.items():
        dataset, regime = condition.split("/")
        for seed in range(10):
            for arm, level in (("good", 1.2), ("bad", 0.4)):
                noise = rng.normal(0, 0.02)
                rows.append(
                    {
                        "condition": condition,
                        "dataset": dataset,
                        "regime": regime,
                        "arm": arm,
                        "seed": seed,
                        "prevalence": 0.25,
                        "ba_all_val": 0.8,
                        "test_ret": random + (level + noise) * (static - random),
                        "test_ret_valbest": random + level * (static - random),
                        "anchor_random_test": random,
                        "anchor_static_test": static,
                        "anchor_reveal_all_test": random,
                        "anchor_clairvoyant_test": static + 1,
                    }
                )
    return pd.DataFrame(rows)


def test_normalisation_puts_random_at_zero_and_static_at_one() -> None:
    frame = add_normalised(_frame())
    means = frame.groupby("arm")["norm"].mean()
    assert means["good"] == pytest.approx(1.2, abs=0.02)
    assert means["bad"] == pytest.approx(0.4, abs=0.02)
    assert not frame["degenerate"].any()


def test_compare_and_verdicts() -> None:
    frame = add_normalised(_frame())
    result = compare(frame, "good", "bad")
    assert result["n_conditions"] == 2 and result["n_pairs"] == 20
    assert result["diff"] == pytest.approx(0.8, abs=0.03) and result["ci_low"] > 0.7
    assert result["poi"] == 1.0 and result["wins"] == 2
    assert verdict(result, result["p"]) == "supported"
    reverse = compare(frame, "bad", "good")
    assert verdict(reverse, reverse["p"]).startswith("reversed")
    noninferior = compare(frame, "bad", "good", kind="noninferiority")
    assert verdict(noninferior, noninferior["p"]).startswith("rejected")
    assert set(paired(frame, "good", "bad")) == {"x/R1", "y/R2"}


def test_a_condition_with_no_gap_is_left_out() -> None:
    frame = _frame()
    frame.loc[frame["condition"] == "x/R1", "anchor_static_test"] = -1.99  # gap 0.01, headroom 3
    normalised = add_normalised(frame)
    assert normalised.loc[normalised["condition"] == "x/R1", "degenerate"].all()
    assert compare(normalised, "good", "bad")["n_conditions"] == 1
