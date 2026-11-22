"""The pre-registered analysis (PLAN.md section 8), on plain arrays.

Runs are paired by seed within a condition (dataset x regime). Pooled
statistics give every condition the same weight and resample or permute
within conditions only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats

Floats = npt.NDArray[np.float64]
Paired = Mapping[str, Floats]  # condition -> paired differences (one per seed)

N_RESAMPLES = 10_000


def pooled_mean(differences: Paired) -> float:
    """Mean over conditions of the mean paired difference."""
    return float(np.mean([np.mean(d) for d in differences.values()]))


def stratified_bootstrap(
    differences: Paired, n: int = N_RESAMPLES, seed: int = 0
) -> tuple[float, float]:
    """95% percentile CI of ``pooled_mean``, resampling seeds within each condition."""
    rng = np.random.default_rng(seed)
    means = np.zeros(n)
    for d in differences.values():
        d = np.asarray(d, dtype=float)
        means += d[rng.integers(len(d), size=(n, len(d)))].mean(axis=1)
    means /= len(differences)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_p(
    differences: Paired,
    *,
    alternative: str = "two-sided",
    shift: float = 0.0,
    n: int = N_RESAMPLES,
    seed: int = 0,
) -> float:
    """Sign-flip permutation p-value of ``pooled_mean``, flipping within conditions.

    Under the null the paired differences (plus ``shift``) are symmetric about
    zero. ``alternative="greater"`` with ``shift=margin`` is the non-inferiority
    test "the difference exceeds ``-margin``".
    """
    rng = np.random.default_rng(seed)
    shifted = {c: np.asarray(d, dtype=float) + shift for c, d in differences.items()}
    observed = pooled_mean(shifted)
    null = np.zeros(n)
    for d in shifted.values():
        signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
        null += (signs * d).mean(axis=1)
    null /= len(shifted)
    if alternative == "two-sided":
        extreme = np.abs(null) >= abs(observed) - 1e-12
    elif alternative == "greater":
        extreme = null >= observed - 1e-12
    else:
        raise ValueError(alternative)
    return float((extreme.sum() + 1) / (n + 1))


def probability_of_improvement(
    differences: Paired, n: int = N_RESAMPLES, seed: int = 0
) -> tuple[float, float, float]:
    """Share of paired seeds won (ties count half), averaged over conditions, with a 95% CI."""

    def share(d: Floats) -> Floats:
        return (d > 0).mean(axis=-1) + 0.5 * (d == 0).mean(axis=-1)

    rng = np.random.default_rng(seed)
    point = float(np.mean([share(np.asarray(d, dtype=float)) for d in differences.values()]))
    boot = np.zeros(n)
    for d in differences.values():
        d = np.asarray(d, dtype=float)
        boot += share(d[rng.integers(len(d), size=(n, len(d)))])
    boot /= len(differences)
    low, high = np.percentile(boot, [2.5, 97.5])
    return point, float(low), float(high)


def iqm(scores: Mapping[str, Floats], n: int = N_RESAMPLES, seed: int = 0) -> tuple[float, float, float]:
    """Interquartile mean over all runs (Agarwal et al., 2021), stratified bootstrap 95% CI."""

    def interquartile_mean(values: Floats) -> Floats:
        return scipy_stats.trim_mean(values, 0.25, axis=-1)

    rng = np.random.default_rng(seed)
    point = float(interquartile_mean(np.concatenate([np.asarray(s) for s in scores.values()])))
    resampled = [
        np.asarray(s)[rng.integers(len(s), size=(n, len(s)))] for s in scores.values()
    ]
    boot = interquartile_mean(np.concatenate(resampled, axis=1))
    low, high = np.percentile(boot, [2.5, 97.5])
    return point, float(low), float(high)


def mean_ci(values: Floats, n: int = N_RESAMPLES, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a 95% percentile bootstrap CI."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(len(values), size=(n, len(values)))].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def wilcoxon_p(differences: Floats) -> float:
    """Two-sided Wilcoxon signed-rank p-value of paired differences (1 if all are zero)."""
    differences = np.asarray(differences, dtype=float)
    if np.allclose(differences, 0):
        return 1.0
    return float(scipy_stats.wilcoxon(differences, zero_method="wilcox").pvalue)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the order given."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.zeros(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return [float(p) for p in adjusted]


def wins_ties_losses(differences: Paired) -> tuple[int, int, int]:
    """Conditions whose 95% CI of the mean difference lies above, across, below zero."""
    wins = ties = losses = 0
    for d in differences.values():
        _, low, high = mean_ci(np.asarray(d, dtype=float))
        if low > 0:
            wins += 1
        elif high < 0:
            losses += 1
        else:
            ties += 1
    return wins, ties, losses
