import numpy as np

from app.ml.drift import histogram, kl_divergence


def test_histogram_density_sums_to_one():
    values = list(np.random.default_rng(1).uniform(0, 1, 200))
    h = histogram(values, bins=10)
    assert len(h["bin_edges"]) == 11
    assert abs(sum(h["density"]) - 1.0) < 1e-9


def test_kl_divergence_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(2)
    baseline_values = list(rng.uniform(0, 1, 2000))
    baseline = histogram(baseline_values, bins=10)

    same_distribution_values = list(rng.uniform(0, 1, 2000))
    score = kl_divergence(baseline, same_distribution_values)

    assert score < 0.05


def test_kl_divergence_is_large_for_shifted_distributions():
    rng = np.random.default_rng(3)
    baseline_values = list(rng.uniform(0, 1, 2000))
    baseline = histogram(baseline_values, bins=10)

    shifted_values = list(rng.uniform(5, 6, 2000))  # completely disjoint range
    score = kl_divergence(baseline, shifted_values)

    assert score > 1.0
