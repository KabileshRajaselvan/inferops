"""The ground-truth generative function for the synthetic CTR dataset.

This module is intentionally never imported by the serving path (app/main.py, app/api/*,
app/evaluation/*) - only by offline scripts (scripts/generate_dataset.py to build the training
set, scripts/seed_traffic.py to simulate delayed user feedback arriving via POST /feedback) and
by test fixtures (to build realistic synthetic data, which is likewise "offline"). That boundary
is what makes the accuracy/drift/retraining loop honest: the running system only ever learns the
true label through the same "production" path a real deployment would use (delayed feedback
events), never by peeking at this function directly.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def click_probability(df: pd.DataFrame, *, drift: bool = False) -> np.ndarray:
    """The true P(click) given features. `drift=True` simulates a live concept/covariate
    shift (used to generate a second wave of "production" traffic that a model trained only
    on the non-drifted distribution will genuinely perform worse on)."""

    device_bonus = df["device"].map({"mobile": 0.10, "desktop": 0.0, "tablet": -0.05}).to_numpy()
    peak_hour = df["hour_of_day"].between(18, 22).to_numpy().astype(float)

    price_centered = (df["item_price"].to_numpy() - 50.0) / 50.0
    price_weight = -0.02 if not drift else -0.09  # drifted traffic is far more price-sensitive
    popularity_weight = 0.4 if not drift else 0.15  # popularity signal decays under drift

    logit = (
        -0.4
        + 0.6 * df["user_category_affinity"].to_numpy()
        + popularity_weight * df["item_popularity"].to_numpy()
        + price_weight * price_centered
        + 0.5 * df["history_click_rate"].to_numpy()
        + device_bonus
        + 0.15 * peak_hour
    )
    if drift:
        logit -= 0.35  # baseline click-through rate also drops (real-world seasonal shift)

    return _sigmoid(logit)


def sample_labels(df: pd.DataFrame, *, drift: bool = False, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(RNG_SEED)
    p = click_probability(df, drift=drift)
    noise = rng.normal(0, 0.05, size=len(df))
    p_noisy = np.clip(p + noise, 0.0, 1.0)
    return rng.binomial(1, p_noisy)
