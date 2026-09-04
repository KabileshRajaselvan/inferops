import numpy as np
import pandas as pd

from app.ml.schema import DEVICES, FEATURE_ORDER, ITEM_CATEGORIES
from app.ml.training import build_pipeline


def _toy_frame(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "user_age": rng.integers(18, 70, n).astype(float),
            "user_tenure_days": rng.integers(0, 2000, n).astype(float),
            "user_avg_session_min": rng.uniform(1, 60, n),
            "user_category_affinity": rng.uniform(0, 1, n),
            "item_price": rng.uniform(5, 300, n),
            "item_popularity": rng.uniform(0, 1, n),
            "item_category": rng.choice(ITEM_CATEGORIES, n),
            "hour_of_day": rng.integers(0, 24, n),
            "history_click_rate": rng.uniform(0, 1, n),
            "device": rng.choice(DEVICES, n),
        }
    )


def test_logistic_regression_pipeline_fits_and_predicts():
    df = _toy_frame()
    labels = np.random.default_rng(0).integers(0, 2, len(df))

    pipeline = build_pipeline("logistic_regression")
    pipeline.fit(df[FEATURE_ORDER], labels)
    proba = pipeline.predict_proba(df[FEATURE_ORDER])

    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gradient_boosting_pipeline_fits_and_predicts():
    df = _toy_frame()
    labels = np.random.default_rng(0).integers(0, 2, len(df))

    pipeline = build_pipeline("gradient_boosting")
    pipeline.fit(df[FEATURE_ORDER], labels)
    proba = pipeline.predict_proba(df[FEATURE_ORDER])

    assert proba.shape == (len(df), 2)
