from app.ml.cache_keys import hash_features
from app.ml.schema import PredictFeatures

BASE_KWARGS = dict(
    user_age=30,
    user_tenure_days=100,
    user_avg_session_min=15.0,
    user_category_affinity=0.5,
    item_price=49.99,
    item_popularity=0.6,
    item_category="electronics",
    hour_of_day=14,
    history_click_rate=0.3,
    device="mobile",
)


def test_identical_features_hash_to_the_same_key():
    a = PredictFeatures(**BASE_KWARGS)
    b = PredictFeatures(**BASE_KWARGS)
    assert hash_features("ctr-recommender", a) == hash_features("ctr-recommender", b)


def test_different_features_hash_to_different_keys():
    a = PredictFeatures(**BASE_KWARGS)
    b = PredictFeatures(**{**BASE_KWARGS, "item_price": 199.99})
    assert hash_features("ctr-recommender", a) != hash_features("ctr-recommender", b)


def test_different_model_name_hashes_to_different_key():
    a = PredictFeatures(**BASE_KWARGS)
    assert hash_features("ctr-recommender", a) != hash_features("other-model", a)
