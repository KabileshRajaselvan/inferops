"""The feature contract for the `ctr-recommender` model.

The PRD's example `/predict` body is freeform (`user_id`, `item_id`, `history`) - fine for a
sketch, but a real model needs a fixed, validated schema. We use an explicit set of engineered
numeric/categorical features instead; a production system would derive these from a feature
store keyed by user_id/item_id, but building a feature-join service is out of scope here (see
README Trade-offs). Redis is still used exactly as the PRD specifies: as the result/feature
cache in front of inference.
"""

from typing import Literal

from pydantic import Field

from app.core.api_base import APIModel

ITEM_CATEGORIES = ("electronics", "books", "fashion", "home", "sports", "beauty")
DEVICES = ("mobile", "desktop", "tablet")


class PredictFeatures(APIModel):
    user_age: float = Field(ge=13, le=100)
    user_tenure_days: float = Field(ge=0, le=5000)
    user_avg_session_min: float = Field(ge=0, le=180)
    user_category_affinity: float = Field(ge=0, le=1)
    item_price: float = Field(ge=0, le=2000)
    item_popularity: float = Field(ge=0, le=1)
    item_category: Literal[ITEM_CATEGORIES]  # type: ignore[valid-type]
    hour_of_day: int = Field(ge=0, le=23)
    history_click_rate: float = Field(ge=0, le=1)
    device: Literal[DEVICES]  # type: ignore[valid-type]


FEATURE_ORDER: list[str] = list(PredictFeatures.model_fields.keys())
CATEGORICAL_FEATURES: list[str] = ["item_category", "device"]
NUMERIC_FEATURES: list[str] = [f for f in FEATURE_ORDER if f not in CATEGORICAL_FEATURES]


class PredictRequest(APIModel):
    model_name: str
    features: PredictFeatures


class Prediction(APIModel):
    score: float
    label: Literal["click", "no_click"]


class PredictResponse(APIModel):
    inference_id: str
    prediction: Prediction
    model_version: str
    latency_ms: int
    deployment_id: str
    cache_hit: bool = False


class FeedbackRequest(APIModel):
    inference_id: str
    actual_label: Literal["click", "no_click"]
