import hashlib
import json

from app.ml.schema import PredictFeatures


def hash_features(model_name: str, features: PredictFeatures) -> str:
    payload = json.dumps({"model": model_name, "features": features.model_dump()}, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"predict:{digest}"
