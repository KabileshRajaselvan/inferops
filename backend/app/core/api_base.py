from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    """Base for request/response models. Disables pydantic's `model_` protected-namespace
    warning since this API legitimately has fields like `model_name`/`model_version`/`model_id`
    (they refer to *our* domain model, not pydantic's)."""

    model_config = ConfigDict(protected_namespaces=())
