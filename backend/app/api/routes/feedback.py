import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.api_base import APIModel
from app.core.db import get_db
from app.core.models import Inference
from app.ml.schema import FeedbackRequest

router = APIRouter(prefix="/api/v1", tags=["feedback"])


class FeedbackAck(APIModel):
    inference_id: str
    recorded: bool


@router.post("/feedback", response_model=FeedbackAck)
def submit_feedback(payload: FeedbackRequest, db: Session = Depends(get_db)) -> FeedbackAck:
    """Records delayed ground truth for a past prediction - the only path by which the true
    label ever enters the system (see app/ml/label_fn.py). Real recommendation/CTR systems get
    labels this way too: a click (or its absence) arrives well after the prediction was served."""

    try:
        inference_uuid = uuid.UUID(payload.inference_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="inference_id must be a UUID") from exc

    inference = db.execute(select(Inference).where(Inference.id == inference_uuid)).scalars().first()
    if inference is None:
        raise HTTPException(
            status_code=404,
            detail=f"Inference {payload.inference_id} not found (the Kafka consumer may not have written it yet)",
        )

    inference.actual_output = {"label": payload.actual_label}
    db.commit()
    return FeedbackAck(inference_id=payload.inference_id, recorded=True)
