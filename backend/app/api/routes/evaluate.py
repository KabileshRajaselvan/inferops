from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.evaluation.checks import run_checks_for_model

router = APIRouter(prefix="/api/v1/models", tags=["evaluate"])


@router.post("/{name}/evaluate")
def evaluate_model(name: str, db: Session = Depends(get_db)) -> dict:
    """Runs the drift/accuracy/retraining-trigger checks for `name` synchronously and returns
    the result immediately, instead of waiting for the evaluator service's next scheduled pass.
    Used by demo/seed scripts (and available for manual ops use) so the loop isn't purely
    time-based."""

    return run_checks_for_model(db, name)
