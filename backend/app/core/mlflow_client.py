import logging

import mlflow

from app.config import get_settings

logger = logging.getLogger("inferops.mlflow")

_attempted = False
_available = False


def configure_mlflow() -> None:
    """Attempts setup exactly once per process, whether it succeeds or fails - safe to call
    from multiple entrypoints (app startup, training scripts, the evaluator's retraining path)
    without repeating the setup HTTP calls.

    MLflow is an experiment-tracking/registry system of record, not a dependency the serving
    path needs to be up (see README Trade-offs), so a briefly unreachable tracking server must
    not crash app startup or a retraining pass. Failing only once (not retrying on every single
    call) matters just as much: without it, an unreachable server turns every request that
    trains or retrains a model - and every test that boots the app - into a multi-second HTTP
    retry/backoff instead of an instant no-op. is_mlflow_available() lets callers (see
    app/ml/training.py) skip the per-run `mlflow.start_run()` HTTP call entirely once we
    already know the server is down, for the same reason.
    """
    global _attempted, _available
    if _attempted:
        return
    _attempted = True

    settings = get_settings()
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        _available = True
    except Exception:
        logger.warning("MLflow at %s unreachable; continuing without experiment tracking", settings.mlflow_tracking_uri)


def is_mlflow_available() -> bool:
    configure_mlflow()
    return _available
