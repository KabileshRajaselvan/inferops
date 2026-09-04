"""Entrypoint for the `evaluator` service: `python -m app.evaluator`.

Runs its own tiny Prometheus HTTP server (separate from the api service's /metrics) because
model_accuracy / data_drift_score / retraining_jobs_total are set in *this* process - the
prometheus_client registry is per-process, so these gauges are otherwise invisible to a scraper
hitting only the api service. Prometheus is configured with both as separate scrape targets."""

import asyncio
import logging

from prometheus_client import start_http_server

from app.config import get_settings

logging.basicConfig(level=get_settings().log_level)

from app.evaluation.loop import run_forever  # noqa: E402

if __name__ == "__main__":
    settings = get_settings()
    start_http_server(settings.evaluator_metrics_port)
    asyncio.run(run_forever())
