#!/usr/bin/env python
"""Hits POST /batch-predict with the 100K-row synthetic CSV and reports real throughput -
the number that backs the PRD's "batch prediction for 100K+ samples" success metric in the
README. Run from the repo root against a running stack:

    python scripts/load_test_batch.py
"""

import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
API_BASE = "http://localhost:8010"
MODEL_NAME = "ctr-recommender"


def main() -> None:
    path = DATA_DIR / "batch_100k.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run scripts/generate_dataset.py first")

    print(f"Uploading {path.name} to POST /api/v1/batch-predict ...")
    wall_start = time.perf_counter()
    with open(path, "rb") as f:
        resp = requests.post(
            f"{API_BASE}/api/v1/batch-predict",
            params={"model_name": MODEL_NAME},
            files={"file": (path.name, f, "text/csv")},
            timeout=600,
        )
    wall_elapsed_s = time.perf_counter() - wall_start
    resp.raise_for_status()
    body = resp.json()

    print(f"\nModel: {body['model_name']} v{body['model_version']} (deployment {body['deployment_id']})")
    print(f"Rows scored: {body['rows']}")
    print(f"Click rate: {body['click_rate']:.3f}")
    print(f"Server-side latency: {body['total_latency_ms']} ms ({body['rows_per_second']:.0f} rows/sec)")
    print(f"Wall-clock (incl. upload over HTTP): {wall_elapsed_s:.2f}s ({body['rows'] / wall_elapsed_s:.0f} rows/sec)")


if __name__ == "__main__":
    main()
