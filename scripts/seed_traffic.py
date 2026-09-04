#!/usr/bin/env python
"""Simulates real production traffic against a running InferOps stack, end to end:

  1. Sends a "normal" wave of predict requests (holdout_normal.csv - same distribution the
     models were trained on), then delivers their true labels via POST /feedback after a delay
     (simulating real delayed ground truth, e.g. a click event arriving later) - then runs an
     on-demand evaluation and expects accuracy near baseline with low drift.
  2. Sends a "drifted" wave (holdout_drifted.csv - feature/concept shift the models never saw
     in training), delivers feedback the same way, then evaluates again - this time expecting
     a real accuracy drop that crosses the 2% threshold and triggers automatic retraining.

Nothing here peeks at the model's true label-generating function directly (see
app/ml/label_fn.py) - it only ever tells the system the truth through POST /feedback, the same
path a real deployment would use. Run from the repo root against a running `docker compose up`
stack:

    python scripts/seed_traffic.py
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

API_BASE = "http://localhost:8010"
MODEL_NAME = "ctr-recommender"
CONSUMER_FLUSH_WAIT_SECONDS = 5


def send_predict_wave(session: requests.Session, df: pd.DataFrame, *, label: str) -> list[tuple[str, str]]:
    pairs = []
    for i, row in df.iterrows():
        features = row.drop("clicked").to_dict()
        resp = session.post(
            f"{API_BASE}/api/v1/predict", json={"model_name": MODEL_NAME, "features": features}, timeout=10
        )
        resp.raise_for_status()
        inference_id = resp.json()["inference_id"]
        true_label = "click" if row["clicked"] == 1 else "no_click"
        pairs.append((inference_id, true_label))
        if (i + 1) % 200 == 0:
            print(f"  [{label}] sent {i + 1}/{len(df)} predict requests")
    return pairs


def send_feedback(session: requests.Session, pairs: list[tuple[str, str]], *, label: str, max_retries: int = 6) -> None:
    pending = list(pairs)
    for attempt in range(max_retries):
        still_pending = []
        for inference_id, true_label in pending:
            resp = session.post(
                f"{API_BASE}/api/v1/feedback",
                json={"inference_id": inference_id, "actual_label": true_label},
                timeout=10,
            )
            if resp.status_code == 404:
                still_pending.append((inference_id, true_label))
            else:
                resp.raise_for_status()
        pending = still_pending
        if not pending:
            print(f"  [{label}] feedback recorded for all {len(pairs)} rows")
            return
        print(f"  [{label}] {len(pending)} rows not yet consumed from Kafka; retrying in 3s ({attempt + 1}/{max_retries})")
        time.sleep(3)
    print(f"  [{label}] WARNING: {len(pending)} rows never landed in Postgres in time; feedback not recorded")


def run_evaluate(session: requests.Session) -> dict:
    resp = session.post(f"{API_BASE}/api/v1/models/{MODEL_NAME}/evaluate", timeout=120)
    resp.raise_for_status()
    return resp.json()


def print_evaluation(result: dict) -> None:
    for v in result["versions"]:
        drift_summary = ", ".join(f"{feat}={score:.3f}" for feat, score in v["drift"].items()) or "n/a"
        print(
            f"  model v{v['version']}: labeled={v['labeled_count']} "
            f"live_accuracy={v['live_accuracy']} baseline_accuracy={v['baseline_accuracy']} "
            f"retrain_triggered={v['retrain_triggered']}"
        )
        print(f"    drift (KL-divergence): {drift_summary}")
        if "retraining_job" in v:
            print(f"    retraining_job: {v['retraining_job']}")


def run_wave(session: requests.Session, filename: str, *, label: str, rows: int | None) -> None:
    path = DATA_DIR / filename
    if not path.exists():
        raise SystemExit(f"{path} not found - run scripts/generate_dataset.py first")

    df = pd.read_csv(path)
    if rows:
        df = df.head(rows)

    print(f"\n=== {label} wave: {len(df)} rows from {filename} ===")
    pairs = send_predict_wave(session, df, label=label)

    print(f"  waiting {CONSUMER_FLUSH_WAIT_SECONDS}s for the Kafka consumer to flush to Postgres...")
    time.sleep(CONSUMER_FLUSH_WAIT_SECONDS)

    send_feedback(session, pairs, label=label)

    print("  running on-demand evaluation (POST /models/{name}/evaluate)...")
    print_evaluation(run_evaluate(session))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=None, help="Limit rows per wave (default: full file)")
    parser.add_argument("--skip-drifted", action="store_true", help="Only send the normal wave")
    args = parser.parse_args()

    session = requests.Session()
    run_wave(session, "holdout_normal.csv", label="normal", rows=args.rows)
    if not args.skip_drifted:
        run_wave(session, "holdout_drifted.csv", label="drifted", rows=args.rows)


if __name__ == "__main__":
    main()
