"""Keeps the `inferences` range-partitioned table supplied with daily partitions a few days
ahead, so writes never fall through to the DEFAULT partition in normal operation (the DEFAULT
partition still exists as a safety net - see the initial Alembic migration)."""

import logging
from datetime import date, timedelta

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("inferops.partitions")


def ensure_partitions(db: Session, *, lookahead_days: int = 3, lookbehind_days: int = 1) -> None:
    today = date.today()
    start = today - timedelta(days=lookbehind_days)
    for offset in range(-lookbehind_days, lookahead_days + 1):
        day = today + timedelta(days=offset)
        _ensure_partition_for_day(db, day)
    db.commit()
    logger.debug("Ensured inferences partitions from %s to %s", start, today + timedelta(days=lookahead_days))


def _ensure_partition_for_day(db: Session, day: date) -> None:
    # Postgres can't infer bind-parameter types inside a DDL "FOR VALUES" clause (it's not
    # planned like a regular DML statement), so the date literals are interpolated directly.
    # Safe here: `day`/`next_day` are always derived from `date.today()` + a timedelta, never
    # from external input.
    next_day = day + timedelta(days=1)
    partition_name = f"inferences_{day.isoformat().replace('-', '_')}"
    db.execute(
        text(
            f'CREATE TABLE IF NOT EXISTS "{partition_name}" PARTITION OF inferences '
            f"FOR VALUES FROM ('{day.isoformat()}') TO ('{next_day.isoformat()}')"
        )
    )
