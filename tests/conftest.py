"""Shared fixtures for insurance-deploy tests."""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Generator

import pytest

from insurance_deploy import (
    ModelRegistry, ModelVersion, Experiment, QuoteLogger,
    KPITracker, ModelComparison, ENBPAuditReport,
)


# ---------------------------------------------------------------------------
# Minimal sklearn-like model for testing
# ---------------------------------------------------------------------------

class DummyModel:
    """Minimal predict-compatible model for testing registry."""
    def __init__(self, constant: float = 400.0):
        self.constant = constant

    def predict(self, X):
        import numpy as np
        return np.full(len(X), self.constant)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def registry(tmp_dir):
    return ModelRegistry(tmp_dir / "registry")


@pytest.fixture
def champion_model():
    return DummyModel(constant=400.0)


@pytest.fixture
def challenger_model():
    return DummyModel(constant=420.0)


@pytest.fixture
def champion_mv(registry, champion_model):
    return registry.register(
        champion_model, name="motor", version="1.0",
        metadata={"training_date": "2024-01-01", "features": ["age", "ncd"]}
    )


@pytest.fixture
def challenger_mv(registry, challenger_model):
    return registry.register(
        challenger_model, name="motor", version="2.0",
        metadata={"training_date": "2024-06-01", "features": ["age", "ncd", "usage"]}
    )


@pytest.fixture
def experiment(champion_mv, challenger_mv):
    return Experiment(
        name="motor_v2_vs_v1",
        champion=champion_mv,
        challenger=challenger_mv,
        challenger_pct=0.10,
        mode="shadow",
    )


@pytest.fixture
def logger(tmp_dir):
    return QuoteLogger(tmp_dir / "quotes.db")


@pytest.fixture
def tracker(logger):
    return KPITracker(logger)


@pytest.fixture
def comparison(tracker):
    return ModelComparison(tracker)


@pytest.fixture
def populated_logger(logger, experiment):
    """Logger with synthetic champion/challenger quote/bind/claim data."""
    rng = __import__("random").Random(42)
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Generate 1000 champion, 300 challenger quotes
    all_pids = [f"POL-{i:05d}" for i in range(1300)]
    champion_pids = all_pids[:1000]
    challenger_pids = all_pids[1000:]

    for i, pid in enumerate(champion_pids):
        ts = base_ts + timedelta(days=i // 10)
        price = rng.gauss(400, 50)
        price = max(100, price)
        enbp = price + rng.gauss(10, 5)
        logger.log_quote(
            pid, experiment.name, "champion", "motor:1.0",
            quoted_price=price, enbp=enbp, renewal_flag=True,
            exposure=1.0, timestamp=ts,
        )
        # ~30% bind rate
        if rng.random() < 0.30:
            logger.log_bind(pid, bound_price=price, bound_timestamp=ts + timedelta(hours=2))
            # ~8% claim frequency
            if rng.random() < 0.08:
                logger.log_claim(pid, claim_date=date(2024, 6, 1),
                                 claim_amount=rng.gauss(1500, 500), development_month=12)

    for i, pid in enumerate(challenger_pids):
        ts = base_ts + timedelta(days=i)
        price = rng.gauss(410, 50)
        price = max(100, price)
        enbp = price + rng.gauss(10, 5)
        logger.log_quote(
            pid, experiment.name, "challenger", "motor:2.0",
            quoted_price=price, enbp=enbp, renewal_flag=True,
            exposure=1.0, timestamp=ts,
        )
        if rng.random() < 0.28:  # slightly lower hit rate
            logger.log_bind(pid, bound_price=price, bound_timestamp=ts + timedelta(hours=2))
            if rng.random() < 0.075:
                logger.log_claim(pid, claim_date=date(2024, 6, 1),
                                 claim_amount=rng.gauss(1400, 500), development_month=12)

    return logger
