"""Shared fixtures for insurance-deploy tests."""

from __future__ import annotations

import tempfile
import warnings
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

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
        return np.full(len(np.atleast_1d(X)), self.constant)


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
    """Logger pre-filled with 200 champion + 60 challenger records.

    Scaled down from the original 1000/300 to keep fixture setup under 2s on
    ARM hardware.  The fixture shares the same ``logger`` instance used by
    ``tracker``/``comparison``, so tests that receive all three see consistent
    data.

    Warnings are suppressed during population to avoid ~30 UserWarning
    emissions (one per ENBP breach) that were the other source of overhead.
    """
    rng = __import__("random").Random(42)
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    all_pids = [f"POL-{i:05d}" for i in range(260)]
    champion_pids = all_pids[:200]
    challenger_pids = all_pids[200:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i, pid in enumerate(champion_pids):
            ts = base_ts + timedelta(days=i // 10)
            price = max(100.0, rng.gauss(400, 50))
            enbp = price + rng.gauss(10, 5)
            logger.log_quote(
                pid, experiment.name, "champion", "motor:1.0",
                quoted_price=price, enbp=enbp, renewal_flag=True,
                exposure=1.0, timestamp=ts,
            )
            if rng.random() < 0.30:
                logger.log_bind(pid, bound_price=price,
                                bound_timestamp=ts + timedelta(hours=2))
                if rng.random() < 0.08:
                    logger.log_claim(pid, claim_date=date(2024, 6, 1),
                                     claim_amount=rng.gauss(1500, 500),
                                     development_month=12)

        for i, pid in enumerate(challenger_pids):
            ts = base_ts + timedelta(days=i)
            price = max(100.0, rng.gauss(410, 50))
            enbp = price + rng.gauss(10, 5)
            logger.log_quote(
                pid, experiment.name, "challenger", "motor:2.0",
                quoted_price=price, enbp=enbp, renewal_flag=True,
                exposure=1.0, timestamp=ts,
            )
            if rng.random() < 0.28:
                logger.log_bind(pid, bound_price=price,
                                bound_timestamp=ts + timedelta(hours=2))
                if rng.random() < 0.075:
                    logger.log_claim(pid, claim_date=date(2024, 6, 1),
                                     claim_amount=rng.gauss(1400, 500),
                                     development_month=12)

    return logger
