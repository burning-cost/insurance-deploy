"""Regression tests for P0-1 and P0-2 bugs."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timezone, timedelta

import pytest

from insurance_deploy import ModelComparison, KPITracker, QuoteLogger, Experiment
from insurance_deploy.registry import ModelRegistry


class _DummyModel:
    """Module-level model stub for joblib pickling in regression tests."""
    def predict(self, X):
        import numpy as np
        return np.ones(len(X))


def make_logger(tmp_path):
    return QuoteLogger(tmp_path / "q.db")


def populate_two_arm_data(
    logger,
    experiment_name="test_exp",
    n_champ=300,
    n_chall=100,
    champ_bind_rate=0.30,
    chall_bind_rate=0.28,
    champ_claim_freq=0.08,
    chall_claim_freq=0.075,
    champ_mean_claim=1500.0,
    chall_mean_claim=1400.0,
    champ_mean_price=400.0,
    chall_mean_price=410.0,
    seed=42,
):
    import random
    rng = random.Random(seed)
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for i in range(n_champ):
        pid = f"CH-{i:05d}"
        ts = base_ts + timedelta(days=i // 5)
        price = max(100, rng.gauss(champ_mean_price, 50))
        logger.log_quote(pid, experiment_name, "champion", "motor:1.0",
                         quoted_price=price, enbp=price + 20, renewal_flag=True,
                         exposure=1.0, timestamp=ts)
        if rng.random() < champ_bind_rate:
            logger.log_bind(pid, price, ts + timedelta(hours=1))
            if rng.random() < champ_claim_freq:
                logger.log_claim(pid, date(2024, 6, 1),
                                 max(0, rng.gauss(champ_mean_claim, 500)), 12)

    for i in range(n_chall):
        pid = f"CL-{i:05d}"
        ts = base_ts + timedelta(days=i // 2)
        price = max(100, rng.gauss(chall_mean_price, 50))
        logger.log_quote(pid, experiment_name, "challenger", "motor:2.0",
                         quoted_price=price, enbp=price + 20, renewal_flag=True,
                         exposure=1.0, timestamp=ts)
        if rng.random() < chall_bind_rate:
            logger.log_bind(pid, price, ts + timedelta(hours=1))
            if rng.random() < chall_claim_freq:
                logger.log_claim(pid, date(2024, 6, 1),
                                 max(0, rng.gauss(chall_mean_claim, 500)), 12)


class TestP01BootstrapPValueRegression:
    """
    P0-1 regression: bootstrap p-value was inverted (>= instead of <=).
    A genuinely better challenger must conclude CHALLENGER_BETTER.
    """

    def test_challenger_better_concludes_challenger_better(self, tmp_dir):
        """
        Large true LR advantage for challenger must produce CHALLENGER_BETTER.
        Before the fix, the inverted >= comparison meant that a challenger with
        much lower LR would get a p-value near 1.0 instead of near 0.0.
        """
        logger = make_logger(tmp_dir)
        populate_two_arm_data(
            logger,
            n_champ=500, n_chall=200,
            champ_claim_freq=0.20, chall_claim_freq=0.02,
            champ_mean_claim=2000.0, chall_mean_claim=500.0,
            seed=7,
        )
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.bootstrap_lr_test("test_exp", n_bootstrap=2000, seed=42)
        assert result.challenger_estimate < result.champion_estimate
        assert result.conclusion == "CHALLENGER_BETTER", (
            "Challenger with dramatically lower LR must conclude CHALLENGER_BETTER. "
            "P0-1 regression: p-value comparison was >= instead of <=."
        )

    def test_p_value_low_when_challenger_clearly_better(self, tmp_dir):
        """p-value should be low (<0.05) when challenger is clearly better."""
        logger = make_logger(tmp_dir)
        populate_two_arm_data(
            logger,
            n_champ=500, n_chall=200,
            champ_claim_freq=0.20, chall_claim_freq=0.02,
            champ_mean_claim=2000.0, chall_mean_claim=500.0,
            seed=7,
        )
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.bootstrap_lr_test("test_exp", n_bootstrap=2000, seed=42)
        assert result.p_value < 0.05, (
            f"p={result.p_value:.4f} should be < 0.05 when challenger is much better. "
            "Before P0-1 fix, this would have been near 1.0."
        )


class TestP02ShadowModelRegression:
    """
    P0-2 regression: shadow_model() returned challenger for only ~10% of policies
    (the hash-routed bucket). It should return challenger for ALL policies in shadow mode.
    """

    def test_shadow_model_returns_challenger_for_all_policies(self, tmp_dir):
        """In shadow mode, shadow_model must return challenger for every policy."""
        reg = ModelRegistry(tmp_dir / "reg")

        champ_mv = reg.register(_DummyModel(), name="motor", version="1.0",
                                metadata={"training_date": "2024-01-01"})
        chall_mv = reg.register(_DummyModel(), name="motor", version="2.0",
                                metadata={"training_date": "2024-06-01"})

        exp = Experiment(
            name="shadow_regression_test",
            champion=champ_mv,
            challenger=chall_mv,
            challenger_pct=0.10,
            mode="shadow",
        )

        pids = [f"POL-{i:06d}" for i in range(1000)]
        shadow_results = [exp.shadow_model(pid).version_id for pid in pids]
        assert all(v == chall_mv.version_id for v in shadow_results), (
            "shadow_model must return challenger for ALL policies in shadow mode. "
            "P0-2 regression: previously only returned challenger for ~10% of policies."
        )

    def test_shadow_model_all_policies_get_challenger_scored(self, tmp_dir):
        """Verify no policies get champion in shadow mode (champion is already live)."""
        reg = ModelRegistry(tmp_dir / "reg")

        champ_mv = reg.register(_DummyModel(), name="motor", version="1.0",
                                metadata={})
        chall_mv = reg.register(_DummyModel(), name="motor", version="2.0",
                                metadata={})

        exp = Experiment(
            name="shadow_full_coverage",
            champion=champ_mv,
            challenger=chall_mv,
            challenger_pct=0.05,
            mode="shadow",
        )

        pids = [f"X-{i}" for i in range(500)]
        champ_shadow_count = sum(
            1 for pid in pids
            if exp.shadow_model(pid).version_id == champ_mv.version_id
        )
        assert champ_shadow_count == 0, (
            f"{champ_shadow_count}/500 policies shadow-scored by champion instead of challenger. "
            "In shadow mode, challenger should score every policy."
        )
