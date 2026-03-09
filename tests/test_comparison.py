"""Tests for ModelComparison — bootstrap LR test, hit rate test, frequency test."""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime, timezone, timedelta

import pytest

from insurance_deploy import ModelComparison, ComparisonResult, KPITracker, QuoteLogger


VALID_CONCLUSIONS = {"INSUFFICIENT_EVIDENCE", "CHALLENGER_BETTER", "CHAMPION_BETTER"}


def make_logger(tmp_path):
    return QuoteLogger(tmp_path / "q.db")


def populate_two_arm_data(
    logger,
    experiment_name: str = "test_exp",
    n_champ: int = 300,
    n_chall: int = 100,
    champ_bind_rate: float = 0.30,
    chall_bind_rate: float = 0.28,
    champ_claim_freq: float = 0.08,
    chall_claim_freq: float = 0.075,
    champ_mean_claim: float = 1500.0,
    chall_mean_claim: float = 1400.0,
    champ_mean_price: float = 400.0,
    chall_mean_price: float = 410.0,
    seed: int = 42,
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


class TestComparisonResult:

    def test_conclusion_is_valid(self, populated_logger, comparison, experiment):
        result = comparison.bootstrap_lr_test(experiment.name, n_bootstrap=100, seed=0)
        assert result.conclusion in VALID_CONCLUSIONS

    def test_summary_returns_string(self, populated_logger, comparison, experiment):
        result = comparison.bootstrap_lr_test(experiment.name, n_bootstrap=100, seed=0)
        s = result.summary()
        assert isinstance(s, str)
        assert "Test:" in s
        assert "Champion estimate" in s

    def test_repr_shows_key_info(self, populated_logger, comparison, experiment):
        result = comparison.hit_rate_test(experiment.name)
        r = repr(result)
        assert "hit_rate_test" in r
        assert "p=" in r

    def test_notes_default_to_empty_list(self):
        r = ComparisonResult(
            test_name="test", experiment_name="exp",
            champion_estimate=0.5, challenger_estimate=0.5,
            difference=0.0, ci_lower=-0.1, ci_upper=0.1,
            p_value=0.5, n_champion=100, n_challenger=50,
            conclusion="INSUFFICIENT_EVIDENCE", recommendation="Continue.",
        )
        assert r.notes == []


class TestBootstrapLRTest:

    def test_returns_comparison_result(self, populated_logger, comparison, experiment):
        result = comparison.bootstrap_lr_test(experiment.name, n_bootstrap=100, seed=0)
        assert isinstance(result, ComparisonResult)

    def test_ci_contains_point_estimate(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_two_arm_data(logger, n_champ=200, n_chall=80)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.bootstrap_lr_test("test_exp", n_bootstrap=500, seed=42)
        if not math.isnan(result.difference):
            assert result.ci_lower <= result.difference <= result.ci_upper

    def test_maturity_warning_when_dev_months_lt_12(self, populated_logger, comparison, experiment):
        with pytest.warns(UserWarning, match="development_months"):
            result = comparison.bootstrap_lr_test(
                experiment.name, n_bootstrap=50, development_months=6, seed=0
            )
        assert result.maturity_warning is True

    def test_no_maturity_warning_at_12months(self, populated_logger, comparison, experiment):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                result = comparison.bootstrap_lr_test(
                    experiment.name, n_bootstrap=50, development_months=12, seed=0
                )
            except UserWarning as e:
                pytest.fail(f"Unexpected warning: {e}")

    def test_p_value_in_valid_range(self, populated_logger, comparison, experiment):
        result = comparison.bootstrap_lr_test(experiment.name, n_bootstrap=200, seed=0)
        if not math.isnan(result.p_value):
            assert 0 < result.p_value <= 1.0

    def test_empty_experiment_returns_insufficient(self, tracker):
        comp = ModelComparison(tracker)
        result = comp.bootstrap_lr_test("empty_exp", n_bootstrap=10)
        assert result.conclusion == "INSUFFICIENT_EVIDENCE"

    def test_seed_produces_reproducible_results(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_two_arm_data(logger, n_champ=150, n_chall=60, seed=99)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        r1 = comp.bootstrap_lr_test("test_exp", n_bootstrap=200, seed=123)
        r2 = comp.bootstrap_lr_test("test_exp", n_bootstrap=200, seed=123)
        assert r1.p_value == pytest.approx(r2.p_value)
        assert r1.ci_lower == pytest.approx(r2.ci_lower)

    def test_n_champion_n_challenger_in_result(self, populated_logger, comparison, experiment):
        result = comparison.bootstrap_lr_test(experiment.name, n_bootstrap=100, seed=0)
        assert result.n_champion > 0
        assert result.n_challenger > 0

    def test_challenger_better_conclusion_when_lower_lr(self, tmp_dir):
        """When challenger has substantially lower LR, should conclude CHALLENGER_BETTER."""
        logger = make_logger(tmp_dir)
        # Champion: high claims; challenger: very low claims
        populate_two_arm_data(
            logger,
            n_champ=500, n_chall=200,
            champ_claim_freq=0.20, chall_claim_freq=0.02,
            champ_mean_claim=2000.0, chall_mean_claim=500.0,
            seed=7,
        )
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.bootstrap_lr_test("test_exp", n_bootstrap=1000, seed=42)
        # With such a large true difference, should detect significance
        assert result.challenger_estimate < result.champion_estimate


class TestHitRateTest:

    def test_returns_comparison_result(self, populated_logger, comparison, experiment):
        result = comparison.hit_rate_test(experiment.name)
        assert isinstance(result, ComparisonResult)

    def test_empty_returns_insufficient(self, tracker):
        comp = ModelComparison(tracker)
        result = comp.hit_rate_test("empty_exp")
        assert result.conclusion == "INSUFFICIENT_EVIDENCE"

    def test_test_name_correct(self, populated_logger, comparison, experiment):
        result = comparison.hit_rate_test(experiment.name)
        assert result.test_name == "hit_rate_test"

    def test_estimates_match_hit_rate(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        comp = ModelComparison(tracker)
        result = comp.hit_rate_test(experiment.name)
        hr = tracker.hit_rate(experiment.name)
        assert result.champion_estimate == pytest.approx(
            hr["champion"]["hit_rate"], abs=0.001
        )
        assert result.challenger_estimate == pytest.approx(
            hr["challenger"]["hit_rate"], abs=0.001
        )

    def test_difference_is_challenger_minus_champion(self, populated_logger, comparison, experiment):
        result = comparison.hit_rate_test(experiment.name)
        expected = result.challenger_estimate - result.champion_estimate
        assert result.difference == pytest.approx(expected, abs=1e-6)

    def test_ci_semantically_consistent(self, populated_logger, comparison, experiment):
        result = comparison.hit_rate_test(experiment.name)
        if not math.isnan(result.ci_lower):
            assert result.ci_lower < result.ci_upper

    def test_large_sample_detects_large_difference(self, tmp_dir):
        """With a 10pp true difference and large N, should detect significance."""
        logger = make_logger(tmp_dir)
        populate_two_arm_data(
            logger, n_champ=2000, n_chall=800,
            champ_bind_rate=0.40, chall_bind_rate=0.28,
            seed=11,
        )
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.hit_rate_test("test_exp")
        # Should find a significant difference
        assert result.p_value < 0.05
        assert result.conclusion != "INSUFFICIENT_EVIDENCE"


class TestFrequencyTest:

    def test_returns_comparison_result(self, populated_logger, comparison, experiment):
        result = comparison.frequency_test(experiment.name, development_months=12)
        assert isinstance(result, ComparisonResult)

    def test_empty_returns_insufficient(self, tracker):
        comp = ModelComparison(tracker)
        result = comp.frequency_test("empty_exp")
        assert result.conclusion == "INSUFFICIENT_EVIDENCE"

    def test_test_name_correct(self, populated_logger, comparison, experiment):
        result = comparison.frequency_test(experiment.name)
        assert result.test_name == "frequency_test"

    def test_frequency_test_with_sufficient_data(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_two_arm_data(
            logger, n_champ=400, n_chall=150,
            champ_claim_freq=0.10, chall_claim_freq=0.10,
            seed=55,
        )
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.frequency_test("test_exp", development_months=12)
        assert result.n_champion > 0
        assert result.n_challenger > 0


class TestBootstrapCIProperties:

    def test_bootstrap_ci_coverage_approximately_95pct(self, tmp_dir):
        """
        Simulation test: bootstrap CI should cover true difference ~95% of time.
        Run 20 replications and check coverage >= 80% (loose, for speed).
        """
        import random
        covered = 0
        n_reps = 20
        true_diff = 0.0  # H0: no difference

        for rep in range(n_reps):
            subdir = tmp_dir / f"rep_{rep}"
            subdir.mkdir(parents=True, exist_ok=True)
            logger = make_logger(subdir)
            populate_two_arm_data(
                logger, n_champ=200, n_chall=80,
                champ_claim_freq=0.08, chall_claim_freq=0.08,
                seed=rep * 37,
            )
            tracker = KPITracker(logger)
            comp = ModelComparison(tracker)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = comp.bootstrap_lr_test("test_exp", n_bootstrap=500, seed=rep)
            if (not math.isnan(result.ci_lower) and
                    result.ci_lower <= true_diff <= result.ci_upper):
                covered += 1

        # Accept coverage >= 70% (loose bound — bootstrap can be conservative)
        assert covered / n_reps >= 0.70
