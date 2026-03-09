"""Tests for KPITracker."""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime, timezone, timedelta

import pytest

from insurance_deploy import KPITracker, QuoteLogger


def make_logger(tmp_path):
    return QuoteLogger(tmp_path / "q.db")


class TestQuoteVolume:

    def test_empty_experiment_returns_nan(self, tracker):
        vol = tracker.quote_volume("nonexistent")
        assert vol["champion"]["n"] == 0
        assert math.isnan(vol["champion"]["mean_price"])

    def test_quote_volume_counts(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i in range(10):
            logger.log_quote(f"P-{i}", "exp", "champion", "m:1", 400.0 + i)
        for i in range(3):
            logger.log_quote(f"C-{i}", "exp", "challenger", "m:2", 420.0 + i)
        tracker = KPITracker(logger)
        vol = tracker.quote_volume("exp")
        assert vol["champion"]["n"] == 10
        assert vol["challenger"]["n"] == 3

    def test_mean_price_correct(self, tmp_dir):
        logger = make_logger(tmp_dir)
        # Champion: prices 100, 200, 300 -> mean 200
        logger.log_quote("P1", "exp", "champion", "m:1", 100.0)
        logger.log_quote("P2", "exp", "champion", "m:1", 200.0)
        logger.log_quote("P3", "exp", "champion", "m:1", 300.0)
        tracker = KPITracker(logger)
        vol = tracker.quote_volume("exp")
        assert vol["champion"]["mean_price"] == pytest.approx(200.0)

    def test_percentiles(self, tmp_dir):
        import numpy as np
        logger = make_logger(tmp_dir)
        prices = list(range(1, 101))  # 1..100
        for i, p in enumerate(prices):
            logger.log_quote(f"P{i}", "exp", "champion", "m:1", float(p))
        tracker = KPITracker(logger)
        vol = tracker.quote_volume("exp")
        assert vol["champion"]["p25_price"] == pytest.approx(25.75, abs=1.0)
        assert vol["champion"]["p75_price"] == pytest.approx(75.25, abs=1.0)


class TestENBPCompliance:

    def test_all_compliant(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i in range(5):
            logger.log_quote(f"P{i}", "exp", "champion", "m:1",
                             400.0, enbp=420.0, renewal_flag=True)
        tracker = KPITracker(logger)
        comp = tracker.enbp_compliance("exp")
        assert comp["champion"]["compliant"] == 5
        assert comp["champion"]["breaches"] == 0
        assert comp["champion"]["compliance_rate"] == pytest.approx(1.0)

    def test_breach_counted(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0, enbp=420.0, renewal_flag=True)
        with pytest.warns(UserWarning):
            logger.log_quote("P2", "exp", "champion", "m:1", 430.0, enbp=420.0, renewal_flag=True)
        tracker = KPITracker(logger)
        comp = tracker.enbp_compliance("exp")
        assert comp["champion"]["compliant"] == 1
        assert comp["champion"]["breaches"] == 1
        assert comp["champion"]["compliance_rate"] == pytest.approx(0.5)

    def test_nb_quotes_excluded(self, tmp_dir):
        logger = make_logger(tmp_dir)
        # New business: no ENBP required
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0, renewal_flag=False)
        tracker = KPITracker(logger)
        comp = tracker.enbp_compliance("exp")
        assert comp["champion"]["renewal_quotes"] == 0


class TestHitRate:

    def test_empty_returns_nan(self, tracker):
        hr = tracker.hit_rate("empty_exp")
        assert math.isnan(hr["champion"]["hit_rate"])

    def test_hit_rate_calculation(self, tmp_dir):
        logger = make_logger(tmp_dir)
        # 4 champion quotes, 2 binds -> 50% hit rate
        for i in range(4):
            logger.log_quote(f"P{i}", "exp", "champion", "m:1", 400.0)
        for i in range(2):
            logger.log_bind(f"P{i}", 400.0)
        tracker = KPITracker(logger)
        hr = tracker.hit_rate("exp")
        assert hr["champion"]["quoted"] == 4
        assert hr["champion"]["bound"] == 2
        assert hr["champion"]["hit_rate"] == pytest.approx(0.50)

    def test_each_policy_counted_once(self, tmp_dir):
        """Multiple quotes for same policy should count as one quote."""
        logger = make_logger(tmp_dir)
        # Policy POL-001 gets two quotes (e.g. re-quote)
        logger.log_quote("POL-001", "exp", "champion", "m:1", 400.0)
        logger.log_quote("POL-001", "exp", "champion", "m:1", 390.0)
        logger.log_quote("POL-002", "exp", "champion", "m:1", 400.0)
        tracker = KPITracker(logger)
        hr = tracker.hit_rate("exp")
        assert hr["champion"]["quoted"] == 2  # unique policies

    def test_hit_rate_champion_and_challenger(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        hr = tracker.hit_rate(experiment.name)
        # Both arms should have positive hit rates
        assert hr["champion"]["hit_rate"] > 0
        assert hr["challenger"]["hit_rate"] > 0
        # Champion should have more quotes
        assert hr["champion"]["quoted"] > hr["challenger"]["quoted"]

    def test_cohort_filter(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        hr_all = tracker.hit_rate(experiment.name)
        hr_champ = tracker.hit_rate(experiment.name, cohort="champion")
        assert "challenger" not in hr_champ
        assert hr_champ["champion"]["quoted"] == hr_all["champion"]["quoted"]


class TestGWP:

    def test_gwp_uses_bound_price(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        logger.log_bind("P1", bound_price=395.0)  # slightly different from quoted
        tracker = KPITracker(logger)
        gwp = tracker.gwp("exp")
        assert gwp["champion"]["total_gwp"] == pytest.approx(395.0)

    def test_gwp_sums_across_policies(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i, price in enumerate([300.0, 400.0, 500.0]):
            logger.log_quote(f"P{i}", "exp", "champion", "m:1", price)
            logger.log_bind(f"P{i}", price)
        tracker = KPITracker(logger)
        gwp = tracker.gwp("exp")
        assert gwp["champion"]["total_gwp"] == pytest.approx(1200.0)
        assert gwp["champion"]["mean_gwp"] == pytest.approx(400.0)
        assert gwp["champion"]["bound_policies"] == 3

    def test_unbound_policies_excluded(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        logger.log_quote("P2", "exp", "champion", "m:1", 500.0)
        logger.log_bind("P1", 400.0)
        # P2 did not bind
        tracker = KPITracker(logger)
        gwp = tracker.gwp("exp")
        assert gwp["champion"]["bound_policies"] == 1
        assert gwp["champion"]["total_gwp"] == pytest.approx(400.0)


class TestFrequency:

    def test_frequency_warns_immature(self, tracker):
        with pytest.warns(UserWarning, match="IBNR"):
            tracker.frequency("exp", development_months=3)

    def test_frequency_no_warning_at_12months(self, tracker):
        # Should not warn at 12 months
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                tracker.frequency("exp", development_months=12)
            except UserWarning:
                pytest.fail("Unexpected warning at 12 months development")
            except Exception:
                pass  # other errors are ok (no data)

    def test_frequency_calculation(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        freq = tracker.frequency(experiment.name, development_months=12, warn_immature=False)
        # Both arms should have frequency > 0
        assert freq["champion"]["frequency"] > 0
        assert freq["challenger"]["frequency"] > 0
        # Claim count matches
        assert freq["champion"]["claim_count"] > 0

    def test_frequency_zero_when_no_claims(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i in range(5):
            logger.log_quote(f"P{i}", "exp", "champion", "m:1", 400.0)
            logger.log_bind(f"P{i}", 400.0)
        tracker = KPITracker(logger)
        freq = tracker.frequency("exp", development_months=12, warn_immature=False)
        assert freq["champion"]["claim_count"] == 0
        assert freq["champion"]["frequency"] == pytest.approx(0.0)


class TestLossRatio:

    def test_lr_warns_immature(self, tracker):
        with pytest.warns(UserWarning, match="immature"):
            tracker.loss_ratio("exp", development_months=6)

    def test_lr_calculation(self, tmp_dir):
        logger = make_logger(tmp_dir)
        # Set up: premium 400, 1 claim of 200 -> LR = 200/400 = 0.5
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        logger.log_bind("P1", 400.0)
        logger.log_claim("P1", date(2024, 6, 1), 200.0, development_month=12)
        tracker = KPITracker(logger)
        lr = tracker.loss_ratio("exp", development_months=12)
        assert lr["champion"]["loss_ratio"] == pytest.approx(0.5)

    def test_lr_zero_when_no_claims(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i in range(3):
            logger.log_quote(f"P{i}", "exp", "champion", "m:1", 400.0)
            logger.log_bind(f"P{i}", 400.0)
        tracker = KPITracker(logger)
        lr = tracker.loss_ratio("exp", development_months=12)
        assert lr["champion"]["loss_ratio"] == pytest.approx(0.0)

    def test_lr_from_populated_logger(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        lr = tracker.loss_ratio(experiment.name, development_months=12)
        # Realistic motor LR should be in plausible range
        champ_lr = lr["champion"]["loss_ratio"]
        assert not math.isnan(champ_lr)
        assert 0 < champ_lr < 2.0

    def test_lr_maturity_warning_flag(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        logger.log_bind("P1", 400.0)
        tracker = KPITracker(logger)
        with pytest.warns(UserWarning):
            lr = tracker.loss_ratio("exp", development_months=6)
        assert lr["champion"]["maturity_warning"] is True


class TestPowerAnalysis:

    def test_power_analysis_returns_dict(self, tracker):
        pa = tracker.power_analysis("exp")
        assert isinstance(pa, dict)

    def test_power_analysis_keys(self, tracker):
        pa = tracker.power_analysis("exp")
        assert "hr_required_n_per_arm" in pa
        assert "lr_required_n_per_arm" in pa
        assert "lr_total_months_with_development" in pa
        assert "notes" in pa

    def test_power_analysis_sample_sizes_positive(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        pa = tracker.power_analysis(experiment.name)
        assert pa["hr_required_n_per_arm"] > 0
        assert pa["lr_required_n_per_arm"] > 0

    def test_power_analysis_lr_includes_development_period(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        pa = tracker.power_analysis(experiment.name)
        # Total = time to bind + development period
        assert pa["lr_total_months_with_development"] > pa["lr_months_to_bind"]

    def test_power_analysis_notes_populated(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        pa = tracker.power_analysis(experiment.name)
        assert len(pa["notes"]) > 0

    def test_power_analysis_larger_delta_needs_fewer(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        pa_small = tracker.power_analysis(experiment.name, target_delta_lr=0.01)
        pa_large = tracker.power_analysis(experiment.name, target_delta_lr=0.05)
        assert pa_small["lr_required_n_per_arm"] > pa_large["lr_required_n_per_arm"]

    def test_power_analysis_summary_report(self, populated_logger, experiment):
        tracker = KPITracker(populated_logger)
        report = tracker.summary_report(experiment.name)
        # Should return something iterable
        assert report is not None
