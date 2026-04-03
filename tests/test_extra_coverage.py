"""
Additional tests expanding coverage for insurance-deploy.

Batch 23 of 34 library sweep — targeting gaps in:
- QuoteLogger: to_pandas, to_polars error, query_quotes without filter,
  query_binds, query_claims, zero-price bind, duplicate policy quotes
- ModelRegistry: predict proxy, repr, model file path, persistence round-trip
- Experiment: shadow_model in live mode, repr format, created_at auto-set
- KPITracker: severity method, summary_report list fallback, enbp_compliance
  challenger arm, frequency warn_immature=False, power analysis edge cases
- ENBPAuditReport: period label helpers, >100 breach truncation,
  no-renewal-quotes edge case
- ModelComparison: _loss_ratio helper, _conclude helper, frequency_test notes,
  bootstrap with immature data warning
- Internal helpers: _filter_by_period, _period_label, _loss_ratio, _conclude,
  _estimate_months, _two_proportion_sample_size, _two_sample_mean_sample_size

Performance note: the populated_logger fixture is very slow on ARM (16-18s
per test). Tests here use lightweight inline data where possible.
"""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pytest

from insurance_deploy import (
    ModelRegistry, ModelVersion, Experiment, QuoteLogger,
    KPITracker, ModelComparison, ComparisonResult, ENBPAuditReport,
)
from insurance_deploy.audit import _filter_by_period, _period_label
from insurance_deploy.comparison import _loss_ratio, _conclude
from insurance_deploy.kpi import (
    _estimate_months, _summarise_prices_by_arm,
    _two_proportion_sample_size, _two_sample_mean_sample_size,
)
from .conftest import DummyModel


# ---------------------------------------------------------------------------
# Helpers reused across tests
# ---------------------------------------------------------------------------

def make_logger(tmp_path: Path) -> QuoteLogger:
    return QuoteLogger(tmp_path / "q.db")


def _populate_minimal(logger, exp_name="exp", n_champ=30, n_chall=10, seed=42):
    """Populate logger with minimal data using no claims."""
    import random
    rng = random.Random(seed)
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n_champ):
        pid = f"C-{i:04d}"
        ts = base_ts + timedelta(days=i)
        price = max(100, rng.gauss(400, 30))
        logger.log_quote(pid, exp_name, "champion", "motor:1.0",
                         quoted_price=price, enbp=price + 15,
                         renewal_flag=True, timestamp=ts)
        if rng.random() < 0.4:
            logger.log_bind(pid, price, ts + timedelta(hours=1))
    for i in range(n_chall):
        pid = f"H-{i:04d}"
        ts = base_ts + timedelta(days=i * 2)
        price = max(100, rng.gauss(410, 30))
        logger.log_quote(pid, exp_name, "challenger", "motor:2.0",
                         quoted_price=price, enbp=price + 15,
                         renewal_flag=True, timestamp=ts)
        if rng.random() < 0.35:
            logger.log_bind(pid, price, ts + timedelta(hours=1))


def _populate_with_claims(logger, exp_name="exp", n_champ=60, n_chall=20, seed=99):
    """Populate logger with binds and claims."""
    import random
    rng = random.Random(seed)
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(n_champ):
        pid = f"CC-{i:04d}"
        ts = base_ts + timedelta(days=i)
        price = max(100, rng.gauss(400, 30))
        logger.log_quote(pid, exp_name, "champion", "motor:1.0",
                         quoted_price=price, enbp=price + 15,
                         renewal_flag=True, exposure=1.0, timestamp=ts)
        if rng.random() < 0.35:
            logger.log_bind(pid, price, ts + timedelta(hours=1))
            if rng.random() < 0.10:
                logger.log_claim(pid, date(2024, 6, 1),
                                 max(0, rng.gauss(1500, 400)), 12)
    for i in range(n_chall):
        pid = f"CH-{i:04d}"
        ts = base_ts + timedelta(days=i * 2)
        price = max(100, rng.gauss(410, 30))
        logger.log_quote(pid, exp_name, "challenger", "motor:2.0",
                         quoted_price=price, enbp=price + 15,
                         renewal_flag=True, exposure=1.0, timestamp=ts)
        if rng.random() < 0.30:
            logger.log_bind(pid, price, ts + timedelta(hours=1))
            if rng.random() < 0.10:
                logger.log_claim(pid, date(2024, 6, 1),
                                 max(0, rng.gauss(1400, 400)), 12)


# ---------------------------------------------------------------------------
# QuoteLogger — additional cases
# ---------------------------------------------------------------------------

class TestQuoteLoggerExtra:

    def test_query_quotes_without_filter_returns_all(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp_a", "champion", "m:1", 400.0)
        logger.log_quote("P2", "exp_b", "champion", "m:1", 410.0)
        all_quotes = logger.query_quotes()  # no filter
        assert len(all_quotes) == 2

    def test_query_binds_returns_all(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_bind("P1", 400.0)
        logger.log_bind("P2", 410.0)
        binds = logger.query_binds()
        assert len(binds) == 2

    def test_query_claims_empty(self, tmp_dir):
        logger = make_logger(tmp_dir)
        claims = logger.query_claims()
        assert claims == []

    def test_log_quote_positive_exposure(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0, exposure=0.5)
        q = logger.query_quotes()[0]
        assert q["exposure"] == pytest.approx(0.5)

    def test_log_quote_negative_exposure_raises(self, tmp_dir):
        logger = make_logger(tmp_dir)
        with pytest.raises(ValueError, match="exposure"):
            logger.log_quote("P1", "exp", "champion", "m:1", 400.0, exposure=-0.1)

    def test_log_quote_challenger_arm(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp", "challenger", "m:2", 420.0)
        q = logger.query_quotes()[0]
        assert q["arm"] == "challenger"

    def test_log_bind_zero_price_allowed(self, tmp_dir):
        """Zero premium is valid (free cover scenario)."""
        logger = make_logger(tmp_dir)
        logger.log_bind("P1", 0.0)
        b = logger.query_binds()[0]
        assert b["bound_price"] == pytest.approx(0.0)

    def test_quote_count_with_none_experiment(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i in range(3):
            logger.log_quote(f"P{i}", "exp", "champion", "m:1", 400.0)
        assert logger.quote_count() == 3
        assert logger.quote_count(experiment_name=None) == 3

    def test_to_pandas_invalid_table_raises(self, tmp_dir):
        """to_pandas with unknown table name should raise ValueError."""
        pd = pytest.importorskip("pandas")
        logger = make_logger(tmp_dir)
        with pytest.raises(ValueError, match="Unknown table"):
            logger.to_pandas(table="unknown_table")

    def test_to_pandas_binds_table(self, tmp_dir):
        """to_pandas should work with binds table."""
        pytest.importorskip("pandas")
        logger = make_logger(tmp_dir)
        logger.log_bind("P1", 400.0)
        df = logger.to_pandas(table="binds")
        assert len(df) == 1
        assert "bound_price" in df.columns

    def test_to_pandas_claims_table(self, tmp_dir):
        """to_pandas should work with claims table."""
        pytest.importorskip("pandas")
        logger = make_logger(tmp_dir)
        logger.log_claim("P1", date(2024, 6, 1), 1200.0, 12)
        df = logger.to_pandas(table="claims")
        assert len(df) == 1

    def test_to_polars_raises_without_polars(self, tmp_dir, monkeypatch):
        """to_polars should raise ImportError with helpful message when polars absent."""
        import sys
        logger = make_logger(tmp_dir)
        # Remove polars from sys.modules if present, mock the import to fail
        import builtins
        orig_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "polars":
                raise ImportError("No module named 'polars'")
            return orig_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="polars"):
            logger.to_polars()

    def test_to_polars_invalid_table_raises(self, tmp_dir):
        """to_polars with unknown table should raise ValueError."""
        pytest.importorskip("polars")
        logger = make_logger(tmp_dir)
        with pytest.raises(ValueError, match="Unknown table"):
            logger.to_polars(table="bad_table")

    def test_log_claim_zero_amount_allowed(self, tmp_dir):
        """Zero claim amount (e.g. nil claim) is valid."""
        logger = make_logger(tmp_dir)
        logger.log_claim("P1", date(2024, 6, 1), 0.0, 3)
        c = logger.query_claims()[0]
        assert c["claim_amount"] == pytest.approx(0.0)

    def test_persistence_after_reopen(self, tmp_dir):
        """Data should persist across QuoteLogger instances on the same file."""
        path = tmp_dir / "persistent.db"
        l1 = QuoteLogger(path)
        l1.log_quote("P1", "exp", "champion", "m:1", 400.0)
        l1.log_bind("P1", 400.0)

        l2 = QuoteLogger(path)
        assert l2.quote_count() == 1
        assert len(l2.query_binds()) == 1

    def test_renewal_without_enbp_warns(self, tmp_dir):
        logger = make_logger(tmp_dir)
        with pytest.warns(UserWarning, match="no enbp provided"):
            logger.log_quote("P1", "exp", "champion", "m:1", 400.0, renewal_flag=True, enbp=None)

    def test_enbp_exact_equality_is_compliant(self, tmp_dir):
        """quoted_price == enbp is compliant (<=)."""
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0, enbp=400.0, renewal_flag=True)
        q = logger.query_quotes()[0]
        assert q["enbp_flag"] == 1


# ---------------------------------------------------------------------------
# ModelRegistry — additional cases
# ---------------------------------------------------------------------------

class TestModelRegistryExtra:

    def test_model_version_repr(self, registry, champion_model):
        mv = registry.register(champion_model, "motor", "1.0")
        r = repr(mv)
        assert "motor:1.0" in r
        assert "registered=" in r

    def test_model_version_predict_proxy(self, registry, champion_model):
        """ModelVersion.predict() delegates to the underlying model."""
        import numpy as np
        mv = registry.register(champion_model, "motor", "1.0")
        result = mv.predict([[1, 2, 3]])
        assert result[0] == pytest.approx(400.0)

    def test_model_version_not_champion_by_default(self, registry, champion_model):
        mv = registry.register(champion_model, "motor", "1.0")
        assert mv.is_champion is False

    def test_set_champion_repr_contains_marker(self, registry):
        registry.register(DummyModel(), "motor", "1.0")
        registry.set_champion("motor", "1.0")
        mv = registry.champion("motor")
        assert "[champion]" in repr(mv)

    def test_register_model_path_is_absolute(self, registry, champion_model):
        mv = registry.register(champion_model, "motor", "1.0")
        assert Path(mv.model_path).is_absolute()

    def test_registry_list_empty_when_no_models(self, registry):
        versions = registry.list()
        assert versions == []

    def test_registry_list_filtered_empty(self, registry, champion_model):
        registry.register(champion_model, "motor", "1.0")
        versions = registry.list(name="home")
        assert versions == []

    def test_champion_after_set_champion_persists_on_reload(self, tmp_dir):
        path = tmp_dir / "reg"
        r1 = ModelRegistry(path)
        r1.register(DummyModel(), "motor", "1.0")
        r1.register(DummyModel(), "motor", "2.0")
        r1.set_champion("motor", "1.0")

        r2 = ModelRegistry(path)
        assert r2.champion("motor").version == "1.0"

    def test_register_home_and_motor_separately(self, registry):
        registry.register(DummyModel(), "motor", "1.0")
        registry.register(DummyModel(), "home", "1.0")
        assert len(registry.list("motor")) == 1
        assert len(registry.list("home")) == 1

    def test_model_hash_is_sha256(self, registry):
        mv = registry.register(DummyModel(), "motor", "1.0")
        # SHA-256 hex digest is 64 chars
        assert len(mv.model_hash) == 64
        assert all(c in "0123456789abcdef" for c in mv.model_hash)


# ---------------------------------------------------------------------------
# Experiment — additional cases
# ---------------------------------------------------------------------------

class TestExperimentExtra:

    def test_shadow_model_returns_challenger_for_all_policies(self, experiment):
        """Shadow mode: shadow_model must return challenger for all policies."""
        pids = [f"P-{i}" for i in range(200)]
        results = [experiment.shadow_model(pid).version_id for pid in pids]
        assert all(v == experiment.challenger.version_id for v in results)

    def test_shadow_model_live_mode_returns_other_arm(self, champion_mv, challenger_mv):
        """In live mode, shadow_model returns the non-live arm."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = Experiment(
                "live_shadow_test",
                champion_mv, challenger_mv,
                challenger_pct=0.5,
                mode="live",
            )
        pids = [f"P-{i}" for i in range(100)]
        for pid in pids:
            arm = exp.route(pid)
            shadow = exp.shadow_model(pid)
            if arm == "challenger":
                assert shadow.version_id == champion_mv.version_id
            else:
                assert shadow.version_id == challenger_mv.version_id

    def test_created_at_is_set_automatically(self, champion_mv, challenger_mv):
        exp = Experiment("auto_ts_test", champion_mv, challenger_mv)
        assert exp.created_at != ""
        assert "T" in exp.created_at  # ISO 8601

    def test_deactivated_at_none_before_deactivate(self, experiment):
        assert experiment.deactivated_at is None

    def test_live_model_deactivated_raises(self, experiment):
        """live_model on deactivated experiment should raise (calls route)."""
        experiment.deactivate()
        with pytest.raises(RuntimeError, match="deactivated"):
            experiment.live_model("POL-001")

    def test_routing_stability_with_numeric_policy_ids(self, experiment):
        """Numeric-looking policy IDs should route stably."""
        pid = "12345"
        arm1 = experiment.route(pid)
        arm2 = experiment.route(pid)
        assert arm1 == arm2

    def test_routing_stability_with_empty_string_policy_id(self, experiment):
        """Empty string policy ID should route stably (edge case)."""
        arm1 = experiment.route("")
        arm2 = experiment.route("")
        assert arm1 == arm2


# ---------------------------------------------------------------------------
# KPITracker — additional cases
# ---------------------------------------------------------------------------

class TestKPITrackerExtra:

    def test_severity_no_claims_returns_nan(self, tmp_dir):
        logger = make_logger(tmp_dir)
        _populate_minimal(logger, n_champ=10, n_chall=5)
        tracker = KPITracker(logger)
        sev = tracker.severity("exp", development_months=12)
        assert math.isnan(sev["champion"]["mean_severity"])
        assert sev["champion"]["claim_count"] == 0
        assert sev["champion"]["total_incurred"] == 0

    def test_severity_with_claims(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        logger.log_bind("P1", 400.0)
        logger.log_claim("P1", date(2024, 6, 1), 1500.0, 12)
        logger.log_claim("P1", date(2024, 7, 1), 800.0, 12)
        tracker = KPITracker(logger)
        sev = tracker.severity("exp", development_months=12)
        assert sev["champion"]["claim_count"] == 2
        assert sev["champion"]["mean_severity"] > 0

    def test_enbp_compliance_challenger_arm(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i in range(4):
            logger.log_quote(f"Q{i}", "exp", "challenger", "m:2",
                             400.0, enbp=420.0, renewal_flag=True)
        tracker = KPITracker(logger)
        comp = tracker.enbp_compliance("exp")
        assert comp["challenger"]["compliant"] == 4
        assert comp["challenger"]["compliance_rate"] == pytest.approx(1.0)

    def test_enbp_compliance_no_renewal_returns_nan(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0, renewal_flag=False)
        tracker = KPITracker(logger)
        comp = tracker.enbp_compliance("exp")
        assert math.isnan(comp["champion"]["compliance_rate"])
        assert comp["champion"]["renewal_quotes"] == 0

    def test_frequency_warn_immature_false(self, tmp_dir):
        """Setting warn_immature=False suppresses the IBNR warning."""
        logger = make_logger(tmp_dir)
        tracker = KPITracker(logger)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                tracker.frequency("exp", development_months=3, warn_immature=False)
            except UserWarning as e:
                pytest.fail(f"Unexpected warning with warn_immature=False: {e}")

    def test_gwp_empty_experiment_returns_nan(self, tmp_dir):
        logger = make_logger(tmp_dir)
        tracker = KPITracker(logger)
        gwp = tracker.gwp("empty_exp")
        assert math.isnan(gwp["champion"]["mean_gwp"])
        assert gwp["champion"]["bound_policies"] == 0

    def test_hit_rate_cohort_challenger_only(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i in range(5):
            logger.log_quote(f"CP-{i}", "exp", "champion", "m:1", 400.0)
        for i in range(3):
            logger.log_quote(f"CH-{i}", "exp", "challenger", "m:2", 410.0)
        tracker = KPITracker(logger)
        hr = tracker.hit_rate("exp", cohort="challenger")
        assert "champion" not in hr
        assert hr["challenger"]["quoted"] == 3

    def test_loss_ratio_exposure_weighted(self, tmp_dir):
        """Loss ratio uses exposure-weighted earned premium."""
        logger = make_logger(tmp_dir)
        # Policy with 0.5 year exposure: earned = 400 * 0.5 = 200
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0, exposure=0.5)
        logger.log_bind("P1", 400.0)
        logger.log_claim("P1", date(2024, 6, 1), 100.0, 12)
        tracker = KPITracker(logger)
        lr = tracker.loss_ratio("exp", development_months=12)
        # LR = 100 / (400 * 0.5) = 100/200 = 0.5
        assert lr["champion"]["loss_ratio"] == pytest.approx(0.5)

    def test_summary_report_has_both_arms(self, tmp_dir):
        logger = make_logger(tmp_dir)
        _populate_minimal(logger, n_champ=5, n_chall=3)
        tracker = KPITracker(logger)
        report = tracker.summary_report("exp")
        # Should be a list of dicts (no pandas dependency assumption)
        if isinstance(report, list):
            arms = {r["arm"] for r in report}
            assert "champion" in arms
            assert "challenger" in arms

    def test_power_analysis_returns_positive_sample_sizes(self, tmp_dir):
        logger = make_logger(tmp_dir)
        _populate_minimal(logger, n_champ=30, n_chall=10)
        tracker = KPITracker(logger)
        pa = tracker.power_analysis("exp")
        assert pa["hr_required_n_per_arm"] > 0
        assert pa["lr_required_bound_n_per_arm"] > 0

    def test_power_analysis_empty_experiment(self, tracker):
        pa = tracker.power_analysis("empty_exp")
        assert isinstance(pa, dict)
        assert "notes" in pa


# ---------------------------------------------------------------------------
# ENBPAuditReport — additional cases
# ---------------------------------------------------------------------------

class TestENBPAuditReportExtra:

    def test_generate_with_no_data_returns_no_renewal_message(self, tmp_dir):
        logger = make_logger(tmp_dir)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("no_data_exp")
        assert "No renewal quotes recorded" in md

    def test_generate_attestation_contains_smf_info(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i in range(5):
            logger.log_quote(f"P{i}", "e1", "champion", "m:1",
                             400.0, enbp=420.0, renewal_flag=True)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("e1", smf_holder="Alice Brown")
        assert "Alice Brown" in md
        assert "SMF" in md

    def test_generate_breach_truncates_at_100_rows(self, tmp_dir):
        """More than 100 breaches should be truncated in the report."""
        logger = make_logger(tmp_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(110):
                logger.log_quote(f"B{i:04d}", "e2", "champion", "m:1",
                                 450.0, enbp=430.0, renewal_flag=True)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("e2")
        assert "further breaches" in md

    def test_generate_compliance_status_at_risk(self, tmp_dir):
        """Report with breaches should show 'AT RISK' status."""
        logger = make_logger(tmp_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 1 compliant + 1 breach
            logger.log_quote("P1", "e3", "champion", "m:1",
                             400.0, enbp=420.0, renewal_flag=True)
            logger.log_quote("P2", "e3", "champion", "m:1",
                             450.0, enbp=430.0, renewal_flag=True)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("e3")
        assert "AT RISK" in md

    def test_generate_compliance_status_compliant(self, tmp_dir):
        """All compliant -> status should be COMPLIANT."""
        logger = make_logger(tmp_dir)
        for i in range(5):
            logger.log_quote(f"P{i}", "e4", "champion", "m:1",
                             400.0, enbp=420.0, renewal_flag=True)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("e4")
        assert "COMPLIANT" in md

    def test_generate_incomplete_records_status(self, tmp_dir):
        """Renewals without ENBP recorded -> RECORDS INCOMPLETE status."""
        logger = make_logger(tmp_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(5):
                logger.log_quote(f"P{i}", "e5", "champion", "m:1",
                                 400.0, enbp=None, renewal_flag=True)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("e5")
        assert "RECORDS INCOMPLETE" in md or "N/A" in md

    def test_generate_period_start_only_label(self, tmp_dir):
        logger = make_logger(tmp_dir)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("e6", period_start="2024-03-01")
        assert "from 2024-03-01" in md

    def test_generate_period_end_only_label(self, tmp_dir):
        logger = make_logger(tmp_dir)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("e7", period_end="2024-09-30")
        assert "to 2024-09-30" in md

    def test_generate_contains_routing_audit_section(self, tmp_dir):
        logger = make_logger(tmp_dir)
        for i in range(3):
            logger.log_quote(f"P{i}", "e8", "champion", "m:1",
                             400.0, enbp=420.0, renewal_flag=True)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("e8")
        assert "Routing Decision Audit" in md

    def test_generate_model_versions_section(self, tmp_dir):
        logger = make_logger(tmp_dir)
        logger.log_quote("P1", "e9", "champion", "m:1.0",
                         400.0, enbp=420.0, renewal_flag=True)
        logger.log_quote("P2", "e9", "challenger", "m:2.0",
                         410.0, enbp=430.0, renewal_flag=True)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("e9")
        assert "m:1.0" in md
        assert "m:2.0" in md


# ---------------------------------------------------------------------------
# _filter_by_period and _period_label helpers
# ---------------------------------------------------------------------------

class TestAuditHelpers:

    def test_filter_by_period_no_filter_returns_all(self):
        quotes = [
            {"timestamp": "2024-01-15T10:00:00"},
            {"timestamp": "2024-06-01T10:00:00"},
        ]
        result = _filter_by_period(quotes, None, None)
        assert len(result) == 2

    def test_filter_by_period_start_only(self):
        quotes = [
            {"timestamp": "2024-01-15T10:00:00"},
            {"timestamp": "2024-06-01T10:00:00"},
            {"timestamp": "2024-09-01T10:00:00"},
        ]
        result = _filter_by_period(quotes, "2024-06-01", None)
        assert len(result) == 2
        assert result[0]["timestamp"].startswith("2024-06")

    def test_filter_by_period_end_only(self):
        quotes = [
            {"timestamp": "2024-01-15T10:00:00"},
            {"timestamp": "2024-06-01T10:00:00"},
            {"timestamp": "2024-09-01T10:00:00"},
        ]
        result = _filter_by_period(quotes, None, "2024-05-31")
        assert len(result) == 1

    def test_filter_by_period_both_bounds(self):
        quotes = [
            {"timestamp": "2024-01-15T10:00:00"},
            {"timestamp": "2024-06-01T10:00:00"},
            {"timestamp": "2024-09-01T10:00:00"},
        ]
        result = _filter_by_period(quotes, "2024-05-01", "2024-07-01")
        assert len(result) == 1

    def test_filter_by_period_empty_quotes(self):
        result = _filter_by_period([], "2024-01-01", "2024-12-31")
        assert result == []

    def test_period_label_both(self):
        assert _period_label("2024-01-01", "2024-12-31") == "2024-01-01 to 2024-12-31"

    def test_period_label_start_only(self):
        assert _period_label("2024-06-01", None) == "from 2024-06-01"

    def test_period_label_end_only(self):
        assert _period_label(None, "2024-12-31") == "to 2024-12-31"

    def test_period_label_neither(self):
        assert _period_label(None, None) == "All available data"


# ---------------------------------------------------------------------------
# ModelComparison helpers
# ---------------------------------------------------------------------------

class TestLossRatioHelper:

    def test_empty_data_returns_nan(self):
        result = _loss_ratio([])
        assert math.isnan(result)

    def test_single_policy(self):
        # (premium, exposure, incurred)
        data = [(400.0, 1.0, 200.0)]
        assert _loss_ratio(data) == pytest.approx(0.5)

    def test_zero_premium_returns_nan(self):
        data = [(0.0, 1.0, 200.0)]
        result = _loss_ratio(data)
        assert math.isnan(result)

    def test_multiple_policies(self):
        # total incurred = 200 + 400 = 600; total earned = 400*1 + 500*1 = 900
        data = [(400.0, 1.0, 200.0), (500.0, 1.0, 400.0)]
        assert _loss_ratio(data) == pytest.approx(600 / 900)

    def test_zero_incurred_gives_zero_lr(self):
        data = [(400.0, 1.0, 0.0), (500.0, 1.0, 0.0)]
        assert _loss_ratio(data) == pytest.approx(0.0)

    def test_exposure_weighted(self):
        # premium 400 * exposure 0.5 = 200 earned; incurred 100 -> LR=0.5
        data = [(400.0, 0.5, 100.0)]
        assert _loss_ratio(data) == pytest.approx(0.5)


class TestConcludeHelper:

    def test_insufficient_evidence_small_sample(self):
        conclusion, _ = _conclude(
            diff=-0.05, ci_lower=-0.1, ci_upper=0.0,
            p_value=0.01, n1=10, n2=10,
            metric="loss_ratio", lower_is_better=True,
        )
        assert conclusion == "INSUFFICIENT_EVIDENCE"

    def test_insufficient_evidence_high_p_value(self):
        conclusion, _ = _conclude(
            diff=-0.05, ci_lower=-0.1, ci_upper=0.0,
            p_value=0.3, n1=200, n2=100,
            metric="loss_ratio", lower_is_better=True,
        )
        assert conclusion == "INSUFFICIENT_EVIDENCE"

    def test_challenger_better_lower_is_better(self):
        conclusion, rec = _conclude(
            diff=-0.05, ci_lower=-0.1, ci_upper=-0.01,
            p_value=0.01, n1=200, n2=100,
            metric="loss_ratio", lower_is_better=True,
        )
        assert conclusion == "CHALLENGER_BETTER"
        assert "challenger" in rec.lower()

    def test_champion_better_lower_is_better(self):
        """Challenger has higher loss ratio than champion -> champion wins."""
        conclusion, rec = _conclude(
            diff=0.05, ci_lower=0.01, ci_upper=0.09,
            p_value=0.01, n1=200, n2=100,
            metric="loss_ratio", lower_is_better=True,
        )
        assert conclusion == "CHAMPION_BETTER"

    def test_challenger_better_higher_is_better(self):
        """For hit_rate, higher is better, so positive diff means challenger wins."""
        conclusion, rec = _conclude(
            diff=0.05, ci_lower=0.01, ci_upper=0.09,
            p_value=0.01, n1=200, n2=100,
            metric="hit_rate", lower_is_better=False,
        )
        assert conclusion == "CHALLENGER_BETTER"

    def test_nan_p_value_gives_insufficient(self):
        conclusion, _ = _conclude(
            diff=float("nan"), ci_lower=float("nan"), ci_upper=float("nan"),
            p_value=float("nan"), n1=200, n2=100,
            metric="loss_ratio", lower_is_better=True,
        )
        assert conclusion == "INSUFFICIENT_EVIDENCE"


class TestEstimateMonths:

    def test_empty_quotes_returns_zero(self):
        result = _estimate_months([])
        assert result == 0.0

    def test_single_quote_returns_zero(self):
        result = _estimate_months([{"timestamp": "2024-01-01T00:00:00"}])
        assert result == 0.0

    def test_two_quotes_thirty_days_apart(self):
        q = [
            {"timestamp": "2024-01-01T00:00:00"},
            {"timestamp": "2024-01-31T00:00:00"},
        ]
        months = _estimate_months(q)
        # 30 days / 30.44 ~= 0.986 months
        assert 0.9 < months < 1.1

    def test_one_year_of_quotes(self):
        q = [
            {"timestamp": "2024-01-01T00:00:00"},
            {"timestamp": "2025-01-01T00:00:00"},
        ]
        months = _estimate_months(q)
        assert 11.5 < months < 12.5


class TestSampleSizeHelpers:

    def test_two_proportion_returns_positive_int(self):
        n = _two_proportion_sample_size(p=0.3, delta=0.05, alpha=0.05, power=0.80)
        assert isinstance(n, int)
        assert n > 0

    def test_larger_delta_needs_smaller_n(self):
        n_small_delta = _two_proportion_sample_size(0.3, 0.02, 0.05, 0.80)
        n_large_delta = _two_proportion_sample_size(0.3, 0.10, 0.05, 0.80)
        assert n_small_delta > n_large_delta

    def test_two_sample_mean_is_double_one_sample(self):
        from insurance_deploy.kpi import _one_sample_mean_sample_size
        n1 = _one_sample_mean_sample_size(sigma=0.3, delta=0.05, alpha=0.05, power=0.80)
        n2 = _two_sample_mean_sample_size(sigma=0.3, delta=0.05, alpha=0.05, power=0.80)
        assert n2 == 2 * n1

    def test_higher_power_needs_larger_n(self):
        n_80 = _two_proportion_sample_size(0.3, 0.05, 0.05, 0.80)
        n_90 = _two_proportion_sample_size(0.3, 0.05, 0.05, 0.90)
        assert n_90 > n_80


# ---------------------------------------------------------------------------
# ModelComparison — additional tests using lightweight data
# ---------------------------------------------------------------------------

class TestModelComparisonExtra:

    def test_bootstrap_lr_test_with_claims_data(self, tmp_dir):
        logger = make_logger(tmp_dir)
        _populate_with_claims(logger, n_champ=60, n_chall=20)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.bootstrap_lr_test("exp", n_bootstrap=100, seed=1)
        assert isinstance(result, ComparisonResult)
        assert result.n_champion > 0 or result.conclusion == "INSUFFICIENT_EVIDENCE"

    def test_hit_rate_test_small_sample_has_note(self, tmp_dir):
        """Small sample sizes should add a note to the result."""
        logger = make_logger(tmp_dir)
        _populate_minimal(logger, n_champ=10, n_chall=5)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.hit_rate_test("exp")
        # With n < DEFAULT_MIN_POLICIES (50), should add a note
        if result.n_champion < 50 or result.n_challenger < 50:
            assert len(result.notes) > 0

    def test_frequency_test_maturity_note_when_development_lt_12(self, tmp_dir):
        """Frequency test at <12 months development should include maturity note."""
        logger = make_logger(tmp_dir)
        _populate_with_claims(logger, n_champ=60, n_chall=20)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = comp.frequency_test("exp", development_months=6)
        # If claims were found, there should be a maturity note
        if result.n_champion > 0 and result.n_challenger > 0:
            assert any("IBNR" in note or "maturity" in note.lower()
                       for note in result.notes)

    def test_comparison_result_summary_mentions_ci(self, tmp_dir):
        logger = make_logger(tmp_dir)
        _populate_with_claims(logger, n_champ=60, n_chall=20)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.bootstrap_lr_test("exp", n_bootstrap=100, seed=2)
        summary = result.summary()
        assert "95% CI" in summary

    def test_comparison_result_summary_warns_maturity_when_flag_set(self, tmp_dir):
        logger = make_logger(tmp_dir)
        _populate_with_claims(logger, n_champ=60, n_chall=20)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = comp.bootstrap_lr_test("exp", n_bootstrap=100,
                                            development_months=6, seed=3)
        if result.maturity_warning:
            summary = result.summary()
            assert "WARNING" in summary or "maturity" in summary.lower()

    def test_bootstrap_maturity_warning_flag_at_6_months(self, tmp_dir):
        logger = make_logger(tmp_dir)
        _populate_with_claims(logger, n_champ=60, n_chall=20)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        with pytest.warns(UserWarning, match="development_months"):
            result = comp.bootstrap_lr_test("exp", n_bootstrap=50,
                                            development_months=6, seed=4)
        assert result.maturity_warning is True

    def test_comparison_result_repr_contains_test_name(self, tmp_dir):
        logger = make_logger(tmp_dir)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.hit_rate_test("empty_exp")
        assert "hit_rate_test" in repr(result)

    def test_adverse_selection_warning_not_set_by_default(self, tmp_dir):
        logger = make_logger(tmp_dir)
        tracker = KPITracker(logger)
        comp = ModelComparison(tracker)
        result = comp.hit_rate_test("empty_exp")
        assert result.adverse_selection_warning is False

    def test_comparison_result_notes_none_becomes_empty_list(self):
        """Notes=None in constructor should become empty list."""
        r = ComparisonResult(
            test_name="test", experiment_name="exp",
            champion_estimate=0.5, challenger_estimate=0.5,
            difference=0.0, ci_lower=-0.1, ci_upper=0.1,
            p_value=0.5, n_champion=100, n_challenger=50,
            conclusion="INSUFFICIENT_EVIDENCE", recommendation="x",
            notes=None,
        )
        assert r.notes == []


# ---------------------------------------------------------------------------
# _summarise_prices_by_arm helper
# ---------------------------------------------------------------------------

class TestSummarisePricesByArm:

    def test_empty_returns_nan(self):
        result = _summarise_prices_by_arm([])
        assert result["champion"]["n"] == 0
        import math
        assert math.isnan(result["champion"]["mean_price"])

    def test_single_quote(self):
        quotes = [{"arm": "champion", "quoted_price": 400.0}]
        result = _summarise_prices_by_arm(quotes)
        assert result["champion"]["n"] == 1
        assert result["champion"]["mean_price"] == pytest.approx(400.0)
        assert result["champion"]["median_price"] == pytest.approx(400.0)

    def test_percentiles_champion_and_challenger(self):
        quotes = (
            [{"arm": "champion", "quoted_price": float(i)} for i in range(1, 11)]
            + [{"arm": "challenger", "quoted_price": float(i)} for i in range(11, 16)]
        )
        result = _summarise_prices_by_arm(quotes)
        assert result["champion"]["n"] == 10
        assert result["challenger"]["n"] == 5
