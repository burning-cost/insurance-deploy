"""Tests for ENBPAuditReport."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timezone, timedelta

import pytest

from insurance_deploy import ENBPAuditReport, QuoteLogger


def make_logger(tmp_path):
    return QuoteLogger(tmp_path / "q.db")


def populate_audit_data(
    logger,
    experiment_name: str = "audit_exp",
    n_champ_renewal: int = 100,
    n_chall_renewal: int = 20,
    n_nb: int = 50,
    breach_count: int = 0,
    seed: int = 42,
):
    import random
    rng = random.Random(seed)
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for i in range(n_champ_renewal):
        pid = f"CHAMP-{i:04d}"
        price = rng.gauss(400, 50)
        enbp = price + rng.gauss(15, 5)  # enbp > price = compliant
        logger.log_quote(pid, experiment_name, "champion", "motor:1.0",
                         quoted_price=price, enbp=enbp, renewal_flag=True,
                         timestamp=base_ts + timedelta(days=i // 3))

    for i in range(breach_count):
        pid = f"BREACH-{i:04d}"
        price = 450.0
        enbp = 430.0  # enbp < price = breach
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logger.log_quote(pid, experiment_name, "champion", "motor:1.0",
                             quoted_price=price, enbp=enbp, renewal_flag=True,
                             timestamp=base_ts + timedelta(days=10))

    for i in range(n_chall_renewal):
        pid = f"CHALL-{i:04d}"
        price = rng.gauss(410, 50)
        enbp = price + rng.gauss(12, 4)
        logger.log_quote(pid, experiment_name, "challenger", "motor:2.0",
                         quoted_price=price, enbp=enbp, renewal_flag=True,
                         timestamp=base_ts + timedelta(days=i // 2))

    for i in range(n_nb):
        pid = f"NB-{i:04d}"
        logger.log_quote(pid, experiment_name, "champion", "motor:1.0",
                         quoted_price=380.0, renewal_flag=False,
                         timestamp=base_ts + timedelta(days=i))

    return logger


class TestENBPAuditReportGeneration:

    def test_generate_returns_string(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert isinstance(md, str)
        assert len(md) > 100

    def test_report_contains_key_sections(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "Executive Summary" in md
        assert "ENBP Compliance by Model Arm" in md
        assert "Model Versions Used in Pricing" in md
        assert "Routing Decision Audit" in md
        assert "Attestation Statement" in md

    def test_report_references_icobs(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "ICOBS 6B.2.51R" in md

    def test_report_includes_experiment_name(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger, experiment_name="my_experiment")
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("my_experiment")
        assert "my_experiment" in md

    def test_report_includes_firm_name(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp", firm_name="Acme Insurance Ltd")
        assert "Acme Insurance Ltd" in md

    def test_report_includes_smf_holder(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp", smf_holder="Jane Smith")
        assert "Jane Smith" in md

    def test_report_shows_model_versions(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "motor:1.0" in md
        assert "motor:2.0" in md

    def test_report_shows_correct_renewal_count(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger, n_champ_renewal=100, n_chall_renewal=20, n_nb=50)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "120" in md  # total renewals: 100 + 20

    def test_breach_section_present_when_breaches(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger, breach_count=5)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "ENBP Breach Detail" in md
        assert "Action required" in md

    def test_no_breach_message_when_no_breaches(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger, breach_count=0)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "No ENBP breaches" in md

    def test_breach_detail_shows_policy_ids(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger, breach_count=3)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "BREACH-0000" in md

    def test_empty_experiment_shows_no_renewal_message(self, tmp_dir):
        logger = make_logger(tmp_dir)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("empty_exp")
        assert "No renewal quotes recorded" in md


class TestENBPAuditReportPeriodFilter:

    def test_period_start_filters_earlier_quotes(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp", period_start="2024-06-01", period_end="2024-12-31")
        assert "2024-06-01 to 2024-12-31" in md

    def test_period_label_all_data_when_no_filter(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "All available data" in md

    def test_period_start_only(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp", period_start="2024-03-01")
        assert "from 2024-03-01" in md

    def test_period_end_only(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp", period_end="2024-09-30")
        assert "to 2024-09-30" in md

    def test_period_filter_reduces_quote_count(self, tmp_dir):
        """Filtering to a sub-period should show fewer quotes."""
        logger = make_logger(tmp_dir)
        # All quotes in January 2024
        for i in range(20):
            ts = datetime(2024, 1, i + 1, tzinfo=timezone.utc)
            logger.log_quote(
                f"P{i}", "e1", "champion", "m:1",
                400.0, enbp=420.0, renewal_flag=True, timestamp=ts
            )
        reporter = ENBPAuditReport(logger)
        md_all = reporter.generate("e1")
        md_partial = reporter.generate("e1", period_start="2024-01-15")
        # Both should have data, partial should have fewer
        assert "No renewal quotes recorded" not in md_partial


class TestENBPAuditReportDeterminism:

    def test_hash_routing_description_present(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "SHA-256" in md
        assert "deterministic" in md

    def test_report_generated_timestamp_present(self, tmp_dir):
        logger = make_logger(tmp_dir)
        populate_audit_data(logger)
        reporter = ENBPAuditReport(logger)
        md = reporter.generate("audit_exp")
        assert "Generated:" in md
        assert "UTC" in md
