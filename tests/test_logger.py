"""Tests for QuoteLogger — append-only SQLite audit log."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timezone, timedelta

import pytest

from insurance_deploy import QuoteLogger


class TestQuoteLoggerSchema:

    def test_logger_creates_database(self, tmp_dir):
        path = tmp_dir / "test.db"
        QuoteLogger(path)
        assert path.exists()

    def test_empty_logger_has_zero_quotes(self, logger):
        assert logger.quote_count() == 0

    def test_logger_tables_exist(self, tmp_dir):
        import sqlite3
        path = tmp_dir / "test.db"
        QuoteLogger(path)
        conn = sqlite3.connect(path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "quotes" in tables
        assert "binds" in tables
        assert "claims" in tables
        conn.close()


class TestLogQuote:

    def test_log_quote_inserts_record(self, logger):
        logger.log_quote("POL-001", "exp_1", "champion", "motor:1.0", quoted_price=400.0)
        assert logger.quote_count() == 1

    def test_log_quote_fields(self, logger):
        ts = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)
        logger.log_quote(
            "POL-001", "exp_1", "champion", "motor:1.0",
            quoted_price=450.0, enbp=460.0, renewal_flag=True, exposure=1.0, timestamp=ts
        )
        q = logger.query_quotes()[0]
        assert q["policy_id"] == "POL-001"
        assert q["experiment_name"] == "exp_1"
        assert q["arm"] == "champion"
        assert q["model_version"] == "motor:1.0"
        assert q["quoted_price"] == pytest.approx(450.0)
        assert q["enbp"] == pytest.approx(460.0)
        assert q["renewal_flag"] == 1
        assert q["enbp_flag"] == 1  # 450 <= 460
        assert q["exposure"] == pytest.approx(1.0)

    def test_enbp_flag_compliant(self, logger):
        logger.log_quote("P1", "e1", "champion", "m:1", 400.0, enbp=420.0, renewal_flag=True)
        q = logger.query_quotes()[0]
        assert q["enbp_flag"] == 1

    def test_enbp_flag_breach_with_warning(self, logger):
        with pytest.warns(UserWarning, match="ENBP breach"):
            logger.log_quote("P1", "e1", "champion", "m:1", 430.0, enbp=420.0, renewal_flag=True)
        q = logger.query_quotes()[0]
        assert q["enbp_flag"] == 0

    def test_enbp_flag_null_for_nb(self, logger):
        logger.log_quote("P1", "e1", "champion", "m:1", 400.0, renewal_flag=False)
        q = logger.query_quotes()[0]
        assert q["enbp_flag"] is None

    def test_enbp_flag_null_when_enbp_not_provided_for_renewal(self, logger):
        with pytest.warns(UserWarning, match="no enbp provided"):
            logger.log_quote("P1", "e1", "champion", "m:1", 400.0, renewal_flag=True, enbp=None)
        q = logger.query_quotes()[0]
        assert q["enbp_flag"] is None

    def test_log_quote_invalid_arm_raises(self, logger):
        with pytest.raises(ValueError, match="arm"):
            logger.log_quote("P1", "e1", "neutral", "m:1", 400.0)

    def test_log_quote_negative_price_raises(self, logger):
        with pytest.raises(ValueError, match="quoted_price"):
            logger.log_quote("P1", "e1", "champion", "m:1", -100.0)

    def test_log_quote_zero_exposure_raises(self, logger):
        with pytest.raises(ValueError, match="exposure"):
            logger.log_quote("P1", "e1", "champion", "m:1", 400.0, exposure=0.0)

    def test_log_quote_defaults_timestamp_to_now(self, logger):
        before = datetime.now(timezone.utc)
        logger.log_quote("P1", "e1", "champion", "m:1", 400.0)
        after = datetime.now(timezone.utc)
        q = logger.query_quotes()[0]
        ts = datetime.fromisoformat(q["timestamp"])
        assert before <= ts <= after

    def test_log_multiple_quotes_append(self, logger):
        for i in range(10):
            logger.log_quote(f"POL-{i:03d}", "exp", "champion", "m:1", 400.0 + i)
        assert logger.quote_count() == 10

    def test_filter_by_experiment(self, logger):
        logger.log_quote("P1", "exp_a", "champion", "m:1", 400.0)
        logger.log_quote("P2", "exp_b", "champion", "m:1", 400.0)
        logger.log_quote("P3", "exp_a", "champion", "m:1", 400.0)
        assert logger.quote_count("exp_a") == 2
        assert logger.quote_count("exp_b") == 1


class TestLogBind:

    def test_log_bind_inserts_record(self, logger):
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        logger.log_bind("P1", 400.0)
        binds = logger.query_binds()
        assert len(binds) == 1

    def test_log_bind_fields(self, logger):
        ts = datetime(2024, 3, 1, tzinfo=timezone.utc)
        logger.log_bind("P1", 425.50, bound_timestamp=ts)
        b = logger.query_binds()[0]
        assert b["policy_id"] == "P1"
        assert b["bound_price"] == pytest.approx(425.50)

    def test_log_bind_negative_price_raises(self, logger):
        with pytest.raises(ValueError, match="bound_price"):
            logger.log_bind("P1", -50.0)

    def test_log_bind_zero_price_allowed(self, logger):
        """Zero premium (e.g. free cover add-on) should be allowed."""
        logger.log_bind("P1", 0.0)
        assert len(logger.query_binds()) == 1

    def test_log_multiple_binds(self, logger):
        for i in range(5):
            logger.log_bind(f"P-{i}", 300.0 + i)
        assert len(logger.query_binds()) == 5


class TestLogClaim:

    def test_log_claim_inserts_record(self, logger):
        logger.log_claim("P1", date(2024, 6, 1), 1200.0, development_month=3)
        claims = logger.query_claims()
        assert len(claims) == 1

    def test_log_claim_fields(self, logger):
        logger.log_claim("P1", date(2024, 6, 1), 1500.75, development_month=12)
        c = logger.query_claims()[0]
        assert c["policy_id"] == "P1"
        assert c["claim_date"] == "2024-06-01"
        assert c["claim_amount"] == pytest.approx(1500.75)
        assert c["development_month"] == 12

    def test_log_claim_negative_amount_raises(self, logger):
        with pytest.raises(ValueError, match="claim_amount"):
            logger.log_claim("P1", date(2024, 6, 1), -100.0, 3)

    def test_log_claim_negative_development_month_raises(self, logger):
        with pytest.raises(ValueError, match="development_month"):
            logger.log_claim("P1", date(2024, 6, 1), 1000.0, -1)

    def test_log_claim_zero_development_allowed(self, logger):
        """FNOL (month 0) should be allowed."""
        logger.log_claim("P1", date(2024, 6, 1), 0.0, 0)
        assert len(logger.query_claims()) == 1

    def test_log_claim_multiple_development_updates(self, logger):
        """Can log the same claim at multiple development stages."""
        logger.log_claim("P1", date(2024, 6, 1), 800.0, 3)
        logger.log_claim("P1", date(2024, 6, 1), 1200.0, 12)
        claims = logger.query_claims()
        assert len(claims) == 2

    def test_claims_ordered_by_date(self, logger):
        logger.log_claim("P2", date(2024, 8, 1), 500.0, 6)
        logger.log_claim("P1", date(2024, 3, 1), 1000.0, 6)
        claims = logger.query_claims()
        assert claims[0]["claim_date"] == "2024-03-01"
        assert claims[1]["claim_date"] == "2024-08-01"


class TestLoggerAppendOnly:

    def test_cannot_delete_quotes(self, tmp_dir):
        """QuoteLogger exposes no delete method."""
        logger = QuoteLogger(tmp_dir / "q.db")
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        assert not hasattr(logger, "delete_quote")
        assert not hasattr(logger, "delete")

    def test_quote_count_monotonically_increases(self, logger):
        counts = []
        for i in range(5):
            logger.log_quote(f"P-{i}", "exp", "champion", "m:1", 400.0)
            counts.append(logger.quote_count())
        assert counts == list(range(1, 6))
