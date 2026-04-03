"""
Extended tests for QuoteLogger.

Gaps covered:
- log_quote validation (invalid arm, negative price, zero exposure)
- log_bind validation (negative price)
- log_claim validation (negative amount, negative development_month)
- to_pandas / to_polars correctness and error paths
- query_quotes with/without experiment_name filter
- quote_count with/without experiment filter
- Multiple experiments are isolated via query_quotes filter
- Timestamps are stored in ISO format and queryable
- Renewal without ENBP triggers warning
- ENBP breach triggers warning
- enbp_flag logic (1 = compliant, 0 = breach, None = no enbp)
- Append-only: re-opening same path preserves data
- to_polars/to_pandas unknown table raises ValueError
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pytest

from insurance_deploy import QuoteLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logger(tmp_path):
    return QuoteLogger(tmp_path / "q.db")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestLogQuoteValidation:

    def test_invalid_arm_raises(self, logger):
        with pytest.raises(ValueError, match="arm"):
            logger.log_quote("P1", "exp", "random_arm", "m:1", 400.0)

    def test_negative_quoted_price_raises(self, logger):
        with pytest.raises(ValueError, match="quoted_price"):
            logger.log_quote("P1", "exp", "champion", "m:1", -1.0)

    def test_zero_quoted_price_is_ok(self, logger):
        """Price of 0 is edge-case but should not raise."""
        logger.log_quote("P1", "exp", "champion", "m:1", 0.0)
        assert logger.quote_count("exp") == 1

    def test_zero_exposure_raises(self, logger):
        with pytest.raises(ValueError, match="exposure"):
            logger.log_quote("P1", "exp", "champion", "m:1", 400.0, exposure=0.0)

    def test_negative_exposure_raises(self, logger):
        with pytest.raises(ValueError, match="exposure"):
            logger.log_quote("P1", "exp", "champion", "m:1", 400.0, exposure=-0.5)

    def test_both_arms_accepted(self, logger):
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        logger.log_quote("P2", "exp", "challenger", "m:2", 420.0)
        assert logger.quote_count("exp") == 2

    def test_custom_timestamp_stored(self, logger):
        ts = datetime(2024, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0, timestamp=ts)
        quotes = logger.query_quotes("exp")
        assert "2024-03-15" in quotes[0]["timestamp"]


class TestLogBindValidation:

    def test_negative_bound_price_raises(self, logger):
        with pytest.raises(ValueError, match="bound_price"):
            logger.log_bind("P1", bound_price=-100.0)

    def test_zero_bound_price_ok(self, logger):
        logger.log_bind("P1", bound_price=0.0)
        assert len(logger.query_binds()) == 1

    def test_custom_timestamp_stored(self, logger):
        ts = datetime(2024, 5, 1, tzinfo=timezone.utc)
        logger.log_bind("P1", 400.0, bound_timestamp=ts)
        binds = logger.query_binds()
        assert "2024-05-01" in binds[0]["bound_timestamp"]


class TestLogClaimValidation:

    def test_negative_claim_amount_raises(self, logger):
        with pytest.raises(ValueError, match="claim_amount"):
            logger.log_claim("P1", date(2024, 6, 1), -500.0, development_month=12)

    def test_zero_claim_amount_ok(self, logger):
        """Zero incurred (NIL claim) should not raise."""
        logger.log_claim("P1", date(2024, 6, 1), 0.0, development_month=12)
        assert len(logger.query_claims()) == 1

    def test_negative_development_month_raises(self, logger):
        with pytest.raises(ValueError, match="development_month"):
            logger.log_claim("P1", date(2024, 6, 1), 500.0, development_month=-1)

    def test_zero_development_month_ok(self, logger):
        """FNOL = development_month=0 should be accepted."""
        logger.log_claim("P1", date(2024, 6, 1), 800.0, development_month=0)
        assert len(logger.query_claims()) == 1


# ---------------------------------------------------------------------------
# ENBP flag logic tests
# ---------------------------------------------------------------------------

class TestENBPFlagLogic:

    def test_compliant_renewal_flag_1(self, logger):
        """quoted_price <= enbp → enbp_flag = 1."""
        logger.log_quote("P1", "exp", "champion", "m:1",
                         quoted_price=400.0, enbp=450.0, renewal_flag=True)
        quotes = logger.query_quotes("exp")
        assert quotes[0]["enbp_flag"] == 1

    def test_breach_renewal_flag_0(self, logger):
        """quoted_price > enbp → enbp_flag = 0 and warning."""
        with pytest.warns(UserWarning, match="ENBP breach"):
            logger.log_quote("P1", "exp", "champion", "m:1",
                             quoted_price=460.0, enbp=450.0, renewal_flag=True)
        quotes = logger.query_quotes("exp")
        assert quotes[0]["enbp_flag"] == 0

    def test_exact_enbp_equal_price_compliant(self, logger):
        """quoted_price == enbp should be compliant."""
        logger.log_quote("P1", "exp", "champion", "m:1",
                         quoted_price=400.0, enbp=400.0, renewal_flag=True)
        quotes = logger.query_quotes("exp")
        assert quotes[0]["enbp_flag"] == 1

    def test_renewal_without_enbp_warns_and_flag_none(self, logger):
        """Renewal with no enbp arg → warning, enbp_flag = None."""
        with pytest.warns(UserWarning, match="no enbp"):
            logger.log_quote("P1", "exp", "champion", "m:1",
                             quoted_price=400.0, renewal_flag=True)
        quotes = logger.query_quotes("exp")
        assert quotes[0]["enbp_flag"] is None

    def test_nb_quote_no_enbp_no_warning(self, logger):
        """New business quote without enbp should NOT warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            logger.log_quote("P1", "exp", "champion", "m:1",
                             quoted_price=400.0, renewal_flag=False)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_nb_quote_enbp_flag_none(self, logger):
        """New business should always have enbp_flag = None."""
        logger.log_quote("P1", "exp", "champion", "m:1",
                         quoted_price=400.0, enbp=420.0, renewal_flag=False)
        quotes = logger.query_quotes("exp")
        # renewal_flag=False so enbp_flag should be NULL regardless of enbp provided
        # Actually: looking at source, enbp_flag is set only if renewal_flag is True
        assert quotes[0]["enbp_flag"] is None


# ---------------------------------------------------------------------------
# Query and filter tests
# ---------------------------------------------------------------------------

class TestQueryFilters:

    def test_query_quotes_all_returns_all(self, logger):
        logger.log_quote("P1", "exp_a", "champion", "m:1", 400.0)
        logger.log_quote("P2", "exp_b", "champion", "m:1", 410.0)
        all_quotes = logger.query_quotes()  # no filter
        assert len(all_quotes) == 2

    def test_query_quotes_filtered_by_experiment(self, logger):
        logger.log_quote("P1", "exp_a", "champion", "m:1", 400.0)
        logger.log_quote("P2", "exp_b", "champion", "m:1", 410.0)
        logger.log_quote("P3", "exp_a", "challenger", "m:2", 420.0)
        quotes_a = logger.query_quotes("exp_a")
        assert len(quotes_a) == 2
        assert all(q["experiment_name"] == "exp_a" for q in quotes_a)

    def test_quote_count_all(self, logger):
        for i in range(5):
            logger.log_quote(f"P{i}", "exp_a", "champion", "m:1", 400.0)
        for i in range(3):
            logger.log_quote(f"Q{i}", "exp_b", "challenger", "m:2", 420.0)
        assert logger.quote_count() == 8

    def test_quote_count_by_experiment(self, logger):
        for i in range(5):
            logger.log_quote(f"P{i}", "exp_a", "champion", "m:1", 400.0)
        for i in range(3):
            logger.log_quote(f"Q{i}", "exp_b", "challenger", "m:2", 420.0)
        assert logger.quote_count("exp_a") == 5
        assert logger.quote_count("exp_b") == 3

    def test_query_binds_returns_all(self, logger):
        logger.log_bind("P1", 400.0)
        logger.log_bind("P2", 420.0)
        binds = logger.query_binds()
        assert len(binds) == 2

    def test_query_claims_returns_all(self, logger):
        logger.log_claim("P1", date(2024, 1, 1), 500.0, 12)
        logger.log_claim("P2", date(2024, 2, 1), 800.0, 12)
        claims = logger.query_claims()
        assert len(claims) == 2

    def test_query_returns_dicts(self, logger):
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        quotes = logger.query_quotes("exp")
        assert isinstance(quotes[0], dict)
        assert "policy_id" in quotes[0]
        assert "quoted_price" in quotes[0]


# ---------------------------------------------------------------------------
# Persistence (append-only)
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_reopen_preserves_data(self, tmp_path):
        path = tmp_path / "q.db"
        logger1 = QuoteLogger(path)
        logger1.log_quote("P1", "exp", "champion", "m:1", 400.0)
        logger1.log_bind("P1", 400.0)
        logger1.log_claim("P1", date(2024, 6, 1), 500.0, 12)

        # Re-open
        logger2 = QuoteLogger(path)
        assert logger2.quote_count() == 1
        assert len(logger2.query_binds()) == 1
        assert len(logger2.query_claims()) == 1

    def test_multiple_appends_accumulate(self, tmp_path):
        path = tmp_path / "q.db"
        for batch in range(3):
            lg = QuoteLogger(path)
            lg.log_quote(f"P{batch}", "exp", "champion", "m:1", 400.0)

        final = QuoteLogger(path)
        assert final.quote_count() == 3


# ---------------------------------------------------------------------------
# to_pandas tests
# ---------------------------------------------------------------------------

class TestToPandas:

    def test_to_pandas_quotes(self, logger):
        pytest.importorskip("pandas")
        import pandas as pd
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        df = logger.to_pandas("quotes")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "policy_id" in df.columns

    def test_to_pandas_binds(self, logger):
        pytest.importorskip("pandas")
        import pandas as pd
        logger.log_bind("P1", 400.0)
        df = logger.to_pandas("binds")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_to_pandas_claims(self, logger):
        pytest.importorskip("pandas")
        import pandas as pd
        logger.log_claim("P1", date(2024, 1, 1), 500.0, 12)
        df = logger.to_pandas("claims")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_to_pandas_unknown_table_raises(self, logger):
        with pytest.raises(ValueError, match="Unknown table"):
            logger.to_pandas("unknown_table")

    def test_to_pandas_empty_returns_empty_df(self, logger):
        pytest.importorskip("pandas")
        df = logger.to_pandas("quotes")
        assert len(df) == 0


# ---------------------------------------------------------------------------
# to_polars tests
# ---------------------------------------------------------------------------

class TestToPolars:

    def test_to_polars_quotes(self, logger):
        pl = pytest.importorskip("polars")
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0)
        df = logger.to_polars("quotes")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 1

    def test_to_polars_unknown_table_raises(self, logger):
        pytest.importorskip("polars")
        with pytest.raises(ValueError, match="Unknown table"):
            logger.to_polars("unknown_table")

    def test_to_polars_empty_returns_empty(self, logger):
        pl = pytest.importorskip("polars")
        df = logger.to_polars("quotes")
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Data fidelity: stored values match what was logged
# ---------------------------------------------------------------------------

class TestDataFidelity:

    def test_quoted_price_stored_correctly(self, logger):
        logger.log_quote("P1", "exp", "champion", "m:1", 387.50)
        quotes = logger.query_quotes("exp")
        assert quotes[0]["quoted_price"] == pytest.approx(387.50)

    def test_exposure_stored_correctly(self, logger):
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0, exposure=0.75)
        quotes = logger.query_quotes("exp")
        assert quotes[0]["exposure"] == pytest.approx(0.75)

    def test_enbp_stored_correctly(self, logger):
        logger.log_quote("P1", "exp", "champion", "m:1", 400.0, enbp=425.0, renewal_flag=True)
        quotes = logger.query_quotes("exp")
        assert quotes[0]["enbp"] == pytest.approx(425.0)

    def test_arm_stored_correctly(self, logger):
        logger.log_quote("P1", "exp", "challenger", "m:2", 420.0)
        quotes = logger.query_quotes("exp")
        assert quotes[0]["arm"] == "challenger"

    def test_model_version_stored_correctly(self, logger):
        logger.log_quote("P1", "exp", "champion", "motor:3.0-rc1", 400.0)
        quotes = logger.query_quotes("exp")
        assert quotes[0]["model_version"] == "motor:3.0-rc1"

    def test_claim_development_month_stored(self, logger):
        logger.log_claim("P1", date(2024, 6, 1), 1200.0, development_month=24)
        claims = logger.query_claims()
        assert claims[0]["development_month"] == 24

    def test_bound_price_stored_correctly(self, logger):
        logger.log_bind("P1", bound_price=395.75)
        binds = logger.query_binds()
        assert binds[0]["bound_price"] == pytest.approx(395.75)
