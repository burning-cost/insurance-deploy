"""
QuoteLogger — append-only SQLite audit log for pricing decisions.

Schema
------
Three tables:

  quotes — one row per quote event
    policy_id, experiment_name, arm (champion/challenger), model_version,
    quoted_price, enbp (nullable), renewal_flag, enbp_flag (bool),
    exposure (years), timestamp

  binds — one row per bind event
    policy_id, bound_price, bound_timestamp

  claims — one row per claim development entry
    policy_id, claim_date, claim_amount, development_month, logged_at

Design notes
------------
SQLite handles 1M–10M rows without issue. For enterprise scale (>10M rows/year)
the adapter pattern documented in README is appropriate — swap QuoteLogger for
a PostgreSQL-backed subclass. The core library stays SQLite-only.

ENBP recording: the library records the ENBP value you provide. It does not
calculate it. ICOBS 6B calculation is your pricing team's responsibility. The
library enforces the constraint (enbp_flag = quoted_price <= enbp) and provides
the audit trail; correctness of the ENBP number is upstream of this library.

The enbp_flag is stored as an integer (0/1) in SQLite. NULL when no ENBP was
provided (non-renewal quote, or renewal where ENBP was not calculated).
"""

from __future__ import annotations

import sqlite3
import warnings
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Generator, Optional


_CREATE_QUOTES = """
CREATE TABLE IF NOT EXISTS quotes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id       TEXT    NOT NULL,
    experiment_name TEXT    NOT NULL,
    arm             TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    quoted_price    REAL    NOT NULL,
    enbp            REAL,
    renewal_flag    INTEGER NOT NULL DEFAULT 0,
    enbp_flag       INTEGER,           -- 1=compliant, 0=breach, NULL=not renewal
    exposure        REAL    NOT NULL DEFAULT 1.0,
    timestamp       TEXT    NOT NULL
)
"""

_CREATE_BINDS = """
CREATE TABLE IF NOT EXISTS binds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id       TEXT    NOT NULL,
    bound_price     REAL    NOT NULL,
    bound_timestamp TEXT    NOT NULL
)
"""

_CREATE_CLAIMS = """
CREATE TABLE IF NOT EXISTS claims (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id         TEXT    NOT NULL,
    claim_date        TEXT    NOT NULL,
    claim_amount      REAL    NOT NULL,
    development_month INTEGER NOT NULL,
    logged_at         TEXT    NOT NULL
)
"""


class QuoteLogger:
    """
    Append-only SQLite audit log.

    Parameters
    ----------
    path : str or Path
        SQLite database file path. Created if absent.

    Examples
    --------
    >>> logger = QuoteLogger("./quotes.db")
    >>> logger.log_quote(
    ...     policy_id="POL-001",
    ...     experiment_name="v3_vs_v2",
    ...     arm="champion",
    ...     model_version="motor:2.0",
    ...     quoted_price=425.00,
    ...     enbp=430.00,
    ...     renewal_flag=True,
    ... )
    >>> logger.log_bind("POL-001", bound_price=425.00)
    >>> logger.log_claim("POL-001", claim_date=date(2024, 8, 1), claim_amount=1200.0, development_month=3)
    """

    def __init__(self, path: str | Path = "./quote_log.db") -> None:
        self.path = Path(path).resolve()
        self._initialise_schema()

    # ------------------------------------------------------------------
    # Context manager for connections
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialise_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_QUOTES)
            conn.execute(_CREATE_BINDS)
            conn.execute(_CREATE_CLAIMS)

    # ------------------------------------------------------------------
    # Write methods (append-only)
    # ------------------------------------------------------------------

    def log_quote(
        self,
        policy_id: str,
        experiment_name: str,
        arm: str,
        model_version: str,
        quoted_price: float,
        enbp: Optional[float] = None,
        renewal_flag: bool = False,
        exposure: float = 1.0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Log a quote event.

        Parameters
        ----------
        policy_id : str
            Unique policy identifier.
        experiment_name : str
            Name of the active experiment.
        arm : str
            ``'champion'`` or ``'challenger'``.
        model_version : str
            Version ID of the model that produced this quote (e.g. ``"motor:2.0"``).
        quoted_price : float
            The price shown to the customer.
        enbp : float, optional
            Equivalent New Business Price (ICOBS 6B). Provide for renewal quotes.
            The library records this value; it does NOT calculate it.
        renewal_flag : bool
            True if this is a renewal quote (not new business).
        exposure : float
            Policy exposure in years. Default 1.0.
        timestamp : datetime, optional
            Quote timestamp. Defaults to UTC now.
        """
        if arm not in ("champion", "challenger"):
            raise ValueError(f"arm must be 'champion' or 'challenger', got {arm!r}.")
        if quoted_price < 0:
            raise ValueError(f"quoted_price must be non-negative, got {quoted_price}.")
        if exposure <= 0:
            raise ValueError(f"exposure must be positive, got {exposure}.")

        ts = (timestamp or datetime.now(timezone.utc)).isoformat()

        enbp_flag: Optional[int] = None
        if renewal_flag:
            if enbp is None:
                warnings.warn(
                    f"log_quote called with renewal_flag=True but no enbp provided "
                    f"for policy_id={policy_id!r}. ENBP compliance cannot be audited "
                    "for this quote.",
                    stacklevel=2,
                )
            else:
                enbp_flag = 1 if quoted_price <= enbp else 0
                if enbp_flag == 0:
                    warnings.warn(
                        f"ENBP breach: quoted_price {quoted_price:.2f} > enbp "
                        f"{enbp:.2f} for policy_id={policy_id!r}. "
                        "This will appear as non-compliant in the ENBP audit report.",
                        stacklevel=2,
                    )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO quotes
                  (policy_id, experiment_name, arm, model_version, quoted_price,
                   enbp, renewal_flag, enbp_flag, exposure, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    policy_id, experiment_name, arm, model_version,
                    float(quoted_price), enbp, int(renewal_flag),
                    enbp_flag, float(exposure), ts,
                ),
            )

    def log_bind(
        self,
        policy_id: str,
        bound_price: float,
        bound_timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Log a bind (policy purchase) event.

        Parameters
        ----------
        policy_id : str
            Must match a previously logged quote.
        bound_price : float
            The actual premium charged at bind. May differ from quoted_price
            (e.g. after broker adjustment).
        bound_timestamp : datetime, optional
            Defaults to UTC now.
        """
        if bound_price < 0:
            raise ValueError(f"bound_price must be non-negative, got {bound_price}.")
        ts = (bound_timestamp or datetime.now(timezone.utc)).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO binds (policy_id, bound_price, bound_timestamp) VALUES (?, ?, ?)",
                (policy_id, float(bound_price), ts),
            )

    def log_claim(
        self,
        policy_id: str,
        claim_date: date,
        claim_amount: float,
        development_month: int,
    ) -> None:
        """
        Log a claim development entry.

        Parameters
        ----------
        policy_id : str
            Policy against which the claim is recorded.
        claim_date : date
            Date of loss.
        claim_amount : float
            Incurred amount at this development month.
        development_month : int
            Months since claim event (0 = FNOL, 12 = 12-month development).
            Call this method again with the same policy_id/claim_date and a
            higher development_month to record development updates.
        """
        if claim_amount < 0:
            raise ValueError(f"claim_amount must be non-negative, got {claim_amount}.")
        if development_month < 0:
            raise ValueError(f"development_month must be >= 0, got {development_month}.")
        logged_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO claims (policy_id, claim_date, claim_amount, development_month, logged_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (policy_id, str(claim_date), float(claim_amount), development_month, logged_at),
            )

    # ------------------------------------------------------------------
    # Read methods
    # ------------------------------------------------------------------

    def query_quotes(self, experiment_name: Optional[str] = None) -> list[dict]:
        """Return all quote records, optionally filtered by experiment."""
        with self._connect() as conn:
            if experiment_name:
                rows = conn.execute(
                    "SELECT * FROM quotes WHERE experiment_name = ? ORDER BY timestamp",
                    (experiment_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM quotes ORDER BY timestamp"
                ).fetchall()
        return [dict(r) for r in rows]

    def query_binds(self) -> list[dict]:
        """Return all bind records."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM binds ORDER BY bound_timestamp").fetchall()
        return [dict(r) for r in rows]

    def query_claims(self) -> list[dict]:
        """Return all claim records."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM claims ORDER BY claim_date, development_month").fetchall()
        return [dict(r) for r in rows]

    def quote_count(self, experiment_name: Optional[str] = None) -> int:
        """Count quote records."""
        with self._connect() as conn:
            if experiment_name:
                return conn.execute(
                    "SELECT COUNT(*) FROM quotes WHERE experiment_name = ?",
                    (experiment_name,),
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
