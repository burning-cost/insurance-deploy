"""
Experiment — champion/challenger routing with deterministic hash assignment.

Routing methodology
-------------------
SHA-256(policy_id + experiment_name), take last 8 hex characters as an integer,
modulo 100. If result < challenger_pct * 100, route to challenger.

This is deterministic: given a policy_id and experiment name, the routing
decision is always the same and can be recomputed at any point from first
principles. Random assignment (random.random() < 0.1) is not reproducible —
unacceptable for an audit trail.

Assignment is by policy, not by quote. A policy that gets routed to challenger
on its first quote will always be routed to challenger within this experiment.
This is required for ENBP audit integrity: the pricing model must be consistent
across the lifecycle of each policy.

Shadow vs live mode
-------------------
shadow (default): champion handles all live quotes. Challenger scores in
parallel, output is logged but never returned to the customer. Zero regulatory
risk. Use this for model quality comparison.

live: routed model's price is used. challenger_pct fraction of policies see
challenger prices. Raises FCA Consumer Duty (PRIN 2A) fair value questions —
get legal sign-off before enabling. See README for discussion.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .registry import ModelVersion


VALID_MODES = ("shadow", "live")


@dataclass
class Experiment:
    """
    A champion/challenger experiment.

    Parameters
    ----------
    name : str
        Unique identifier for this experiment. Used in hash-based routing —
        changing the name changes all routing decisions.
    champion : ModelVersion
        The current production model. In shadow mode, always prices the quote.
    challenger : ModelVersion
        The model under test.
    challenger_pct : float
        Fraction of policies routed to challenger (0.0–1.0). Default 0.10.
        Only affects live mode; in shadow mode all quotes are priced by champion.
    mode : str
        ``'shadow'`` (default) or ``'live'``. Shadow mode has zero regulatory
        risk. Live mode requires careful FCA Consumer Duty consideration.

    Examples
    --------
    >>> exp = Experiment(
    ...     name="v3_vs_v2",
    ...     champion=registry.get("motor", "2.0"),
    ...     challenger=registry.get("motor", "3.0"),
    ...     challenger_pct=0.10,
    ...     mode="shadow",
    ... )
    >>> arm = exp.route("POL-12345")  # always "champion" or "challenger"
    """

    name: str
    champion: ModelVersion
    challenger: ModelVersion
    challenger_pct: float = 0.10
    mode: str = "shadow"
    created_at: str = ""
    deactivated_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.challenger_pct < 1.0:
            raise ValueError(
                f"challenger_pct must be between 0 and 1 (exclusive), "
                f"got {self.challenger_pct}."
            )
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES!r}, got {self.mode!r}."
            )
        if self.mode == "live":
            warnings.warn(
                "Live mode routes real quotes to challenger model. This may raise "
                "FCA Consumer Duty (PRIN 2A) fair value concerns — two customers "
                "of identical risk profile priced differently simultaneously. "
                "Obtain legal sign-off before enabling live mode in production. "
                "Shadow mode (default) carries zero regulatory risk.",
                stacklevel=2,
            )
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, policy_id: str) -> str:
        """
        Determine routing arm for a policy.

        Parameters
        ----------
        policy_id : str
            Unique policy identifier. Routing is stable within an experiment:
            the same policy_id always maps to the same arm.

        Returns
        -------
        str
            ``'champion'`` or ``'challenger'``.
        """
        if not self.is_active():
            raise RuntimeError(
                f"Experiment '{self.name}' is deactivated. "
                "Create a new experiment to run further tests."
            )
        key = (policy_id + self.name).encode()
        digest = hashlib.sha256(key).hexdigest()
        # Last 8 hex chars = 32-bit integer, modulo 100 gives 0-99
        slot = int(digest[-8:], 16) % 100
        threshold = int(self.challenger_pct * 100)
        return "challenger" if slot < threshold else "champion"

    def live_model(self, policy_id: str) -> ModelVersion:
        """
        Return the ModelVersion that should price this quote.

        In shadow mode, always returns champion regardless of routing.
        In live mode, returns the routed model.
        """
        arm = self.route(policy_id)
        if self.mode == "shadow":
            return self.champion
        return self.challenger if arm == "challenger" else self.champion

    def shadow_model(self, policy_id: str) -> ModelVersion:
        """
        Return the ModelVersion that should score in shadow (not price the quote).

        In shadow mode, returns challenger for challenger-routed policies,
        and champion (again) for champion-routed policies.
        In live mode, returns the non-live model.
        """
        arm = self.route(policy_id)
        if self.mode == "shadow":
            return self.challenger if arm == "challenger" else self.champion
        return self.champion if arm == "challenger" else self.challenger

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """True if the experiment has not been deactivated."""
        return self.deactivated_at is None

    def deactivate(self) -> None:
        """
        Deactivate the experiment.

        After deactivation, ``route()`` raises RuntimeError. Existing log
        records are unaffected — the audit trail is permanent.
        """
        self.deactivated_at = datetime.now(timezone.utc).isoformat()

    def __repr__(self) -> str:
        status = "active" if self.is_active() else "deactivated"
        return (
            f"Experiment('{self.name}', mode={self.mode!r}, "
            f"split={self.challenger_pct:.0%}, {status})"
        )
