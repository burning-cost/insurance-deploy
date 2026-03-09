"""
KPITracker — compute insurance KPIs by experiment cohort.

Metric maturity tiers
---------------------
Tier 1 (immediate): quote volume, price distribution, ENBP compliance rate.
Tier 2 (at bind): hit rate, GWP per bound policy.
Tier 3 (3–6 months): claim frequency (immature, IBNR caveat).
Tier 4 (12+ months): developed loss ratio. Bootstrap comparison in comparison.py.

The tracker queries the QuoteLogger's SQLite database. All computation is in
Python/NumPy — no pandas required for basic KPIs, though summary_report()
returns a DataFrame for presentation convenience.

Power analysis
--------------
Estimates months to statistical significance for hit rate and loss ratio given
current volumes and split. Output is intentionally conservative: the library
should make teams set realistic expectations, not false ones. UK motor at 10%
challenger split needs roughly 15 months for credible LR comparison; the power
analysis will say so.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np

from .logger import QuoteLogger

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# Minimum development months before LR is considered credible
DEFAULT_LR_MIN_DEVELOPMENT = 12
# Minimum policies per arm before any statistical test is meaningful
DEFAULT_MIN_POLICIES = 50


class KPITracker:
    """
    Compute insurance KPIs from a QuoteLogger.

    Parameters
    ----------
    logger : QuoteLogger
        The audit log to query.

    Examples
    --------
    >>> tracker = KPITracker(logger)
    >>> tracker.hit_rate("v3_vs_v2")
    {'champion': {'quoted': 900, 'bound': 288, 'hit_rate': 0.32},
     'challenger': {'quoted': 100, 'bound': 31, 'hit_rate': 0.31}}
    """

    def __init__(self, logger: QuoteLogger) -> None:
        self.logger = logger

    # ------------------------------------------------------------------
    # Tier 1: Immediately observable
    # ------------------------------------------------------------------

    def quote_volume(self, experiment_name: str) -> dict[str, Any]:
        """
        Quote counts and price distribution by arm.

        Returns
        -------
        dict with keys 'champion' and 'challenger', each containing:
        ``{'n': int, 'mean_price': float, 'median_price': float,
           'p25_price': float, 'p75_price': float}``
        """
        quotes = self.logger.query_quotes(experiment_name)
        return _summarise_prices_by_arm(quotes)

    def enbp_compliance(self, experiment_name: str) -> dict[str, Any]:
        """
        ENBP compliance rate for renewal quotes by arm.

        Only includes quotes where renewal_flag=1 and enbp was provided.
        Returns
        -------
        dict with keys 'champion' and 'challenger', each containing:
        ``{'renewal_quotes': int, 'compliant': int, 'breaches': int,
           'compliance_rate': float}``
        """
        quotes = self.logger.query_quotes(experiment_name)
        result = {}
        for arm in ("champion", "challenger"):
            renewal = [
                q for q in quotes
                if q["arm"] == arm and q["renewal_flag"] == 1 and q["enbp_flag"] is not None
            ]
            compliant = sum(1 for q in renewal if q["enbp_flag"] == 1)
            breaches = len(renewal) - compliant
            rate = compliant / len(renewal) if renewal else float("nan")
            result[arm] = {
                "renewal_quotes": len(renewal),
                "compliant": compliant,
                "breaches": breaches,
                "compliance_rate": rate,
            }
        return result

    # ------------------------------------------------------------------
    # Tier 2: Observable at bind
    # ------------------------------------------------------------------

    def hit_rate(
        self, experiment_name: str, cohort: str = "all"
    ) -> dict[str, Any]:
        """
        Conversion rate (bound / quoted) by arm.

        Parameters
        ----------
        experiment_name : str
        cohort : str
            ``'all'``, ``'champion'``, or ``'challenger'``.

        Returns
        -------
        dict keyed by arm with ``{'quoted': int, 'bound': int, 'hit_rate': float}``.
        """
        quotes = self.logger.query_quotes(experiment_name)
        binds = self.logger.query_binds()
        bound_pids = {b["policy_id"] for b in binds}

        result = {}
        arms = ("champion", "challenger") if cohort == "all" else (cohort,)
        for arm in arms:
            arm_quotes = [q for q in quotes if q["arm"] == arm]
            # One quote per policy_id (first quote per policy in this experiment)
            seen: set[str] = set()
            unique_quotes = []
            for q in arm_quotes:
                if q["policy_id"] not in seen:
                    seen.add(q["policy_id"])
                    unique_quotes.append(q)
            n_quoted = len(unique_quotes)
            n_bound = sum(1 for q in unique_quotes if q["policy_id"] in bound_pids)
            rate = n_bound / n_quoted if n_quoted > 0 else float("nan")
            result[arm] = {"quoted": n_quoted, "bound": n_bound, "hit_rate": rate}
        return result

    def gwp(self, experiment_name: str) -> dict[str, Any]:
        """
        Gross Written Premium on bound policies by arm.

        Joins quotes to binds by policy_id. Uses bound_price from binds table
        (not quoted_price — these may differ after broker adjustment).

        Returns
        -------
        dict keyed by arm with
        ``{'bound_policies': int, 'total_gwp': float, 'mean_gwp': float}``.
        """
        quotes = self.logger.query_quotes(experiment_name)
        binds = self.logger.query_binds()
        bind_map = {b["policy_id"]: b["bound_price"] for b in binds}

        # Map policy_id -> arm (take arm from first quote for this policy)
        policy_arm: dict[str, str] = {}
        for q in quotes:
            if q["policy_id"] not in policy_arm:
                policy_arm[q["policy_id"]] = q["arm"]

        result = {}
        for arm in ("champion", "challenger"):
            premiums = [
                bind_map[pid]
                for pid, a in policy_arm.items()
                if a == arm and pid in bind_map
            ]
            total = sum(premiums)
            mean = total / len(premiums) if premiums else float("nan")
            result[arm] = {
                "bound_policies": len(premiums),
                "total_gwp": total,
                "mean_gwp": mean,
            }
        return result

    # ------------------------------------------------------------------
    # Tier 3: 3–6 months development
    # ------------------------------------------------------------------

    def frequency(
        self,
        experiment_name: str,
        development_months: int = 6,
        warn_immature: bool = True,
    ) -> dict[str, Any]:
        """
        Claim frequency by arm (claims per earned policy-year).

        Parameters
        ----------
        development_months : int
            Minimum development month to include in frequency calculation.
            Claims with development_month < this threshold are excluded.
        warn_immature : bool
            If True, warn when development_months < 12 (motor motor IBNR caveat).

        Returns
        -------
        dict keyed by arm with
        ``{'policy_years': float, 'claim_count': int, 'frequency': float,
           'maturity_warning': bool}``.
        """
        if warn_immature and development_months < 12:
            warnings.warn(
                f"Claim frequency computed at {development_months} months development. "
                "At this stage, approximately 30–40% of ultimate motor claims may not "
                "yet be reported (IBNR). Treat as indicative, not credible.",
                stacklevel=2,
            )

        quotes = self.logger.query_quotes(experiment_name)
        binds = self.logger.query_binds()
        claims = self.logger.query_claims()
        bound_pids = {b["policy_id"] for b in binds}

        # Map policy_id -> arm, exposure
        policy_arm: dict[str, str] = {}
        policy_exposure: dict[str, float] = {}
        for q in quotes:
            if q["policy_id"] not in policy_arm:
                policy_arm[q["policy_id"]] = q["arm"]
                policy_exposure[q["policy_id"]] = q["exposure"]

        # Count distinct claims at or after development_months threshold
        # Use max development_month per (policy_id, claim_date) pair
        claim_pairs: dict[tuple, float] = {}
        for c in claims:
            key = (c["policy_id"], c["claim_date"])
            if c["development_month"] >= development_months:
                if key not in claim_pairs or c["development_month"] > claim_pairs.get(key, -1):
                    claim_pairs[key] = c["claim_amount"]

        result = {}
        for arm in ("champion", "challenger"):
            arm_pids = {pid for pid, a in policy_arm.items() if a == arm and pid in bound_pids}
            policy_years = sum(policy_exposure.get(pid, 1.0) for pid in arm_pids)
            claim_count = sum(
                1 for (pid, _) in claim_pairs if pid in arm_pids
            )
            freq = claim_count / policy_years if policy_years > 0 else float("nan")
            result[arm] = {
                "policy_years": policy_years,
                "claim_count": claim_count,
                "frequency": freq,
                "maturity_warning": development_months < 12,
            }
        return result

    # ------------------------------------------------------------------
    # Tier 4: 12+ months development
    # ------------------------------------------------------------------

    def loss_ratio(
        self,
        experiment_name: str,
        development_months: int = DEFAULT_LR_MIN_DEVELOPMENT,
    ) -> dict[str, Any]:
        """
        Developed loss ratio by arm.

        Loss ratio = total incurred claims / total earned premium.
        Earned premium = sum of bound_price × exposure for bound policies.

        Parameters
        ----------
        development_months : int
            Only include claims at this development stage or later.
            Raises warning if < 12 months.

        Returns
        -------
        dict keyed by arm with
        ``{'earned_premium': float, 'incurred_claims': float,
           'loss_ratio': float, 'policy_count': int, 'maturity_warning': bool}``.
        """
        maturity_warning = development_months < DEFAULT_LR_MIN_DEVELOPMENT
        if maturity_warning:
            warnings.warn(
                f"Loss ratio requested at {development_months} months development, "
                f"below the {DEFAULT_LR_MIN_DEVELOPMENT}-month recommended minimum. "
                "Loss ratios on immature data are unreliable. "
                "Power analysis shows you need ~12 months minimum for motor.",
                stacklevel=2,
            )

        quotes = self.logger.query_quotes(experiment_name)
        binds = self.logger.query_binds()
        claims = self.logger.query_claims()

        bind_map = {b["policy_id"]: b["bound_price"] for b in binds}
        bound_pids = set(bind_map.keys())

        policy_arm: dict[str, str] = {}
        policy_exposure: dict[str, float] = {}
        for q in quotes:
            if q["policy_id"] not in policy_arm:
                policy_arm[q["policy_id"]] = q["arm"]
                policy_exposure[q["policy_id"]] = q["exposure"]

        # Latest incurred per (policy_id, claim_date) at >= development_months
        claim_incurred: dict[tuple, float] = {}
        claim_dm: dict[tuple, int] = {}
        for c in claims:
            if c["development_month"] >= development_months:
                key = (c["policy_id"], c["claim_date"])
                if c["development_month"] > claim_dm.get(key, -1):
                    claim_incurred[key] = c["claim_amount"]
                    claim_dm[key] = c["development_month"]

        result = {}
        for arm in ("champion", "challenger"):
            arm_pids = {pid for pid, a in policy_arm.items() if a == arm and pid in bound_pids}
            earned = sum(
                bind_map[pid] * policy_exposure.get(pid, 1.0) for pid in arm_pids
            )
            incurred = sum(
                amt for (pid, _), amt in claim_incurred.items() if pid in arm_pids
            )
            lr = incurred / earned if earned > 0 else float("nan")
            result[arm] = {
                "earned_premium": earned,
                "incurred_claims": incurred,
                "loss_ratio": lr,
                "policy_count": len(arm_pids),
                "maturity_warning": maturity_warning,
            }
        return result

    def severity(
        self,
        experiment_name: str,
        development_months: int = 12,
    ) -> dict[str, Any]:
        """
        Mean claim severity by arm at specified development.

        Returns
        -------
        dict keyed by arm with
        ``{'claim_count': int, 'mean_severity': float, 'total_incurred': float}``.
        """
        quotes = self.logger.query_quotes(experiment_name)
        binds = self.logger.query_binds()
        claims = self.logger.query_claims()
        bound_pids = {b["policy_id"] for b in binds}

        policy_arm: dict[str, str] = {}
        for q in quotes:
            if q["policy_id"] not in policy_arm:
                policy_arm[q["policy_id"]] = q["arm"]

        # Latest incurred per (policy_id, claim_date)
        claim_incurred: dict[tuple, float] = {}
        claim_dm: dict[tuple, int] = {}
        for c in claims:
            key = (c["policy_id"], c["claim_date"])
            if c["development_month"] >= development_months:
                if c["development_month"] > claim_dm.get(key, -1):
                    claim_incurred[key] = c["claim_amount"]
                    claim_dm[key] = c["development_month"]

        result = {}
        for arm in ("champion", "challenger"):
            arm_pids = {pid for pid, a in policy_arm.items() if a == arm and pid in bound_pids}
            amounts = [amt for (pid, _), amt in claim_incurred.items() if pid in arm_pids]
            mean_sev = float(np.mean(amounts)) if amounts else float("nan")
            result[arm] = {
                "claim_count": len(amounts),
                "mean_severity": mean_sev,
                "total_incurred": sum(amounts),
            }
        return result

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary_report(self, experiment_name: str) -> Any:
        """
        Tabular summary of all available KPIs.

        Returns a pandas DataFrame if pandas is available, else a list of dicts.
        """
        vol = self.quote_volume(experiment_name)
        hr = self.hit_rate(experiment_name)
        gw = self.gwp(experiment_name)

        rows = []
        for arm in ("champion", "challenger"):
            rows.append({
                "arm": arm,
                "quoted_policies": vol.get(arm, {}).get("n", 0),
                "mean_quoted_price": vol.get(arm, {}).get("mean_price", float("nan")),
                "bound_policies": hr.get(arm, {}).get("bound", 0),
                "hit_rate": hr.get(arm, {}).get("hit_rate", float("nan")),
                "total_gwp": gw.get(arm, {}).get("total_gwp", 0.0),
                "mean_gwp": gw.get(arm, {}).get("mean_gwp", float("nan")),
            })

        if _PANDAS_AVAILABLE:
            return pd.DataFrame(rows).set_index("arm")
        return rows

    # ------------------------------------------------------------------
    # Power analysis
    # ------------------------------------------------------------------

    def power_analysis(
        self,
        experiment_name: str,
        target_delta_lr: float = 0.03,
        target_delta_hr: float = 0.02,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> dict[str, Any]:
        """
        Estimate months to statistical significance given current volumes.

        This is a first-class output, not an afterthought. Pricing teams
        routinely underestimate how long champion/challenger experiments take.
        This method makes the timeline explicit.

        Parameters
        ----------
        experiment_name : str
        target_delta_lr : float
            Minimum detectable difference in loss ratio (absolute). Default 0.03 (3pp).
        target_delta_hr : float
            Minimum detectable difference in hit rate (absolute). Default 0.02 (2pp).
        alpha : float
            Type I error rate. Default 0.05.
        power : float
            Statistical power. Default 0.80.

        Returns
        -------
        dict with keys:
        ``{'current_n_champion': int, 'current_n_challenger': int,
           'monthly_rate_champion': float, 'monthly_rate_challenger': float,
           'hr_required_n': int, 'hr_months_to_significance': float,
           'lr_required_n': int, 'lr_months_to_significance': float,
           'lr_total_months_with_development': float,
           'notes': list[str]}``
        """
        hr = self.hit_rate(experiment_name)
        champ = hr.get("champion", {})
        chall = hr.get("challenger", {})

        n_champ = champ.get("quoted", 0)
        n_chall = chall.get("quoted", 0)

        # Estimate monthly quote rate from time span in logs
        quotes = self.logger.query_quotes(experiment_name)
        months_elapsed = _estimate_months(quotes)

        monthly_champ = n_champ / months_elapsed if months_elapsed > 0 else 0.0
        monthly_chall = n_chall / months_elapsed if months_elapsed > 0 else 0.0

        # Hit rate: two-proportion z-test sample size
        # Using pooled proportion estimate
        p_champ = champ.get("hit_rate", 0.30)
        p_chall = chall.get("hit_rate", 0.30)
        p_bar = (p_champ + p_chall) / 2
        if math.isnan(p_bar):
            p_bar = 0.30  # default assumption

        hr_n = _two_proportion_sample_size(p_bar, target_delta_hr, alpha, power)
        hr_months = hr_n / monthly_chall if monthly_chall > 0 else float("inf")

        # Loss ratio: rough approximation using normal approximation
        # LR variance approximated from observed data or assumed sigma
        # Typical motor LR ~0.65 with CV ~0.4, so sigma_lr ~0.26
        sigma_lr = 0.26
        lr_n = _one_sample_mean_sample_size(sigma_lr, target_delta_lr, alpha, power)
        lr_months_to_bind = lr_n / monthly_chall if monthly_chall > 0 else float("inf")
        lr_total = lr_months_to_bind + DEFAULT_LR_MIN_DEVELOPMENT  # bind + develop

        notes = []
        if n_chall < DEFAULT_MIN_POLICIES:
            notes.append(
                f"Challenger has only {n_chall} quoted policies — "
                "volumes too low for reliable estimates."
            )
        if months_elapsed < 1:
            notes.append("Less than 1 month of data — monthly rate estimates unreliable.")
        notes.append(
            f"LR estimate assumes motor sigma_LR ≈ 0.26 and "
            f"{DEFAULT_LR_MIN_DEVELOPMENT}-month development period. "
            "Home insurance or long-tail lines will take longer."
        )
        notes.append(
            "These are point estimates. Actual significance depends on true LR "
            "volatility in your book. Run bootstrap_lr_test() once data matures."
        )

        return {
            "current_n_champion": n_champ,
            "current_n_challenger": n_chall,
            "months_elapsed": months_elapsed,
            "monthly_rate_champion": monthly_champ,
            "monthly_rate_challenger": monthly_chall,
            "hr_required_n_per_arm": hr_n,
            "hr_months_to_significance": hr_months,
            "lr_required_n_per_arm": lr_n,
            "lr_months_to_bind": lr_months_to_bind,
            "lr_total_months_with_development": lr_total,
            "target_delta_lr": target_delta_lr,
            "target_delta_hr": target_delta_hr,
            "notes": notes,
        }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _summarise_prices_by_arm(quotes: list[dict]) -> dict[str, Any]:
    result = {}
    for arm in ("champion", "challenger"):
        prices = [q["quoted_price"] for q in quotes if q["arm"] == arm]
        if not prices:
            result[arm] = {
                "n": 0,
                "mean_price": float("nan"),
                "median_price": float("nan"),
                "p25_price": float("nan"),
                "p75_price": float("nan"),
            }
        else:
            arr = np.array(prices)
            result[arm] = {
                "n": len(prices),
                "mean_price": float(np.mean(arr)),
                "median_price": float(np.median(arr)),
                "p25_price": float(np.percentile(arr, 25)),
                "p75_price": float(np.percentile(arr, 75)),
            }
    return result


def _estimate_months(quotes: list[dict]) -> float:
    """Estimate experiment duration in months from quote timestamps."""
    if len(quotes) < 2:
        return 0.0
    timestamps = sorted(q["timestamp"] for q in quotes)
    # Parse first and last
    from datetime import datetime
    fmt = lambda s: datetime.fromisoformat(s)
    try:
        t0 = fmt(timestamps[0])
        t1 = fmt(timestamps[-1])
        delta_days = (t1 - t0).total_seconds() / 86400
        return delta_days / 30.44  # average days per month
    except Exception:
        return 0.0


def _two_proportion_sample_size(
    p: float, delta: float, alpha: float, power: float
) -> int:
    """
    Two-proportion z-test sample size per arm.

    N = (z_alpha/2 + z_beta)^2 * 2 * p * (1-p) / delta^2
    """
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2 * 2 * p * (1 - p)) / (delta ** 2)
    return max(1, math.ceil(n))


def _one_sample_mean_sample_size(
    sigma: float, delta: float, alpha: float, power: float
) -> int:
    """
    One-sample mean test sample size.

    N = (z_alpha/2 + z_beta)^2 * sigma^2 / delta^2
    """
    from scipy import stats
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2 * sigma ** 2) / (delta ** 2)
    return max(1, math.ceil(n))
