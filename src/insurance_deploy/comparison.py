"""
ModelComparison — statistical tests for champion/challenger promotion decisions.

Three tests:

1. bootstrap_lr_test: block bootstrap on policy-level loss ratios.
   Most appropriate for developed (12m+) claims data. Policy-level resampling
   preserves intra-policy correlation. 10,000 iterations by default.

2. hit_rate_test: two-proportion z-test on conversion rates.
   Available after a few months. Fast signal but confounded by price differences
   in live mode.

3. frequency_test: Poisson GLM with model_version as covariate.
   Available at 6–9 months. Frequency is a better early signal than LR because
   it doesn't require claims to be fully developed.

Recommendation engine
---------------------
The library produces one of three conclusions:
  INSUFFICIENT_EVIDENCE — continue experiment, show N required
  CHALLENGER_BETTER — human review recommended for promotion
  CHAMPION_BETTER — consider terminating experiment

Never automatic promotion. The library surfaces evidence; humans decide.

Adverse selection warning
-------------------------
In live mode, if champion and challenger quote different prices, the bound
cohorts will have different risk mixes. Observed LR difference ≠ model quality
difference. This is flagged in the ComparisonResult. Shadow mode avoids this
problem entirely.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .kpi import KPITracker, DEFAULT_LR_MIN_DEVELOPMENT, DEFAULT_MIN_POLICIES


@dataclass
class ComparisonResult:
    """Output of a statistical comparison between champion and challenger."""

    test_name: str
    experiment_name: str
    champion_estimate: float
    challenger_estimate: float
    difference: float          # challenger - champion
    ci_lower: float            # 95% CI lower
    ci_upper: float            # 95% CI upper
    p_value: float
    n_champion: int
    n_challenger: int
    conclusion: str            # INSUFFICIENT_EVIDENCE | CHALLENGER_BETTER | CHAMPION_BETTER
    recommendation: str
    maturity_warning: bool = False
    adverse_selection_warning: bool = False
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []

    def __repr__(self) -> str:
        return (
            f"ComparisonResult({self.test_name!r}, "
            f"diff={self.difference:+.4f}, "
            f"p={self.p_value:.4f}, "
            f"n=[{self.n_champion}, {self.n_challenger}], "
            f"{self.conclusion})"
        )

    def summary(self) -> str:
        """Human-readable summary for reports and notebooks."""
        lines = [
            f"Test: {self.test_name} | Experiment: {self.experiment_name}",
            f"Champion estimate: {self.champion_estimate:.4f} (n={self.n_champion})",
            f"Challenger estimate: {self.challenger_estimate:.4f} (n={self.n_challenger})",
            f"Difference (challenger - champion): {self.difference:+.4f}",
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"p-value: {self.p_value:.4f}",
            f"",
            f"Conclusion: {self.conclusion}",
            f"Recommendation: {self.recommendation}",
        ]
        if self.maturity_warning:
            lines.append("WARNING: Claims data is immature. Treat LR results with caution.")
        if self.adverse_selection_warning:
            lines.append(
                "WARNING: Adverse selection bias possible in live mode. "
                "LR difference may not purely reflect model quality."
            )
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)


class ModelComparison:
    """
    Statistical comparison between champion and challenger cohorts.

    Parameters
    ----------
    tracker : KPITracker
        Tracker wrapping the experiment's QuoteLogger.

    Examples
    --------
    >>> comp = ModelComparison(tracker)
    >>> result = comp.bootstrap_lr_test("v3_vs_v2", development_months=12)
    >>> print(result.summary())
    """

    def __init__(self, tracker: KPITracker) -> None:
        self.tracker = tracker

    # ------------------------------------------------------------------
    # Bootstrap loss ratio test
    # ------------------------------------------------------------------

    def bootstrap_lr_test(
        self,
        experiment_name: str,
        n_bootstrap: int = 10_000,
        development_months: int = DEFAULT_LR_MIN_DEVELOPMENT,
        seed: Optional[int] = None,
    ) -> ComparisonResult:
        """
        Block bootstrap comparison of loss ratios between arms.

        Resamples at policy level (preserving within-policy correlation).
        Reports point estimate, 95% percentile CI, and one-sided p-value
        P(challenger LR < champion LR).

        Parameters
        ----------
        experiment_name : str
        n_bootstrap : int
            Bootstrap iterations. Default 10,000.
        development_months : int
            Minimum claim development months to include. Default 12.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ComparisonResult
        """
        maturity_warning = development_months < DEFAULT_LR_MIN_DEVELOPMENT
        if maturity_warning:
            warnings.warn(
                f"bootstrap_lr_test called with development_months={development_months}, "
                f"below recommended minimum of {DEFAULT_LR_MIN_DEVELOPMENT}. "
                "Results should be treated as exploratory, not for promotion decisions.",
                stacklevel=2,
            )

        policy_data = self._build_policy_loss_data(experiment_name, development_months)

        champ_data = [(p, e, c) for p, e, c, arm in policy_data if arm == "champion"]
        chall_data = [(p, e, c) for p, e, c, arm in policy_data if arm == "challenger"]

        n_champ = len(champ_data)
        n_chall = len(chall_data)

        notes = []

        if n_champ < DEFAULT_MIN_POLICIES:
            notes.append(
                f"Champion arm has only {n_champ} bound policies — "
                "below minimum for reliable bootstrap."
            )
        if n_chall < DEFAULT_MIN_POLICIES:
            notes.append(
                f"Challenger arm has only {n_chall} bound policies — "
                "below minimum for reliable bootstrap."
            )

        if not champ_data or not chall_data:
            return ComparisonResult(
                test_name="bootstrap_lr_test",
                experiment_name=experiment_name,
                champion_estimate=float("nan"),
                challenger_estimate=float("nan"),
                difference=float("nan"),
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                p_value=float("nan"),
                n_champion=n_champ,
                n_challenger=n_chall,
                conclusion="INSUFFICIENT_EVIDENCE",
                recommendation="Insufficient data to run bootstrap test. Continue experiment.",
                maturity_warning=maturity_warning,
                notes=notes,
            )

        # Point estimates
        lr_champ = _loss_ratio(champ_data)
        lr_chall = _loss_ratio(chall_data)
        point_diff = lr_chall - lr_champ

        # Bootstrap
        rng = np.random.default_rng(seed)
        boot_diffs = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            bs_champ = rng.choice(len(champ_data), size=len(champ_data), replace=True)
            bs_chall = rng.choice(len(chall_data), size=len(chall_data), replace=True)
            lr_c = _loss_ratio([champ_data[j] for j in bs_champ])
            lr_h = _loss_ratio([chall_data[j] for j in bs_chall])
            boot_diffs[i] = lr_h - lr_c

        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))

        # One-sided p-value: P(challenger LR >= champion LR | H0: no difference)
        # Under H0, centre bootstrap distribution around 0
        centred = boot_diffs - np.mean(boot_diffs)
        p_value = float(np.mean(centred >= point_diff))
        p_value = max(1.0 / n_bootstrap, p_value)  # avoid exact zero

        conclusion, recommendation = _conclude(
            point_diff, ci_lower, ci_upper, p_value, n_champ, n_chall,
            metric="loss_ratio", lower_is_better=True
        )

        return ComparisonResult(
            test_name="bootstrap_lr_test",
            experiment_name=experiment_name,
            champion_estimate=lr_champ,
            challenger_estimate=lr_chall,
            difference=point_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_champion=n_champ,
            n_challenger=n_chall,
            conclusion=conclusion,
            recommendation=recommendation,
            maturity_warning=maturity_warning,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Hit rate test (two-proportion z-test)
    # ------------------------------------------------------------------

    def hit_rate_test(
        self,
        experiment_name: str,
        alpha: float = 0.05,
    ) -> ComparisonResult:
        """
        Two-proportion z-test on conversion rates.

        Tests H0: hit_rate_champion == hit_rate_challenger.
        Two-sided test; p-value < alpha implies significant difference.

        Returns
        -------
        ComparisonResult
        """
        from scipy import stats

        hr = self.tracker.hit_rate(experiment_name)
        champ = hr.get("champion", {})
        chall = hr.get("challenger", {})

        n1 = champ.get("quoted", 0)
        n2 = chall.get("quoted", 0)
        x1 = champ.get("bound", 0)
        x2 = chall.get("bound", 0)

        p1 = x1 / n1 if n1 > 0 else float("nan")
        p2 = x2 / n2 if n2 > 0 else float("nan")

        notes = []
        if n1 < DEFAULT_MIN_POLICIES or n2 < DEFAULT_MIN_POLICIES:
            notes.append(f"Small sample sizes (n_champion={n1}, n_challenger={n2}).")

        if n1 == 0 or n2 == 0 or math.isnan(p1) or math.isnan(p2):
            return ComparisonResult(
                test_name="hit_rate_test",
                experiment_name=experiment_name,
                champion_estimate=p1,
                challenger_estimate=p2,
                difference=float("nan"),
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                p_value=float("nan"),
                n_champion=n1,
                n_challenger=n2,
                conclusion="INSUFFICIENT_EVIDENCE",
                recommendation="No data to test. Continue experiment.",
                notes=notes,
            )

        # Two-proportion z-test
        p_bar = (x1 + x2) / (n1 + n2)
        se = math.sqrt(p_bar * (1 - p_bar) * (1 / n1 + 1 / n2))
        z = (p2 - p1) / se if se > 0 else 0.0
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

        # 95% CI on difference
        se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z_95 = stats.norm.ppf(0.975)
        diff = p2 - p1
        ci_lower = diff - z_95 * se_diff
        ci_upper = diff + z_95 * se_diff

        conclusion, recommendation = _conclude(
            diff, ci_lower, ci_upper, p_value, n1, n2,
            metric="hit_rate", lower_is_better=False
        )

        return ComparisonResult(
            test_name="hit_rate_test",
            experiment_name=experiment_name,
            champion_estimate=p1,
            challenger_estimate=p2,
            difference=diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_champion=n1,
            n_challenger=n2,
            conclusion=conclusion,
            recommendation=recommendation,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Frequency test (Poisson GLM)
    # ------------------------------------------------------------------

    def frequency_test(
        self,
        experiment_name: str,
        development_months: int = 6,
    ) -> ComparisonResult:
        """
        Poisson GLM comparison of claim frequency between arms.

        Model: claims ~ offset(log(exposure)) + arm, family=Poisson.
        The arm coefficient gives the log rate ratio (challenger vs champion).
        Wald test p-value on the arm coefficient.

        Parameters
        ----------
        development_months : int
            Minimum claim development months. Default 6.

        Returns
        -------
        ComparisonResult
        """
        from scipy import stats

        freq = self.tracker.frequency(experiment_name, development_months=development_months)
        champ = freq.get("champion", {})
        chall = freq.get("challenger", {})

        n_champ = champ.get("claim_count", 0)
        n_chall = chall.get("claim_count", 0)
        exp_champ = champ.get("policy_years", 0.0)
        exp_chall = chall.get("policy_years", 0.0)
        f_champ = champ.get("frequency", float("nan"))
        f_chall = chall.get("frequency", float("nan"))

        notes = []
        if champ.get("maturity_warning"):
            notes.append(f"Frequency computed at {development_months} months — IBNR caveat applies.")

        if exp_champ == 0 or exp_chall == 0 or n_champ == 0 or n_chall == 0:
            return ComparisonResult(
                test_name="frequency_test",
                experiment_name=experiment_name,
                champion_estimate=f_champ,
                challenger_estimate=f_chall,
                difference=float("nan"),
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                p_value=float("nan"),
                n_champion=n_champ,
                n_challenger=n_chall,
                conclusion="INSUFFICIENT_EVIDENCE",
                recommendation="No claim data available. Continue experiment.",
                notes=notes,
            )

        # Poisson GLM approximation via rate ratio test
        # log rate ratio = log(f_chall) - log(f_champ)
        # Variance approximation: 1/n_chall + 1/n_champ (Poisson asymptotic)
        log_rr = math.log(f_chall) - math.log(f_champ)
        se_log_rr = math.sqrt(1.0 / n_chall + 1.0 / n_champ)
        z = log_rr / se_log_rr
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

        z_95 = stats.norm.ppf(0.975)
        ci_lower_rr = math.exp(log_rr - z_95 * se_log_rr)
        ci_upper_rr = math.exp(log_rr + z_95 * se_log_rr)

        # For ComparisonResult, use frequency scale for estimates and CI on difference
        diff = f_chall - f_champ
        ci_lower = f_champ * (ci_lower_rr - 1)
        ci_upper = f_champ * (ci_upper_rr - 1)

        conclusion, recommendation = _conclude(
            diff, ci_lower, ci_upper, p_value, n_champ, n_chall,
            metric="frequency", lower_is_better=True
        )

        return ComparisonResult(
            test_name="frequency_test",
            experiment_name=experiment_name,
            champion_estimate=f_champ,
            challenger_estimate=f_chall,
            difference=diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_champion=n_champ,
            n_challenger=n_chall,
            conclusion=conclusion,
            recommendation=recommendation,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_policy_loss_data(
        self,
        experiment_name: str,
        development_months: int,
    ) -> list[tuple]:
        """
        Build (premium, exposure, incurred, arm) tuples for bound policies.
        """
        quotes = self.tracker.logger.query_quotes(experiment_name)
        binds = self.tracker.logger.query_binds()
        claims = self.tracker.logger.query_claims()

        bind_map = {b["policy_id"]: b["bound_price"] for b in binds}
        bound_pids = set(bind_map.keys())

        policy_arm: dict[str, str] = {}
        policy_exposure: dict[str, float] = {}
        for q in quotes:
            if q["policy_id"] not in policy_arm:
                policy_arm[q["policy_id"]] = q["arm"]
                policy_exposure[q["policy_id"]] = q["exposure"]

        # Latest incurred per (policy_id, claim_date) at >= development_months
        claim_incurred: dict[str, float] = {}  # policy_id -> total incurred
        claim_dm: dict[tuple, int] = {}
        claim_amt: dict[tuple, float] = {}

        for c in claims:
            key = (c["policy_id"], c["claim_date"])
            if c["development_month"] >= development_months:
                if c["development_month"] > claim_dm.get(key, -1):
                    claim_dm[key] = c["development_month"]
                    claim_amt[key] = c["claim_amount"]

        # Aggregate to policy level
        policy_incurred: dict[str, float] = {}
        for (pid, _), amt in claim_amt.items():
            policy_incurred[pid] = policy_incurred.get(pid, 0.0) + amt

        result = []
        for pid in bound_pids:
            if pid not in policy_arm:
                continue
            arm = policy_arm[pid]
            premium = bind_map[pid]
            exposure = policy_exposure.get(pid, 1.0)
            incurred = policy_incurred.get(pid, 0.0)
            result.append((premium, exposure, incurred, arm))

        return result


# ------------------------------------------------------------------
# Internal functions
# ------------------------------------------------------------------

def _loss_ratio(data: list[tuple]) -> float:
    """LR = sum(incurred) / sum(premium * exposure)."""
    if not data:
        return float("nan")
    total_premium = sum(p * e for p, e, c in data)
    total_claims = sum(c for p, e, c in data)
    return total_claims / total_premium if total_premium > 0 else float("nan")


def _conclude(
    diff: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    n1: int,
    n2: int,
    metric: str,
    lower_is_better: bool,
    alpha: float = 0.05,
    min_n: int = DEFAULT_MIN_POLICIES,
) -> tuple[str, str]:
    """Determine conclusion and recommendation from test results."""
    if n1 < min_n or n2 < min_n or math.isnan(p_value):
        return (
            "INSUFFICIENT_EVIDENCE",
            f"Sample sizes too small (n1={n1}, n2={n2}). "
            f"Minimum {min_n} per arm required. Continue experiment.",
        )

    if p_value >= alpha:
        return (
            "INSUFFICIENT_EVIDENCE",
            f"No statistically significant difference detected in {metric} "
            f"(p={p_value:.3f} >= {alpha}). Continue experiment. "
            "Consider running power_analysis() to estimate time to significance.",
        )

    # Significant result — is it in the right direction?
    challenger_better = diff < 0 if lower_is_better else diff > 0

    if challenger_better:
        return (
            "CHALLENGER_BETTER",
            f"Challenger shows significantly better {metric} "
            f"(p={p_value:.3f}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]). "
            "Human review recommended before promotion. Document the promotion "
            "decision with reviewer name and date.",
        )
    else:
        return (
            "CHAMPION_BETTER",
            f"Champion shows significantly better {metric} "
            f"(p={p_value:.3f}). "
            "Consider terminating the experiment. Retain logs for audit trail.",
        )
