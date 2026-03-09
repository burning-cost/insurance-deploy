"""
ENBPAuditReport — ICOBS 6B.2.51R compliance report generator.

Regulatory context
------------------
ICOBS 6B.2.51R requires firms to maintain written records demonstrating:
1. How they satisfy themselves there is no systematic tenure discrimination
2. How they resolved areas of discretion in determining compliance
3. Records sufficient to enable the annual attestation

FCA multi-firm review (2023) found 83% of firms non-compliant. This report
provides the technical record-keeping infrastructure. The SMF holder signing
the annual attestation needs to be able to point to a document like this.

What this report does NOT do
-----------------------------
It does not calculate ENBP. The firm's pricing team calculates ENBP per
ICOBS 6B methodology. This report presents your ENBP records, which must
have been logged via QuoteLogger.log_quote(enbp=..., renewal_flag=True).

If enbp was not provided to QuoteLogger, those quotes will appear as
'ENBP not recorded' in the report.

Output format
-------------
Markdown. Designed for inclusion in internal governance documents, attestation
packs, or Databricks notebook outputs. The recipient is an SMF holder or
compliance team, not a data scientist.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .logger import QuoteLogger


class ENBPAuditReport:
    """
    Generate ICOBS 6B.2.51R audit reports from QuoteLogger data.

    Parameters
    ----------
    logger : QuoteLogger
        The audit log to report against.

    Examples
    --------
    >>> reporter = ENBPAuditReport(logger)
    >>> md = reporter.generate(
    ...     experiment_name="v3_vs_v2",
    ...     period_start="2024-01-01",
    ...     period_end="2024-12-31",
    ... )
    >>> print(md)
    """

    def __init__(self, logger: QuoteLogger) -> None:
        self.logger = logger

    def generate(
        self,
        experiment_name: str,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        firm_name: str = "[Firm Name]",
        smf_holder: str = "[SMF Holder]",
    ) -> str:
        """
        Generate the ENBP audit report as markdown.

        Parameters
        ----------
        experiment_name : str
            Experiment to report on.
        period_start : str, optional
            ISO date string, e.g. ``"2024-01-01"``. Filters quotes by timestamp.
        period_end : str, optional
            ISO date string, e.g. ``"2024-12-31"``.
        firm_name : str
            Firm name for report header.
        smf_holder : str
            SMF holder name for attestation section.

        Returns
        -------
        str
            Markdown-formatted audit report.
        """
        quotes = self.logger.query_quotes(experiment_name)
        quotes = _filter_by_period(quotes, period_start, period_end)

        renewal_quotes = [q for q in quotes if q["renewal_flag"] == 1]
        nb_quotes = [q for q in quotes if q["renewal_flag"] == 0]

        enbp_provided = [q for q in renewal_quotes if q["enbp_flag"] is not None]
        compliant = [q for q in enbp_provided if q["enbp_flag"] == 1]
        breaches = [q for q in enbp_provided if q["enbp_flag"] == 0]
        no_enbp = [q for q in renewal_quotes if q["enbp_flag"] is None]

        # By arm
        arm_stats = {}
        for arm in ("champion", "challenger"):
            arm_renewals = [q for q in renewal_quotes if q["arm"] == arm]
            arm_enbp = [q for q in arm_renewals if q["enbp_flag"] is not None]
            arm_ok = [q for q in arm_enbp if q["enbp_flag"] == 1]
            arm_breach = [q for q in arm_enbp if q["enbp_flag"] == 0]
            arm_stats[arm] = {
                "renewals": len(arm_renewals),
                "enbp_provided": len(arm_enbp),
                "compliant": len(arm_ok),
                "breaches": len(arm_breach),
                "no_enbp_recorded": len(arm_renewals) - len(arm_enbp),
                "compliance_rate": len(arm_ok) / len(arm_enbp) if arm_enbp else None,
            }

        # Model versions used
        model_versions = {}
        for q in quotes:
            mv = q["model_version"]
            if mv not in model_versions:
                model_versions[mv] = {"arm": q["arm"], "count": 0}
            model_versions[mv]["count"] += 1

        # Build report
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        period_label = _period_label(period_start, period_end)

        lines = [
            f"# ENBP Compliance Audit Report",
            f"",
            f"**Firm:** {firm_name}  ",
            f"**Experiment:** {experiment_name}  ",
            f"**Period:** {period_label}  ",
            f"**Generated:** {generated_at}  ",
            f"**Regulatory reference:** ICOBS 6B.2.51R",
            f"",
            "---",
            "",
            "## 1. Executive Summary",
            "",
        ]

        if not renewal_quotes:
            lines.append(
                "_No renewal quotes recorded for this experiment and period. "
                "If this is unexpected, verify that log_quote() was called with "
                "renewal_flag=True for renewal business._"
            )
        else:
            compliance_rate = len(compliant) / len(enbp_provided) if enbp_provided else None
            if compliance_rate is not None:
                compliance_str = f"{compliance_rate:.1%}"
                status = "COMPLIANT" if compliance_rate == 1.0 else (
                    "AT RISK" if len(breaches) > 0 else "INCOMPLETE RECORDS"
                )
            else:
                compliance_str = "N/A (no ENBP recorded)"
                status = "RECORDS INCOMPLETE"

            lines += [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Total quotes (all) | {len(quotes):,} |",
                f"| New business quotes | {len(nb_quotes):,} |",
                f"| Renewal quotes | {len(renewal_quotes):,} |",
                f"| Renewals with ENBP recorded | {len(enbp_provided):,} |",
                f"| ENBP-compliant renewals | {len(compliant):,} |",
                f"| ENBP breaches | {len(breaches):,} |",
                f"| Renewals without ENBP recorded | {len(no_enbp):,} |",
                f"| Overall compliance rate | {compliance_str} |",
                f"| Compliance status | **{status}** |",
                "",
            ]

        # By arm
        lines += [
            "## 2. ENBP Compliance by Model Arm",
            "",
            "| Arm | Renewal Quotes | ENBP Recorded | Compliant | Breaches | Not Recorded | Compliance Rate |",
            "|-----|---------------|---------------|-----------|----------|--------------|-----------------|",
        ]
        for arm in ("champion", "challenger"):
            s = arm_stats[arm]
            rate = f"{s['compliance_rate']:.1%}" if s["compliance_rate"] is not None else "N/A"
            lines.append(
                f"| {arm.capitalize()} | {s['renewals']:,} | {s['enbp_provided']:,} | "
                f"{s['compliant']:,} | {s['breaches']:,} | {s['no_enbp_recorded']:,} | {rate} |"
            )
        lines.append("")

        # Model versions
        lines += [
            "## 3. Model Versions Used in Pricing",
            "",
            "_Per ICOBS 6B.2.51R, firms must be able to demonstrate which model "
            "priced each renewal. This section provides that audit trail._",
            "",
            "| Model Version | Arm | Total Quotes |",
            "|--------------|-----|--------------|",
        ]
        for mv, info in sorted(model_versions.items()):
            lines.append(
                f"| `{mv}` | {info['arm'].capitalize()} | {info['count']:,} |"
            )
        lines.append("")

        # Breach detail (if any)
        if breaches:
            lines += [
                "## 4. ENBP Breach Detail",
                "",
                f"**{len(breaches)} renewal quotes where quoted_price > ENBP.**",
                "",
                "| Policy ID | Model Version | Arm | Quoted Price | ENBP | Excess | Timestamp |",
                "|-----------|--------------|-----|-------------|------|--------|-----------|",
            ]
            for q in breaches[:100]:  # cap at 100 rows
                excess = q["quoted_price"] - (q["enbp"] or 0)
                lines.append(
                    f"| `{q['policy_id']}` | `{q['model_version']}` | {q['arm'].capitalize()} | "
                    f"£{q['quoted_price']:.2f} | £{q['enbp']:.2f} | "
                    f"£{excess:.2f} | {q['timestamp'][:19]} |"
                )
            if len(breaches) > 100:
                lines.append(f"_... and {len(breaches) - 100} further breaches (truncated)._")
            lines.append("")
            lines += [
                "> **Action required:** ENBP breaches indicate quoted renewal prices exceeded "
                "the Equivalent New Business Price for identical risk profiles. "
                "Investigate root cause and remediate affected policies. "
                "Report to compliance function.",
                "",
            ]
        else:
            lines += [
                "## 4. ENBP Breach Detail",
                "",
                "_No ENBP breaches recorded in this period._",
                "",
            ]

        # Routing audit
        routing_by_model: dict[str, dict[str, int]] = {}
        for q in quotes:
            mv = q["model_version"]
            arm = q["arm"]
            if mv not in routing_by_model:
                routing_by_model[mv] = {"champion": 0, "challenger": 0}
            routing_by_model[mv][arm] = routing_by_model[mv].get(arm, 0) + 1

        lines += [
            "## 5. Routing Decision Audit",
            "",
            "Routing is deterministic: SHA-256(policy_id + experiment_name), "
            "last 8 hex characters modulo 100. Any routing decision can be "
            "recomputed independently from policy_id and experiment_name.",
            "",
            "| Model Version | Champion Quotes | Challenger Quotes |",
            "|--------------|-----------------|-------------------|",
        ]
        for mv, counts in sorted(routing_by_model.items()):
            lines.append(
                f"| `{mv}` | {counts.get('champion', 0):,} | {counts.get('challenger', 0):,} |"
            )
        lines.append("")

        # Attestation section
        lines += [
            "## 6. Attestation Statement",
            "",
            "_For completion by SMF holder or director. Delete inapplicable wording._",
            "",
            f"I, **{smf_holder}**, confirm that:",
            "",
            "1. The pricing records shown in this report were generated by the "
            "`insurance-deploy` audit framework, recording which model version "
            "priced each renewal quote in the period covered.",
            "",
            "2. Where ENBP was recorded, renewal prices were compared to ENBP at "
            "quote time and the results are shown above. [The firm] is satisfied "
            "that renewal prices did not exceed ENBP except where [insert any "
            "exceptions and justification].",
            "",
            "3. The routing methodology (hash-based, deterministic) does not "
            "introduce systematic discrimination against customers based on tenure "
            "or any other protected characteristic.",
            "",
            "4. This report constitutes the written record required under ICOBS "
            "6B.2.51R for the period specified.",
            "",
            f"**Signed:** ____________________________  ",
            f"**Name:** {smf_holder}  ",
            f"**Date:** ____________________________  ",
            f"**SMF reference:** ____________________________  ",
            "",
            "---",
            "",
            "_This report was generated by `insurance-deploy` v0.1.1. "
            "The library records ENBP values provided by the firm's pricing team; "
            "it does not calculate ENBP. ENBP calculation per ICOBS 6B methodology "
            "remains the firm's responsibility._",
        ]

        return "\n".join(lines)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _filter_by_period(
    quotes: list[dict],
    period_start: Optional[str],
    period_end: Optional[str],
) -> list[dict]:
    if not period_start and not period_end:
        return quotes
    result = []
    for q in quotes:
        ts = q["timestamp"][:10]  # YYYY-MM-DD
        if period_start and ts < period_start:
            continue
        if period_end and ts > period_end:
            continue
        result.append(q)
    return result


def _period_label(start: Optional[str], end: Optional[str]) -> str:
    if start and end:
        return f"{start} to {end}"
    if start:
        return f"from {start}"
    if end:
        return f"to {end}"
    return "All available data"
