"""
insurance-deploy: champion/challenger pricing framework for UK insurance.

Five modules, one purpose: deploy pricing models safely with audit trail,
ENBP compliance logging, and evidence-based promotion decisions.

Regulatory context
------------------
ICOBS 6B.2.51R requires firms to keep written records demonstrating ENBP
compliance for every renewal quote. FCA multi-firm review (2023) found 83% of
firms non-compliant. This library provides the technical infrastructure for
that record-keeping.

Shadow mode is the default. Live A/B pricing raises FCA Consumer Duty (PRIN
2A) fair value concerns — see README for discussion. Most teams should use
shadow mode for model comparison and only enable live routing after taking
legal advice.

Quick start
-----------
>>> from insurance_deploy import ModelRegistry, Experiment, QuoteLogger
>>> from insurance_deploy import KPITracker, ModelComparison
>>>
>>> registry = ModelRegistry("./registry")
>>> mv = registry.register(my_model, name="motor_v3", version="1.0",
...                        metadata={"training_date": "2024-01-01"})
>>> exp = Experiment("v3_vs_v2", champion=old_mv, challenger=mv)
>>> logger = QuoteLogger("./quotes.db")
>>> logger.log_quote("POL-001", exp.name, mv.version_id,
...                  inputs={}, quoted_price=450.0)
"""

from .registry import ModelRegistry, ModelVersion
from .experiment import Experiment
from .logger import QuoteLogger
from .kpi import KPITracker
from .comparison import ModelComparison, ComparisonResult
from .audit import ENBPAuditReport

__version__ = "0.1.1"
__all__ = [
    "ModelRegistry",
    "ModelVersion",
    "Experiment",
    "QuoteLogger",
    "KPITracker",
    "ModelComparison",
    "ComparisonResult",
    "ENBPAuditReport",
]
