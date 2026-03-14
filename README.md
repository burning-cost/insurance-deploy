# insurance-deploy
[![Tests](https://github.com/burning-cost/insurance-deploy/actions/workflows/tests.yml/badge.svg)](https://github.com/burning-cost/insurance-deploy/actions/workflows/tests.yml)

Champion/challenger pricing framework for UK insurance — model registry, quote routing, ENBP audit logging, and statistical promotion tests.

---

## The problem

You've built a better pricing model. CatBoost instead of GLM, or an updated GLM with two years more data and a rebuilt rating factor for NCB. The model validates well in holdout. Your actuarial team wants to deploy it.

The problem is everything after model training.

How do you run the challenger alongside the champion without disrupting live pricing? How do you log which model priced each quote — per-quote, per-policy, permanently — so you can run the FCA-required ENBP audit? How do you know when you have enough data to make a statistically credible promotion decision, rather than guessing after three months on a sample too small to tell you anything?

Every UK pricing team faces this. Most solve it with ad-hoc scripts, spreadsheet logs, and informal sign-off. This library provides the infrastructure.

**Blog post:** [Your Champion/Challenger Test Has No Audit Trail](https://burning-cost.github.io/2026/03/09/your-champion-challenger-test-has-no-audit-trail/) — worked example of the full workflow, the routing determinism guarantee, and why loss ratio significance takes 18 months.

---

## Regulatory context

**ICOBS 6B.2.51R** (the ENBP rules, effective January 2022) requires firms to maintain written records demonstrating that renewal prices do not exceed the Equivalent New Business Price for identical risk profiles.

FCA multi-firm review (2023): 83% of firms were non-compliant with record-keeping requirements. Most lacked records granular enough for the SMF holder to sign the annual attestation.

When a pricing model changes mid-year, you must be able to demonstrate which model priced each renewal and that the model change did not introduce tenure discrimination. That requires a per-quote model version log. This library is that log.

**FCA Consumer Duty (PRIN 2A)** creates a risk for live A/B pricing. Charging two customers of identical profile differently simultaneously could be challenged as inconsistent with fair value obligations. Shadow mode (the default) eliminates this risk — challenger scores in parallel but the customer always sees the champion price.

---

## What this library does

Five modules:

| Module | Contents |
|--------|----------|
| `insurance_deploy.registry` | `ModelRegistry` — append-only version-tagged model store with hash verification |
| `insurance_deploy.experiment` | `Experiment` — deterministic hash-based routing, shadow and live modes |
| `insurance_deploy.logger` | `QuoteLogger` — append-only SQLite audit log with ENBP compliance flagging |
| `insurance_deploy.kpi` | `KPITracker` — hit rate, GWP, loss ratio, frequency, power analysis |
| `insurance_deploy.comparison` | `ModelComparison` — bootstrap LR test, z-test on hit rate, Poisson frequency test |
| `insurance_deploy.audit` | `ENBPAuditReport` — ICOBS 6B.2.51R compliance report in Markdown |

---

## Install

```bash
uv add insurance-deploy
# or
pip install insurance-deploy
```

Dependencies: NumPy, SciPy, Pandas, joblib.

---

## Quick start

### Register models

```python
from insurance_deploy import ModelRegistry

registry = ModelRegistry("./registry")

# Register current champion
champion_mv = registry.register(
    champion_model,           # any sklearn-compatible object with .predict()
    name="motor",
    version="2.0",
    metadata={
        "training_date": "2024-01-01",
        "features": ["age", "ncd", "postcode_band"],
        "holdout_gini": 0.42,
    }
)

# Register challenger
challenger_mv = registry.register(
    challenger_model,
    name="motor",
    version="3.0",
    metadata={
        "training_date": "2024-07-01",
        "features": ["age", "ncd", "postcode_band", "vehicle_value"],
        "holdout_gini": 0.45,
    }
)
```

### Set up the experiment

```python
from insurance_deploy import Experiment

exp = Experiment(
    name="motor_v3_vs_v2",
    champion=champion_mv,
    challenger=challenger_mv,
    challenger_pct=0.10,  # 10% of policies to challenger
    mode="shadow",        # Default. Challenger scores but does not price.
)
```

### Route quotes and log

```python
from insurance_deploy import QuoteLogger

logger = QuoteLogger("./quotes.db")

# In your quote handler:
def handle_quote(policy_id, inputs, renewal_flag=False, enbp=None):
    arm = exp.route(policy_id)

    # Champion always prices in shadow mode
    champion_price = champion_mv.predict([inputs])[0]

    # Challenger scores in shadow — output logged, not shown to customer
    challenger_price = challenger_mv.predict([inputs])[0]

    # Log champion quote (the one the customer sees)
    logger.log_quote(
        policy_id=policy_id,
        experiment_name=exp.name,
        arm=arm,
        model_version=champion_mv.version_id,  # champion always prices
        quoted_price=champion_price,
        enbp=enbp,           # Provide for renewals — you calculate this, not us
        renewal_flag=renewal_flag,
    )

    return champion_price

# When policy binds:
logger.log_bind("POL-12345", bound_price=425.0)

# When claim is reported (log at each development stage):
from datetime import date
logger.log_claim("POL-12345", claim_date=date(2024, 8, 1),
                 claim_amount=1200.0, development_month=3)
# Update at 12 months:
logger.log_claim("POL-12345", claim_date=date(2024, 8, 1),
                 claim_amount=1450.0, development_month=12)
```

### Track KPIs

```python
from insurance_deploy import KPITracker

tracker = KPITracker(logger)

# Immediately available
print(tracker.hit_rate("motor_v3_vs_v2"))
# {'champion': {'quoted': 900, 'bound': 270, 'hit_rate': 0.30},
#  'challenger': {'quoted': 100, 'bound': 28, 'hit_rate': 0.28}}

print(tracker.gwp("motor_v3_vs_v2"))
# {'champion': {'bound_policies': 270, 'total_gwp': 108000.0, 'mean_gwp': 400.0},
#  'challenger': {'bound_policies': 28, 'total_gwp': 11480.0, 'mean_gwp': 410.0}}

# After 12 months development
lr = tracker.loss_ratio("motor_v3_vs_v2", development_months=12)
print(lr)
# {'champion': {'loss_ratio': 0.64, 'policy_count': 270, ...},
#  'challenger': {'loss_ratio': 0.61, 'policy_count': 28, ...}}

# Power analysis: how long until we can decide?
pa = tracker.power_analysis("motor_v3_vs_v2", target_delta_lr=0.03)
print(f"Months to LR significance (incl. 12m development): "
      f"{pa['lr_total_months_with_development']:.0f}")
# Months to LR significance (incl. 12m development): 28
```

### Statistical comparison

```python
from insurance_deploy import ModelComparison

comp = ModelComparison(tracker)

# Bootstrap loss ratio test (requires 12m+ development)
result = comp.bootstrap_lr_test("motor_v3_vs_v2", n_bootstrap=10_000, seed=42)
print(result.summary())
# Test: bootstrap_lr_test | Experiment: motor_v3_vs_v2
# Champion estimate: 0.6402 (n=270)
# Challenger estimate: 0.6118 (n=28)
# Difference (challenger - champion): -0.0284
# 95% CI: [-0.0751, 0.0183]
# p-value: 0.2341
#
# Conclusion: INSUFFICIENT_EVIDENCE
# Recommendation: No statistically significant difference detected in loss_ratio
# (p=0.234 >= 0.05). Continue experiment. Consider running power_analysis()
# to estimate time to significance.

# Hit rate test (available earlier)
hr_result = comp.hit_rate_test("motor_v3_vs_v2")
```

### ENBP audit report

```python
from insurance_deploy import ENBPAuditReport

reporter = ENBPAuditReport(logger)
md = reporter.generate(
    "motor_v3_vs_v2",
    period_start="2024-01-01",
    period_end="2024-12-31",
    firm_name="Acme Insurance Ltd",
    smf_holder="Jane Smith",
)
print(md)  # Markdown: paste into attestation pack or Databricks notebook
```

---

## Shadow mode vs live mode

Shadow mode is the default and is right for most use cases.

**Shadow mode:** champion prices every quote. Challenger runs on identical inputs, output is logged but not shown to the customer. Zero fair value regulatory risk. Model quality comparison is clean — no adverse selection confound.

**Live mode:** challenger prices its routed fraction of policies (10% by default). Enables market response testing (does challenger's different pricing affect conversion?). Raises FCA Consumer Duty fair value questions. Also introduces adverse selection bias: if challenger prices differently, the bound cohorts will have different risk profiles, making loss ratio comparison harder to interpret.

```python
# Live mode — get legal sign-off first
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # suppress the FCA warning if you've taken legal advice
    exp = Experiment(
        name="live_test",
        champion=champion_mv,
        challenger=challenger_mv,
        mode="live",
    )
```

The library is opinionated here. The warning is intentional. Suppress it when you've done the legal groundwork, not before.

---

## Routing determinism

Routing uses SHA-256(policy_id + experiment_name), last 8 hex characters as integer, modulo 100. If result < challenger_pct * 100, route to challenger.

This is deterministic and stateless. Given a policy_id and experiment name, the routing decision is always the same and can be recomputed at any point independently. No database of assignments required. Any assignment can be verified from first principles.

Assignment is by policy, not by quote. A policy routed to challenger on first quote will always be routed to challenger within this experiment. This is required for ENBP audit integrity.

---

## Why loss ratio significance takes 18 months

At 10% challenger split with 3,000 bound policies/month total:
- Challenger receives ~300 policies/month
- Hit rate significance (2pp delta, 80% power): ~5 months
- Claim frequency significance (0.5pp delta): ~10 months
- **Developed loss ratio significance (3pp delta, 12-month development): ~17 months from first quote, total ~29 months from experiment start**

This is not a limitation of the library. It is physics. LR has a 12-36 month reward tail. Any framework claiming to optimise on LR signal faster than this using bandits or similar methods is either using a proxy metric (hit rate) or lying.

The `power_analysis()` method makes this timeline explicit. Run it before starting an experiment so your stakeholders have realistic expectations.

---

## Radar wrapper pattern

Most UK personal lines insurers deploy rates via WTW Radar Live. The library integrates as a governance layer around Radar:

```python
import requests
from insurance_deploy import Experiment, QuoteLogger

def get_quote(policy_id, risk_dict, renewal_flag=False, enbp=None):
    # Champion = Radar Live (existing production system)
    radar_response = requests.post(RADAR_LIVE_URL, json=risk_dict)
    champion_price = radar_response.json()["premium"]

    # Challenger = Python model (your new model)
    arm = exp.route(policy_id)
    challenger_price = challenger_mv.predict([risk_dict])[0]

    # Log the quote (champion prices; challenger is shadow)
    logger.log_quote(
        policy_id=policy_id,
        experiment_name=exp.name,
        arm=arm,
        model_version=champion_mv.version_id,
        quoted_price=champion_price,
        enbp=enbp,
        renewal_flag=renewal_flag,
    )

    return champion_price  # customer always sees champion price
```

No Radar infrastructure changes required. The library handles the governance layer.

---

## ENBP: what the library does and doesn't do

The library records ENBP. It does not calculate it.

ENBP calculation per ICOBS 6B is your pricing team's responsibility. You pass the ENBP value to `log_quote(enbp=...)`. The library records the value, flags whether `quoted_price <= enbp`, and includes this in the audit report.

If your ENBP calculation is wrong, the log is wrong — but that is upstream of this library's scope. The separation is intentional.

---

## Databricks companion notebook

`notebooks/benchmark.py` demonstrates the full workflow on synthetic data:
- Model registry setup
- Experiment configuration and routing verification
- Quote/bind/claim data generation and logging
- KPI dashboard
- Bootstrap LR test
- ENBP audit report generation

Run as a Databricks Python notebook. Requires `pip install insurance-deploy`.

---

## Scope

This library handles: model version registry, champion/challenger routing, audit logging, KPI computation, statistical tests, ENBP compliance reports.

It does not handle: model training, rate optimisation (see `insurance-optimise`), model drift monitoring (see `insurance-monitor`), causal effect estimation (see `insurance-causal-policy`), or real-time API infrastructure.

---

## Performance

This library is a deployment and governance framework, not a predictive model — so there is no vs-baseline accuracy comparison. The benchmark notebook demonstrates the full workflow on synthetic UK motor data (10,000 policies) rather than testing predictive lift.

What the notebook validates:

**Routing determinism:** SHA-256 hash-based routing produces the expected 20% challenger allocation (configurable) within 0.5 percentage points across 10,000 policies. The same policy_id always routes to the same arm within a named experiment, which is required for ENBP audit integrity.

**Statistical test calibration:** the three promotion tests — hit rate z-test, Poisson frequency test, and bootstrap loss ratio test (10,000 iterations, policy-level resampling) — are demonstrated on a champion (GLM-logistic) vs challenger (CatBoost) pair with known Gini separation. Under shadow mode with identical prices for both arms, the bootstrap LR test is expected to return `INSUFFICIENT_EVIDENCE` at typical experiment volumes, reflecting the true statistical difficulty of detecting a 3pp loss ratio improvement.

**Power analysis:** at 20% challenger allocation and 10,000 total policies, the power analysis is expected to report approximately 18–24 months to loss ratio significance (including 12-month claim development). This matches the theoretical calculation and is the honest answer about champion/challenger timelines.

**ENBP compliance rate:** with a 2% intentional breach rate injected into the synthetic data, the ENBP audit report is expected to flag approximately 2% of renewal quotes as non-compliant, confirming the logger records and surfaces breaches correctly.

Run `notebooks/benchmark.py` on Databricks to reproduce.

---

## Other Burning Cost libraries

**Model building**

| Library | Description |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs using SHAP |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection via CANN and NID scores |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation respecting IBNR structure |

**Uncertainty quantification**

| Library | Description |
|---------|-------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for Tweedie models |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data segments |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility weighting |

**Deployment and optimisation**

| Library | Description |
|---------|-------------|
| [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) | Causal price elasticity via Double Machine Learning |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation with FCA PS21/5 compliance |

**Governance**

| Library | Description |
|---------|-------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing for UK insurance models |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 model validation reports |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model monitoring: PSI, A/E ratios, Gini drift test |

[All libraries and blog posts →](https://burning-cost.github.io)

---


## Related Libraries

| Library | What it does |
|---------|-------------|
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 model validation — deploy requires governance sign-off; this library produces the documentation |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Post-deployment model monitoring — PSI, A/E ratios, and Gini drift tracking for deployed models |
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained rate change optimisation — determines the rates the deployment pipeline serves |

## Licence

MIT. Part of the [Burning Cost](https://github.com/burning-cost) insurance pricing toolkit.
