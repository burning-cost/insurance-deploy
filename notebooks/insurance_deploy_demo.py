# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-deploy: Champion/Challenger Pricing Demo
# MAGIC
# MAGIC Full workflow on synthetic UK motor data:
# MAGIC 1. Model registry setup
# MAGIC 2. Experiment configuration and routing verification
# MAGIC 3. Quote/bind/claim data generation and logging
# MAGIC 4. KPI dashboard
# MAGIC 5. Bootstrap loss ratio comparison
# MAGIC 6. Power analysis ("how long until we can decide?")
# MAGIC 7. ENBP audit report (ICOBS 6B.2.51R)
# MAGIC
# MAGIC **Regulatory context:** ICOBS 6B.2.51R requires firms to maintain written
# MAGIC records demonstrating ENBP compliance for every renewal quote. FCA multi-firm
# MAGIC review (2023) found 83% of firms non-compliant. This notebook demonstrates
# MAGIC the technical infrastructure that closes this gap.

# COMMAND ----------

# MAGIC %pip install insurance-deploy

# COMMAND ----------

import random
import warnings
import tempfile
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from insurance_deploy import (
    ModelRegistry, Experiment, QuoteLogger,
    KPITracker, ModelComparison, ENBPAuditReport,
)

print("insurance-deploy loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synthetic Pricing Models
# MAGIC
# MAGIC Two minimal models representing champion (GLM-based factor table output)
# MAGIC and challenger (updated model with one additional feature).
# MAGIC In practice these would be CatBoost or sklearn objects loaded from MLflow.

# COMMAND ----------

class SyntheticMotorModel:
    """
    Minimal sklearn-compatible model for demonstration.
    In production: replace with your CatBoost/sklearn model.
    """
    def __init__(self, base_premium=400.0, age_effect=0.5, ncd_discount=0.05,
                 vehicle_value_effect=0.0, noise_level=30.0, seed=42):
        self.base_premium = base_premium
        self.age_effect = age_effect
        self.ncd_discount = ncd_discount
        self.vehicle_value_effect = vehicle_value_effect
        self.noise_level = noise_level
        self._rng = np.random.default_rng(seed)

    def predict(self, X):
        """X: list/array of (age, ncd_years, vehicle_value) tuples."""
        results = []
        for row in X:
            age, ncd, val = row[0], row[1], row[2]
            premium = (
                self.base_premium
                + self.age_effect * max(0, 25 - age)  # younger = more expensive
                - self.ncd_discount * self.base_premium * ncd
                + self.vehicle_value_effect * val / 1000
            )
            premium = max(150.0, premium)
            results.append(float(premium))
        return np.array(results)


# Champion: existing production model
champion_model = SyntheticMotorModel(
    base_premium=410.0, age_effect=0.6, ncd_discount=0.04,
    vehicle_value_effect=0.0, seed=42
)

# Challenger: updated model with vehicle value as additional feature
# Also slightly better calibrated (lower base premium, better NCD discount)
challenger_model = SyntheticMotorModel(
    base_premium=395.0, age_effect=0.55, ncd_discount=0.045,
    vehicle_value_effect=1.2, seed=99
)

print("Models defined")
print(f"Champion sample prediction (age=30, ncd=5, value=15000): "
      f"£{champion_model.predict([[30, 5, 15000]])[0]:.2f}")
print(f"Challenger sample prediction (age=30, ncd=5, value=15000): "
      f"£{challenger_model.predict([[30, 5, 15000]])[0]:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model Registry

# COMMAND ----------

tmp = tempfile.mkdtemp()
registry_path = Path(tmp) / "registry"

registry = ModelRegistry(registry_path)

champion_mv = registry.register(
    champion_model,
    name="motor",
    version="2.0",
    metadata={
        "training_date": "2024-01-01",
        "features": ["age", "ncd_years", "vehicle_value"],
        "holdout_gini": 0.41,
        "development_sample": "2018-2023",
        "validated_by": "pricing_team",
    }
)

challenger_mv = registry.register(
    challenger_model,
    name="motor",
    version="3.0",
    metadata={
        "training_date": "2024-07-01",
        "features": ["age", "ncd_years", "vehicle_value"],
        "holdout_gini": 0.44,
        "development_sample": "2019-2024",
        "improvement": "Added vehicle_value interaction; recalibrated on 2024 data",
        "validated_by": "pricing_team",
    }
)

print("Registered models:")
for mv in registry.list():
    print(f"  {mv}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Experiment Setup and Routing Verification

# COMMAND ----------

log_path = Path(tmp) / "quotes.db"
logger = QuoteLogger(log_path)

exp = Experiment(
    name="motor_v3_vs_v2_2024H2",
    champion=champion_mv,
    challenger=challenger_mv,
    challenger_pct=0.10,  # 10% challenger split
    mode="shadow",
)

print(f"Experiment: {exp}")
print()

# Verify routing determinism
test_pids = [f"POL-{i:06d}" for i in range(10)]
print("Routing verification (same result each call):")
for pid in test_pids[:5]:
    arm1 = exp.route(pid)
    arm2 = exp.route(pid)
    assert arm1 == arm2, f"Non-deterministic routing for {pid}"
    print(f"  {pid}: {arm1} (consistent)")

# Check approximate split
all_pids = [f"POL-{i:06d}" for i in range(10_000)]
challenger_count = sum(1 for p in all_pids if exp.route(p) == "challenger")
print(f"\nRouting split over 10,000 policies:")
print(f"  Champion: {10000 - challenger_count:,} ({(10000 - challenger_count)/100:.1f}%)")
print(f"  Challenger: {challenger_count:,} ({challenger_count/100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Synthetic Quote/Bind/Claims Data
# MAGIC
# MAGIC Simulate one year of motor pricing: 1,000 champion quotes, 100 challenger quotes
# MAGIC (10% split), with realistic bind rates and claim experience.

# COMMAND ----------

rng = random.Random(42)
np_rng = np.random.default_rng(42)

base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

# Motor risk parameters
def generate_risk():
    age = rng.randint(21, 70)
    ncd = rng.randint(0, 9)
    vehicle_value = rng.randint(5000, 40000)
    return {"age": age, "ncd": ncd, "vehicle_value": vehicle_value}

def calculate_enbp(risk, champion_model):
    """
    ENBP = champion model price for this risk profile.
    The library records this; your team calculates it.
    In practice: call your rating engine with risk profile stripped of tenure info.
    """
    inputs = [[risk["age"], risk["ncd"], risk["vehicle_value"]]]
    return champion_model.predict(inputs)[0]

quote_count = 0
bind_count = 0
claim_count = 0

print("Generating synthetic quote data...")

# Generate quotes
for i in range(1100):
    pid = f"POL-{i:06d}"
    ts = base_ts + timedelta(hours=i * 8)
    risk = generate_risk()
    inputs = [[risk["age"], risk["ncd"], risk["vehicle_value"]]]

    # Determine arm
    arm = exp.route(pid)

    # Champion always prices in shadow mode
    champion_price = float(champion_model.predict(inputs)[0])
    challenger_price = float(challenger_model.predict(inputs)[0])

    # ENBP for renewals (only relevant for renewals; use champion price as NB equivalent)
    is_renewal = i > 100  # first 100 are new business
    enbp = calculate_enbp(risk, champion_model) if is_renewal else None

    # Add small random noise to simulate quote-time variability
    champion_quoted = max(150.0, champion_price + rng.gauss(0, 5))

    # Log quote
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logger.log_quote(
            policy_id=pid,
            experiment_name=exp.name,
            arm=arm,
            model_version=champion_mv.version_id,  # champion prices in shadow
            quoted_price=champion_quoted,
            enbp=enbp,
            renewal_flag=is_renewal,
            exposure=1.0,
            timestamp=ts,
        )
    quote_count += 1

    # Bind rate: ~28% base, slightly higher for lower-priced quotes
    price_factor = max(0, (450 - champion_quoted) / 500)
    bind_prob = 0.25 + 0.10 * price_factor
    if rng.random() < bind_prob:
        bind_ts = ts + timedelta(hours=rng.uniform(0.5, 48))
        logger.log_bind(pid, champion_quoted, bind_ts)
        bind_count += 1

        # Claim frequency: ~8% for champion cohort, ~7.5% for challenger cohort
        freq = 0.08 if arm == "champion" else 0.075
        if rng.random() < freq:
            claim_date = ts.date() + timedelta(days=rng.randint(30, 300))
            # Log at 3-month development (FNOL)
            fnol_amount = max(0, rng.gauss(600, 300))
            logger.log_claim(pid, claim_date, fnol_amount, development_month=3)
            # Log at 12-month development (developed)
            developed_amount = max(fnol_amount, fnol_amount + rng.gauss(600, 400))
            logger.log_claim(pid, claim_date, developed_amount, development_month=12)
            claim_count += 1

print(f"Data generation complete:")
print(f"  Total quotes: {quote_count:,}")
print(f"  Binds: {bind_count:,} ({bind_count/quote_count:.1%} conversion)")
print(f"  Claims (policies): {claim_count:,} ({claim_count/bind_count:.1%} frequency)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. KPI Dashboard

# COMMAND ----------

tracker = KPITracker(logger)

print("=" * 60)
print(f"EXPERIMENT: {exp.name}")
print("=" * 60)

# Quote volume
vol = tracker.quote_volume(exp.name)
print("\nQUOTE VOLUME & PRICE DISTRIBUTION:")
for arm in ("champion", "challenger"):
    v = vol[arm]
    print(f"  {arm.capitalize():12s}: n={v['n']:4d} | "
          f"mean=£{v['mean_price']:.0f} | "
          f"median=£{v['median_price']:.0f} | "
          f"p25=£{v['p25_price']:.0f} | "
          f"p75=£{v['p75_price']:.0f}")

# Hit rate
print("\nHIT RATE (CONVERSION):")
hr = tracker.hit_rate(exp.name)
for arm in ("champion", "challenger"):
    h = hr[arm]
    print(f"  {arm.capitalize():12s}: {h['bound']:3d}/{h['quoted']:4d} = {h['hit_rate']:.1%}")

# GWP
print("\nGROSS WRITTEN PREMIUM:")
gwp = tracker.gwp(exp.name)
for arm in ("champion", "challenger"):
    g = gwp[arm]
    print(f"  {arm.capitalize():12s}: {g['bound_policies']:3d} policies | "
          f"total GWP £{g['total_gwp']:,.0f} | "
          f"mean £{g['mean_gwp']:.0f}")

# Loss ratio (12-month development)
print("\nLOSS RATIO (12-month development):")
lr = tracker.loss_ratio(exp.name, development_months=12)
for arm in ("champion", "challenger"):
    l = lr[arm]
    print(f"  {arm.capitalize():12s}: LR={l['loss_ratio']:.1%} | "
          f"earned £{l['earned_premium']:,.0f} | "
          f"incurred £{l['incurred_claims']:,.0f} | "
          f"n={l['policy_count']:3d} policies")

# ENBP compliance
print("\nENBP COMPLIANCE (renewals only):")
comp_stats = tracker.enbp_compliance(exp.name)
for arm in ("champion", "challenger"):
    c = comp_stats[arm]
    rate = f"{c['compliance_rate']:.1%}" if not (c['compliance_rate'] != c['compliance_rate']) else "N/A"
    print(f"  {arm.capitalize():12s}: {c['renewal_quotes']:3d} renewals | "
          f"{c['compliant']:3d} compliant | {c['breaches']:2d} breaches | "
          f"rate: {rate}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Statistical Comparison

# COMMAND ----------

comp = ModelComparison(tracker)

print("BOOTSTRAP LOSS RATIO TEST (12-month development, 5,000 iterations)")
print("-" * 60)
lr_result = comp.bootstrap_lr_test(
    exp.name,
    n_bootstrap=5_000,
    development_months=12,
    seed=42,
)
print(lr_result.summary())

print("\n" + "=" * 60)
print("HIT RATE TEST (two-proportion z-test)")
print("-" * 60)
hr_result = comp.hit_rate_test(exp.name)
print(hr_result.summary())

print("\n" + "=" * 60)
print("CLAIM FREQUENCY TEST (Poisson, 12-month development)")
print("-" * 60)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    freq_result = comp.frequency_test(exp.name, development_months=12)
print(freq_result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Power Analysis — How Long Until We Can Decide?
# MAGIC
# MAGIC The most common mistake in champion/challenger is expecting a promotion decision
# MAGIC after 3 months on a 10% challenger split. This shows you the realistic timeline.

# COMMAND ----------

pa = tracker.power_analysis(
    exp.name,
    target_delta_lr=0.03,   # detect 3 percentage-point LR improvement
    target_delta_hr=0.02,   # detect 2 percentage-point hit rate improvement
    alpha=0.05,
    power=0.80,
)

print("POWER ANALYSIS RESULTS")
print("=" * 60)
print(f"\nCurrent state:")
print(f"  Champion quoted policies: {pa['current_n_champion']:,}")
print(f"  Challenger quoted policies: {pa['current_n_challenger']:,}")
print(f"  Estimated months elapsed: {pa['months_elapsed']:.1f}")
print(f"  Monthly challenger rate: {pa['monthly_rate_challenger']:.0f} policies/month")

print(f"\nHit rate significance (target delta: {pa['target_delta_hr']:.0%}):")
print(f"  Required n per arm: {pa['hr_required_n_per_arm']:,}")
print(f"  Estimated months to significance: {pa['hr_months_to_significance']:.0f} months")

print(f"\nLoss ratio significance (target delta: {pa['target_delta_lr']:.0%}):")
print(f"  Required n per arm (bind): {pa['lr_required_n_per_arm']:,}")
print(f"  Months to reach bind n: {pa['lr_months_to_bind']:.0f}")
print(f"  Plus 12-month development: 12 months")
print(f"  Total: {pa['lr_total_months_with_development']:.0f} months from experiment start")

print(f"\nNotes:")
for note in pa["notes"]:
    print(f"  - {note}")

print()
print("Key message: loss ratio significance on a 10% challenger split takes")
print("approximately 2-3 years for a mid-size motor book. Set expectations accordingly.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. ENBP Audit Report (ICOBS 6B.2.51R)
# MAGIC
# MAGIC This is the document your SMF holder signs for the annual ENBP attestation.

# COMMAND ----------

reporter = ENBPAuditReport(logger)

audit_md = reporter.generate(
    experiment_name=exp.name,
    period_start="2024-01-01",
    period_end="2024-12-31",
    firm_name="Example Insurance Ltd",
    smf_holder="[Chief Actuary / SMF7]",
)

# Display in notebook
displayHTML(f"<pre>{audit_md[:3000]}...</pre>")

# In production: write to file or Databricks table
audit_path = Path(tmp) / "enbp_audit_report.md"
audit_path.write_text(audit_md)
print(f"\nFull report written to: {audit_path}")
print(f"Report length: {len(audit_md.splitlines())} lines")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Routing Audit — Verify Any Decision
# MAGIC
# MAGIC The SHA-256 routing is deterministic. You can verify any routing decision
# MAGIC from first principles without querying the database.

# COMMAND ----------

import hashlib

def verify_routing(policy_id: str, experiment_name: str, challenger_pct: float = 0.10) -> str:
    """Recompute routing decision independently of the logger."""
    key = (policy_id + experiment_name).encode()
    digest = hashlib.sha256(key).hexdigest()
    slot = int(digest[-8:], 16) % 100
    threshold = int(challenger_pct * 100)
    return "challenger" if slot < threshold else "champion"

# Verify 10 known quotes from the logger
quotes = logger.query_quotes(exp.name)[:10]
print("Routing audit: verifying 10 logged routing decisions")
print(f"{'Policy ID':15s} {'Logged Arm':12s} {'Recomputed':12s} {'Match':6s}")
print("-" * 50)
all_match = True
for q in quotes:
    recomputed = verify_routing(q["policy_id"], exp.name)
    match = q["arm"] == recomputed
    all_match = all_match and match
    status = "OK" if match else "MISMATCH"
    print(f"{q['policy_id']:15s} {q['arm']:12s} {recomputed:12s} {status:6s}")

print(f"\nAll routing decisions verified: {all_match}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC
# MAGIC - **Model registry**: register and retrieve versioned model objects with hash verification
# MAGIC - **Experiment setup**: shadow mode, 10% challenger split, deterministic routing
# MAGIC - **Audit logging**: per-quote ENBP compliance flagging, append-only SQLite
# MAGIC - **KPI tracking**: hit rate, GWP, loss ratio, claim frequency — all segmented by arm
# MAGIC - **Statistical tests**: bootstrap LR, z-test on hit rate, Poisson frequency GLM
# MAGIC - **Power analysis**: realistic timeline to significance (spoiler: it's longer than you think)
# MAGIC - **ENBP audit report**: ICOBS 6B.2.51R compliance document ready for SMF attestation
# MAGIC
# MAGIC **Next steps for production deployment:**
# MAGIC 1. Replace `SyntheticMotorModel` with your real model (CatBoost, sklearn, etc.)
# MAGIC 2. Hook `logger.log_quote()` into your live quote handler
# MAGIC 3. Set up daily scheduled notebook to refresh KPI dashboard
# MAGIC 4. Schedule monthly ENBP audit report with alert on any breaches
# MAGIC 5. Run bootstrap LR test quarterly — expect inconclusive results until 12+ months development
