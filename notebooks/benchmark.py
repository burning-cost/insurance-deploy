# Databricks notebook source
# This file is in Databricks notebook format:
#   # COMMAND ----------  separates cells
#   # MAGIC %md           starts a markdown cell line

# COMMAND ----------

# MAGIC %md
# MAGIC # insurance-deploy: Champion/Challenger Workflow Demo
# MAGIC
# MAGIC **Library:** `insurance-deploy` v0.1.1 — Champion/challenger pricing framework
# MAGIC
# MAGIC **What this notebook demonstrates:**
# MAGIC - Registering model versions with hash verification
# MAGIC - Shadow-mode experiment routing (challenger scores without affecting customer price)
# MAGIC - Append-only SQLite quote audit log with ENBP compliance flagging
# MAGIC - KPI tracking: hit rate, GWP, loss ratio, frequency
# MAGIC - Statistical promotion tests: bootstrap LR, z-test on hit rate, Poisson frequency test
# MAGIC - ICOBS 6B.2.51R ENBP compliance audit report
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## The problem this library solves
# MAGIC
# MAGIC Most UK pricing teams run champion/challenger informally: the new model scores
# MAGIC a spreadsheet extract, someone eyeballs the means, and promotion happens via
# MAGIC email. There is no audit trail. When the FCA asks which model priced a
# MAGIC specific renewal, the answer is usually a folder name and a prayer.
# MAGIC
# MAGIC `insurance-deploy` makes the infrastructure explicit: a version-tagged model
# MAGIC store, deterministic hash-based routing, an append-only quote log, and
# MAGIC statistical tests that tell you when the data actually supports promotion.
# MAGIC The ENBP audit report directly addresses ICOBS 6B.2.51R — the rule that
# MAGIC 83% of firms failed in the FCA's 2023 multi-firm review.
# MAGIC
# MAGIC This demo walks through the full workflow: train two models, register them,
# MAGIC run a shadow experiment, log quotes and outcomes, compute KPIs, run
# MAGIC statistical tests, and generate the compliance report.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# Install insurance-deploy and dependencies
%pip install git+https://github.com/burning-cost/insurance-deploy.git
%pip install scikit-learn catboost pandas numpy scipy matplotlib

# COMMAND ----------

# Restart Python after pip installs (required on Databricks)
dbutils.library.restartPython()

# COMMAND ----------

import os
import tempfile
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from insurance_deploy.registry import ModelRegistry
from insurance_deploy.experiment import Experiment
from insurance_deploy.logger import QuoteLogger
from insurance_deploy.kpi import KPITracker
from insurance_deploy.comparison import ModelComparison
from insurance_deploy.audit import ENBPAuditReport

# Suppress noisy warnings during demo
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"Demo run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data: Synthetic UK Motor Portfolio
# MAGIC
# MAGIC We generate a synthetic portfolio of 10,000 motor policies. The data
# MAGIC generating process is simple but realistic enough to produce two models
# MAGIC with meaningfully different characteristics:
# MAGIC
# MAGIC - **Champion (GLM-style logistic):** fit on one-hot encoded factors, no interactions.
# MAGIC   Solid, interpretable, file-able. The model most UK pricing teams actually have
# MAGIC   in production today.
# MAGIC - **Challenger (CatBoost gradient boosting):** fit on raw categoricals with
# MAGIC   native handling. Typically captures interactions the GLM misses; costs more
# MAGIC   to explain to a Lloyd's signing actuary.
# MAGIC
# MAGIC Each policy has: vehicle class, driver age band, NCD band, region, vehicle age,
# MAGIC a true underlying loss probability (known to us, not to the models), a
# MAGIC simulated claim outcome, a quoted price from the incumbent rating engine,
# MAGIC and an ENBP (relevant for renewal policies).
# MAGIC
# MAGIC The hit rate (bind probability) decreases as price increases — a plausible
# MAGIC demand curve. We use this to simulate which quotes converted to policies.

# COMMAND ----------

rng = np.random.default_rng(42)
N = 10_000

# Rating factors
vehicle_classes = ["A", "B", "C", "D", "E"]
age_bands       = ["17-25", "26-35", "36-50", "51-65", "66+"]
ncd_bands       = ["0", "1", "2", "3", "4", "5+"]
regions         = ["London", "SE", "Midlands", "North", "Scotland", "Wales"]

vc  = rng.choice(vehicle_classes, N)
ab  = rng.choice(age_bands, N)
ncd = rng.choice(ncd_bands, N)
reg = rng.choice(regions, N)
veh_age = rng.integers(0, 15, N).astype(float)

# True loss probability (known DGP, not available to models)
vc_effect  = {"A": 0.04, "B": 0.06, "C": 0.08, "D": 0.10, "E": 0.14}
ab_effect  = {"17-25": 0.06, "26-35": 0.04, "36-50": 0.03, "51-65": 0.035, "66+": 0.05}
ncd_effect = {"0": 0.05, "1": 0.04, "2": 0.03, "3": 0.025, "4": 0.02, "5+": 0.015}
reg_effect = {"London": 0.05, "SE": 0.04, "Midlands": 0.035, "North": 0.03,
              "Scotland": 0.025, "Wales": 0.025}

true_prob = np.array([
    vc_effect[v] + ab_effect[a] + ncd_effect[n] + reg_effect[r] + 0.002 * va
    for v, a, n, r, va in zip(vc, ab, ncd, reg, veh_age)
])
true_prob = np.clip(true_prob, 0.01, 0.40)

# Simulate claims (Bernoulli)
claims = rng.binomial(1, true_prob)

# Incumbent pricing: GLM-ish with some miscalibration in high-risk segments
# The challenger should pick this up
base_price = 350
incumbent_rate = (
    np.array([vc_effect[v] for v in vc]) * 2200
    + np.array([ab_effect[a] for a in ab]) * 1500
    + np.array([ncd_effect[n] for n in ncd]) * 1800
    + 0.005 * veh_age * base_price
    + rng.normal(0, 15, N)
)
incumbent_rate = np.clip(incumbent_rate, 200, 1800)

# Renewal flag: ~40% of portfolio is renewal
renewal_flag = rng.binomial(1, 0.40, N).astype(bool)

# ENBP: new business equivalent price. For renewals, ENBP = current_quoted * NB_factor.
# We deliberately inject a small number of ENBP breaches (~2%) for demo realism.
enbp_factor = rng.uniform(0.92, 1.05, N)
enbp_values = incumbent_rate * enbp_factor
enbp_breach = rng.binomial(1, 0.02, N).astype(bool)  # 2% intentional breaches for demo
enbp_values_adj = np.where(enbp_breach, incumbent_rate * 0.97, enbp_values)  # price > enbp

# Hit rate: demand curve — lower price = higher bind probability
hit_prob = 1 / (1 + np.exp(0.004 * (incumbent_rate - 450)))
hit_prob = np.clip(hit_prob, 0.05, 0.70)
bound = rng.binomial(1, hit_prob).astype(bool)

df = pd.DataFrame({
    "policy_id":     [f"POL-{i:05d}" for i in range(N)],
    "vehicle_class": vc,
    "age_band":      ab,
    "ncd_band":      ncd,
    "region":        reg,
    "vehicle_age":   veh_age,
    "true_prob":     true_prob,
    "claim":         claims,
    "quoted_price":  incumbent_rate.round(2),
    "enbp":          enbp_values_adj.round(2),
    "renewal_flag":  renewal_flag,
    "bound":         bound,
})

print(f"Portfolio: {len(df):,} policies")
print(f"  Claims:      {df['claim'].sum():,} ({df['claim'].mean():.1%})")
print(f"  Bound:       {df['bound'].sum():,} ({df['bound'].mean():.1%})")
print(f"  Renewals:    {df['renewal_flag'].sum():,} ({df['renewal_flag'].mean():.1%})")
print(f"  ENBP breaches introduced: {enbp_breach.sum():,}")
print(f"\nPrice distribution (£):")
print(df["quoted_price"].describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train champion and challenger models
# MAGIC
# MAGIC Both models predict claim probability. We split 70/30 train/test.
# MAGIC In a real deployment you would also have a calibration set. Here we
# MAGIC keep it simple — the point is the registry, logging, and testing
# MAGIC infrastructure, not squeezing Gini points.

# COMMAND ----------

from sklearn.model_selection import train_test_split

CAT_FEATURES = ["vehicle_class", "age_band", "ncd_band", "region"]
NUM_FEATURES = ["vehicle_age"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "claim"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# COMMAND ----------

# Champion: logistic GLM with one-hot encoding
# This represents the standard approach: a production GLM scoring via a
# rating engine pipeline.

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ("ohe",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ("scal", StandardScaler(), NUM_FEATURES),
])

champion_model = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(max_iter=500, C=1.0, random_state=42)),
])

champion_model.fit(X_train, y_train)
champ_prob_test = champion_model.predict_proba(X_test)[:, 1]

print(f"Champion (GLM-Logistic) fitted.")
print(f"  Mean predicted probability: {champ_prob_test.mean():.4f}")
print(f"  True mean frequency (test): {y_test.mean():.4f}")

# COMMAND ----------

# Challenger: CatBoost gradient boosting
# Handles categoricals natively; often picks up interactions the GLM misses.
# Higher complexity, requires more justification for filing.

try:
    from catboost import CatBoostClassifier
    challenger_model = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        cat_features=CAT_FEATURES,
        loss_function="Logloss",
        random_seed=42,
        verbose=0,
    )
    challenger_model.fit(X_train, y_train)
    chall_prob_test = challenger_model.predict_proba(X_test)[:, 1]
    print(f"Challenger (CatBoost) fitted.")
    print(f"  Mean predicted probability: {chall_prob_test.mean():.4f}")
except ImportError:
    # Fall back to GradientBoostingClassifier if CatBoost not available
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder

    ohe_fallback = ColumnTransformer([
        ("ohe",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("pass", "passthrough", NUM_FEATURES),
    ])
    challenger_model = Pipeline([
        ("prep", ohe_fallback),
        ("clf",  GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                             learning_rate=0.05, random_state=42)),
    ])
    challenger_model.fit(X_train, y_train)
    chall_prob_test = challenger_model.predict_proba(X_test)[:, 1]
    print(f"Challenger (GradientBoosting fallback) fitted.")
    print(f"  Mean predicted probability: {chall_prob_test.mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quick model comparison: Gini on holdout
# MAGIC
# MAGIC Before we touch the registry, let's verify the challenger actually adds
# MAGIC something. We use the Gini coefficient on the test set. A higher Gini means
# MAGIC the model discriminates better between claims and non-claims.

# COMMAND ----------

def gini_coefficient(y_true, y_pred):
    """Normalised Gini, computed from the Lorenz curve. Higher is better."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order   = np.argsort(y_pred)
    cum_y   = np.cumsum(y_true[order]) / y_true.sum()
    cum_n   = np.arange(1, len(y_true) + 1) / len(y_true)
    lorenz  = np.trapz(cum_y, cum_n)
    return 2 * lorenz - 1

y_test_arr = y_test.values
gini_champ = gini_coefficient(y_test_arr, champ_prob_test)
gini_chall = gini_coefficient(y_test_arr, chall_prob_test)

print(f"Gini coefficient on holdout set:")
print(f"  Champion (GLM-Logistic): {gini_champ:.4f}")
print(f"  Challenger (CatBoost):   {gini_chall:.4f}")
print(f"  Challenger lead:         {gini_chall - gini_champ:+.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Registry: Register Both Models
# MAGIC
# MAGIC The registry is append-only. Each model version gets:
# MAGIC - A SHA-256 hash of its serialised bytes (tamper detection)
# MAGIC - Arbitrary metadata: training date, feature list, validation KPIs
# MAGIC - A version string you control
# MAGIC
# MAGIC The registry stores model objects as joblib files alongside a JSON
# MAGIC manifest. No database required. Works on a shared filesystem or
# MAGIC Databricks DBFS.
# MAGIC
# MAGIC **Design note on append-only:** you cannot delete or overwrite a
# MAGIC registered version. This is intentional. If you need to fix a model,
# MAGIC register a new version. The audit trail requires knowing what was
# MAGIC deployed, not just what is deployed now.

# COMMAND ----------

# Create registry in /tmp (persists for the cluster lifetime)
REGISTRY_DIR = "/tmp/insurance_deploy_demo_registry"

registry = ModelRegistry(REGISTRY_DIR)
print(f"Registry at: {registry.path}")

# COMMAND ----------

# Register champion
mv_champion = registry.register(
    champion_model,
    name="motor_frequency",
    version="1.0",
    metadata={
        "model_type": "LogisticRegression (GLM)",
        "training_date": "2026-01-15",
        "features": FEATURES,
        "gini_holdout": round(gini_champ, 4),
        "n_train": len(X_train),
        "notes": "Production model. One-hot encoding, no interactions.",
    },
)

print(f"Registered: {mv_champion}")
print(f"  Hash: {mv_champion.model_hash[:16]}...")
print(f"  File: {mv_champion.model_path}")

# COMMAND ----------

# Register challenger
mv_challenger = registry.register(
    challenger_model,
    name="motor_frequency",
    version="2.0",
    metadata={
        "model_type": "CatBoostClassifier",
        "training_date": "2026-02-20",
        "features": FEATURES,
        "gini_holdout": round(gini_chall, 4),
        "n_train": len(X_train),
        "notes": "Challenger. Native categorical handling, depth-5 trees.",
    },
)

print(f"Registered: {mv_challenger}")
print(f"  Hash: {mv_challenger.model_hash[:16]}...")

# COMMAND ----------

# Show the registry contents
print("All registered versions for 'motor_frequency':")
for mv in registry.list("motor_frequency"):
    print(f"  {mv}")
    print(f"    Gini: {mv.metadata.get('gini_holdout')}")
    print(f"    Type: {mv.metadata.get('model_type')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hash verification in action
# MAGIC
# MAGIC When you load a model from the registry, it re-hashes the file and
# MAGIC compares to the stored hash. Corruption or tampering raises a `ValueError`
# MAGIC before the model object is returned.

# COMMAND ----------

# Load champion back from registry — hash is verified on load
loaded_champion = registry.get("motor_frequency", "1.0")
print(f"Loaded: {loaded_champion}")
print(f"Hash matches: True (or we'd have an exception)")

# Verify it still produces the same predictions
test_pred_verify = loaded_champion.predict(X_test)
# predict() calls model.predict(), which for a Pipeline returns class labels
# For a price prediction comparison we'd use predict_proba
loaded_proba = loaded_champion.model.predict_proba(X_test)[:, 1]
assert np.allclose(loaded_proba, champ_prob_test, rtol=1e-6), "Prediction mismatch after reload!"
print("Prediction round-trip: OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Experiment: Shadow Mode
# MAGIC
# MAGIC Shadow mode is the right starting point for every champion/challenger
# MAGIC experiment. The champion prices all live quotes. The challenger scores
# MAGIC in parallel — its output is logged but never shown to the customer.
# MAGIC
# MAGIC Zero regulatory risk. You accumulate challenger data without any FCA
# MAGIC Consumer Duty (PRIN 2A) fair-value exposure that would arise from
# MAGIC actually splitting live traffic.
# MAGIC
# MAGIC Routing is deterministic: `SHA-256(policy_id + experiment_name)`,
# MAGIC last 8 hex characters modulo 100. The same policy always routes to
# MAGIC the same arm within a named experiment. This is required for audit
# MAGIC integrity — you need to be able to say which model priced a specific
# MAGIC policy, not just which model was running that day.

# COMMAND ----------

# Register the champion as the official champion
registry.set_champion("motor_frequency", "1.0")

# Retrieve ModelVersion objects from registry
champ_mv   = registry.get("motor_frequency", "1.0")
chall_mv   = registry.get("motor_frequency", "2.0")

# Create shadow experiment: 20% of policies routed to challenger for scoring
# (in shadow mode, routing only determines which model scores in the background —
#  champion always prices)
experiment = Experiment(
    name="motor_v2_vs_v1_shadow",
    champion=champ_mv,
    challenger=chall_mv,
    challenger_pct=0.20,
    mode="shadow",
)

print(f"Experiment: {experiment}")
print()

# Demonstrate deterministic routing on a few policy IDs
test_pols = ["POL-00001", "POL-00002", "POL-00003", "POL-00100", "POL-00200"]
print("Routing decisions (deterministic):")
for pid in test_pols:
    arm  = experiment.route(pid)
    live = experiment.live_model(pid)
    shad = experiment.shadow_model(pid)
    print(f"  {pid}: arm={arm}, live={live.version_id}, shadow={shad.version_id}")

print()
print("Note: In shadow mode, live_model() always returns champion.")
print("      shadow_model() returns challenger for challenger-routed policies.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Routing distribution check
# MAGIC
# MAGIC With 20% challenger allocation and a hash-based scheme, we expect
# MAGIC approximately 20% of policies to route to challenger. The hash
# MAGIC is uniform — verify it on the portfolio.

# COMMAND ----------

routes = [experiment.route(pid) for pid in df["policy_id"]]
n_challenger_routed = sum(1 for r in routes if r == "challenger")
n_champion_routed   = sum(1 for r in routes if r == "champion")

print(f"Routing distribution across {N:,} policies:")
print(f"  Champion arm:   {n_champion_routed:,}  ({n_champion_routed/N:.1%})")
print(f"  Challenger arm: {n_challenger_routed:,}  ({n_challenger_routed/N:.1%})")
print(f"  Expected challenger allocation: 20.0%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quote Logging
# MAGIC
# MAGIC Every quote is logged to an append-only SQLite database. Three tables:
# MAGIC
# MAGIC - **quotes** — one row per quote, including arm, model version, price,
# MAGIC   ENBP (if renewal), and ENBP compliance flag
# MAGIC - **binds** — one row per policy that converted
# MAGIC - **claims** — one row per claim development update (FNOL, 3m, 6m, 12m)
# MAGIC
# MAGIC The logger enforces business rules:
# MAGIC - Raises a warning if a renewal quote is logged without an ENBP value
# MAGIC - Raises a warning and sets `enbp_flag=0` if quoted price > ENBP
# MAGIC - Validates arm must be 'champion' or 'challenger'
# MAGIC
# MAGIC SQLite handles 1M–10M rows comfortably. For larger books, the adapter
# MAGIC pattern in the README describes how to swap in a PostgreSQL backend.

# COMMAND ----------

# Create the quote logger
LOG_PATH = "/tmp/insurance_deploy_demo_quotes.db"

# Remove any prior demo run
import os
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

logger = QuoteLogger(LOG_PATH)

# Simulate quote timestamps spread over 6 months
base_date = datetime(2025, 7, 1, tzinfo=timezone.utc)

print("Logging quotes for all 10,000 policies...")

for i, row in df.iterrows():
    pid     = row["policy_id"]
    arm     = experiment.route(pid)
    price   = float(row["quoted_price"])
    renew   = bool(row["renewal_flag"])
    enbp    = float(row["enbp"]) if renew else None

    # Spread quotes over 6 months for realistic KPI time-series
    ts = base_date + timedelta(days=rng.integers(0, 183))

    # Log quote — in shadow mode, arm is used for logging only;
    # the live price is always the champion price
    logger.log_quote(
        policy_id=pid,
        experiment_name=experiment.name,
        arm=arm,
        model_version=champ_mv.version_id if arm == "champion" else chall_mv.version_id,
        quoted_price=price,
        enbp=enbp,
        renewal_flag=renew,
        exposure=1.0,
        timestamp=ts,
    )

print(f"Logged {logger.quote_count(experiment.name):,} quotes.")

# COMMAND ----------

# Log binds: policies that converted
n_bound = 0
for i, row in df.iterrows():
    if row["bound"]:
        pid = row["policy_id"]
        ts  = base_date + timedelta(days=rng.integers(0, 183))
        logger.log_bind(pid, bound_price=float(row["quoted_price"]), bound_timestamp=ts)
        n_bound += 1

print(f"Logged {n_bound:,} bind events.")

# COMMAND ----------

# Log claims with development months
# We log at FNOL (month 0), 6-month, and 12-month development
# to simulate a realistic claims triangle
MEAN_SEVERITY = 2800
n_claims_logged = 0

for i, row in df.iterrows():
    if row["claim"] and row["bound"]:
        pid   = row["policy_id"]
        # Claim date: randomly within the policy year
        claim_date = date(2025, rng.integers(7, 13), rng.integers(1, 28))
        severity = float(rng.gamma(shape=2.0, scale=MEAN_SEVERITY / 2.0))

        # FNOL (month 0): initial reserve, 60% of ultimate
        logger.log_claim(pid, claim_date, round(severity * 0.60, 2), development_month=0)
        # 6-month: 80% developed
        logger.log_claim(pid, claim_date, round(severity * 0.80, 2), development_month=6)
        # 12-month: fully developed
        logger.log_claim(pid, claim_date, round(severity, 2), development_month=12)
        n_claims_logged += 1

print(f"Logged {n_claims_logged:,} claims ({n_claims_logged * 3:,} development rows).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect the audit trail
# MAGIC
# MAGIC The SQLite database is a permanent record. Here we pull the first
# MAGIC few rows to confirm the schema looks right.

# COMMAND ----------

# Show a sample from each table
quotes_sample = logger.query_quotes(experiment.name)[:5]
binds_sample  = logger.query_binds()[:5]
claims_sample = logger.query_claims()[:5]

print("=== QUOTES (first 5) ===")
for q in quotes_sample:
    renewal_str = " [RENEWAL]" if q["renewal_flag"] else ""
    enbp_str    = f"  ENBP: £{q['enbp']:.2f} flag={q['enbp_flag']}" if q["enbp_flag"] is not None else ""
    print(f"  {q['policy_id']}  arm={q['arm']:<12}  model={q['model_version']:<22}"
          f"  price=£{q['quoted_price']:.2f}{renewal_str}{enbp_str}")

print()
print("=== BINDS (first 5) ===")
for b in binds_sample:
    print(f"  {b['policy_id']}  bound=£{b['bound_price']:.2f}  at {b['bound_timestamp'][:19]}")

print()
print("=== CLAIMS (first 5) ===")
for c in claims_sample:
    print(f"  {c['policy_id']}  {c['claim_date']}  dev_month={c['development_month']}  "
          f"incurred=£{c['claim_amount']:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. KPI Tracking
# MAGIC
# MAGIC The `KPITracker` queries the SQLite log and computes KPIs by arm.
# MAGIC KPIs are organised into maturity tiers — because some metrics are
# MAGIC available immediately, others need 6 months, and loss ratio needs
# MAGIC 12+ months of development.
# MAGIC
# MAGIC | Tier | Metric | When available |
# MAGIC |------|--------|---------------|
# MAGIC | 1 | Quote volume, price distribution, ENBP compliance | Immediately |
# MAGIC | 2 | Hit rate (conversion), GWP | At bind |
# MAGIC | 3 | Claim frequency | 3–6 months |
# MAGIC | 4 | Loss ratio | 12+ months |
# MAGIC
# MAGIC This tiering matters. Teams that wait for loss ratio before making any
# MAGIC promotion decision are waiting 18+ months unnecessarily. Hit rate and
# MAGIC frequency give you earlier — imperfect, but real — signals.

# COMMAND ----------

tracker = KPITracker(logger)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tier 1: Quote volume and price distribution

# COMMAND ----------

vol = tracker.quote_volume(experiment.name)
print("Quote volume by arm:")
for arm, stats in vol.items():
    print(f"  {arm.capitalize():<12}: "
          f"n={stats['n']:>5,}  "
          f"mean_price=£{stats['mean_price']:.2f}  "
          f"median=£{stats['median_price']:.2f}  "
          f"IQR=[£{stats['p25_price']:.2f}, £{stats['p75_price']:.2f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tier 1: ENBP compliance by arm

# COMMAND ----------

enbp = tracker.enbp_compliance(experiment.name)
print("ENBP compliance by arm:")
for arm, stats in enbp.items():
    rate_str = f"{stats['compliance_rate']:.1%}" if not pd.isna(stats['compliance_rate']) else "N/A"
    print(f"  {arm.capitalize():<12}: "
          f"renewal_quotes={stats['renewal_quotes']:>4,}  "
          f"compliant={stats['compliant']:>4,}  "
          f"breaches={stats['breaches']:>3,}  "
          f"compliance_rate={rate_str}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tier 2: Hit rate and GWP

# COMMAND ----------

hr = tracker.hit_rate(experiment.name)
gw = tracker.gwp(experiment.name)

print("Hit rate (conversion) by arm:")
for arm, stats in hr.items():
    rate_str = f"{stats['hit_rate']:.1%}"
    print(f"  {arm.capitalize():<12}: quoted={stats['quoted']:>5,}  "
          f"bound={stats['bound']:>4,}  hit_rate={rate_str}")

print()
print("GWP on bound policies by arm:")
for arm, stats in gw.items():
    mean_str = f"£{stats['mean_gwp']:.2f}" if not pd.isna(stats['mean_gwp']) else "N/A"
    print(f"  {arm.capitalize():<12}: policies={stats['bound_policies']:>4,}  "
          f"total_gwp=£{stats['total_gwp']:>10,.2f}  mean_gwp={mean_str}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tier 3: Claim frequency (6-month development)
# MAGIC
# MAGIC Frequency at 6 months carries IBNR uncertainty — the library warns
# MAGIC you. At this stage approximately 30–40% of ultimate motor claims
# MAGIC may not yet be reported. Use as an early directional signal only.

# COMMAND ----------

freq = tracker.frequency(experiment.name, development_months=6, warn_immature=False)
print("Claim frequency by arm (6-month development):")
for arm, stats in freq.items():
    ibnr_note = " [IBNR caveat]" if stats["maturity_warning"] else ""
    print(f"  {arm.capitalize():<12}: "
          f"policy_years={stats['policy_years']:.0f}  "
          f"claims={stats['claim_count']:>4,}  "
          f"frequency={stats['frequency']:.4f}{ibnr_note}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tier 4: Loss ratio (12-month development)

# COMMAND ----------

lr = tracker.loss_ratio(experiment.name, development_months=12)
print("Loss ratio by arm (12-month development):")
for arm, stats in lr.items():
    lr_str = f"{stats['loss_ratio']:.1%}" if not pd.isna(stats['loss_ratio']) else "N/A"
    print(f"  {arm.capitalize():<12}: "
          f"policies={stats['policy_count']:>4,}  "
          f"earned=£{stats['earned_premium']:>10,.2f}  "
          f"incurred=£{stats['incurred_claims']:>9,.2f}  "
          f"LR={lr_str}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### KPI summary report
# MAGIC
# MAGIC The summary report is a pandas DataFrame view of Tiers 1–2 KPIs.
# MAGIC It is the snapshot you would include in a monthly steering committee pack.

# COMMAND ----------

summary = tracker.summary_report(experiment.name)
print("KPI Summary:")
print(summary.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Power analysis: how long until we can promote?
# MAGIC
# MAGIC This is where most teams go wrong. They eyeball a 2pp improvement in
# MAGIC loss ratio after 3 months and call it done. The power analysis tells
# MAGIC you the sample size actually required to distinguish a 3pp LR improvement
# MAGIC from noise at 80% power, given current volume and the 10/90 split.
# MAGIC
# MAGIC The answer is usually disappointing. That is the point.

# COMMAND ----------

pa = tracker.power_analysis(
    experiment.name,
    target_delta_lr=0.03,   # detect 3pp LR improvement
    target_delta_hr=0.02,   # detect 2pp hit rate improvement
    alpha=0.05,
    power=0.80,
)

print("Power Analysis:")
print(f"  Current n_champion:   {pa['current_n_champion']:,}")
print(f"  Current n_challenger: {pa['current_n_challenger']:,}")
print(f"  Months elapsed:       {pa['months_elapsed']:.1f}")
print()
print(f"  Hit rate (detect {pa['target_delta_hr']:.0%} delta):")
print(f"    Required n per arm: {pa['hr_required_n_per_arm']:,}")
print(f"    Months to significance: {pa['hr_months_to_significance']:.1f}")
print()
print(f"  Loss ratio (detect {pa['target_delta_lr']:.0%} delta):")
print(f"    Required n per arm: {pa['lr_required_n_per_arm']:,}")
print(f"    Months to bind: {pa['lr_months_to_bind']:.1f}")
print(f"    Total (bind + 12m dev): {pa['lr_total_months_with_development']:.1f} months")
print()
print("Notes:")
for note in pa["notes"]:
    print(f"  - {note}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Statistical Comparison
# MAGIC
# MAGIC Three tests, in order of when you can run them:
# MAGIC
# MAGIC 1. **Hit rate z-test** — available after a few months once conversion
# MAGIC    data accumulates. A useful early signal, but confounded by price
# MAGIC    differences in live mode (not an issue here since we're in shadow).
# MAGIC
# MAGIC 2. **Poisson frequency test** — available at 6–9 months. Frequency is
# MAGIC    a better early signal than LR because you don't need fully developed
# MAGIC    claims. The library uses a log rate ratio test (Poisson GLM approximation).
# MAGIC
# MAGIC 3. **Bootstrap loss ratio test** — the definitive test. Only appropriate
# MAGIC    at 12+ months development. Policy-level block bootstrap preserves
# MAGIC    within-policy claim correlation (a policy with one bad claim shouldn't
# MAGIC    look like evidence of model failure just because of variance).
# MAGIC
# MAGIC The library never automatically promotes. It returns a `ComparisonResult`
# MAGIC with a conclusion (`CHALLENGER_BETTER`, `CHAMPION_BETTER`, or
# MAGIC `INSUFFICIENT_EVIDENCE`) and a recommendation. Humans decide.

# COMMAND ----------

comparison = ModelComparison(tracker)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 1: Hit rate z-test

# COMMAND ----------

hr_result = comparison.hit_rate_test(experiment.name, alpha=0.05)
print(hr_result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 2: Poisson frequency test (6-month development)

# COMMAND ----------

freq_result = comparison.frequency_test(experiment.name, development_months=6)
print(freq_result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test 3: Bootstrap loss ratio test (12-month development)
# MAGIC
# MAGIC This is the definitive comparison. 10,000 bootstrap iterations, resampling
# MAGIC at policy level. The 95% CI is percentile-based (not normal approximation),
# MAGIC which handles the heavy-tailed distribution of loss ratios correctly.
# MAGIC
# MAGIC In shadow mode the two arms have identical price distributions (both priced
# MAGIC by champion), so there is no adverse selection concern. The library flags
# MAGIC this issue if you switch to live mode.

# COMMAND ----------

lr_result = comparison.bootstrap_lr_test(
    experiment.name,
    n_bootstrap=10_000,
    development_months=12,
    seed=42,
)
print(lr_result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualise the bootstrap distribution

# COMMAND ----------

# Re-run bootstrap to get the raw distribution for plotting
rng_plot = np.random.default_rng(42)
policy_data = comparison._build_policy_loss_data(experiment.name, 12)
champ_data  = [(p, e, c) for p, e, c, arm in policy_data if arm == "champion"]
chall_data  = [(p, e, c) for p, e, c, arm in policy_data if arm == "challenger"]

n_boot = 5_000
boot_diffs = np.empty(n_boot)
for i in range(n_boot):
    bs_c = rng_plot.choice(len(champ_data), size=len(champ_data), replace=True)
    bs_h = rng_plot.choice(len(chall_data), size=len(chall_data), replace=True)
    lr_c = sum(c for p, e, c in [champ_data[j] for j in bs_c]) / sum(p * e for p, e, c in [champ_data[j] for j in bs_c])
    lr_h = sum(c for p, e, c in [chall_data[j] for j in bs_h]) / sum(p * e for p, e, c in [chall_data[j] for j in bs_h])
    boot_diffs[i] = lr_h - lr_c

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: bootstrap distribution
ax = axes[0]
ax.hist(boot_diffs, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
ax.axvline(lr_result.difference, color="crimson", linewidth=2, label=f"Point estimate ({lr_result.difference:+.4f})")
ax.axvline(lr_result.ci_lower, color="darkorange", linewidth=1.5, linestyle="--", label=f"95% CI lower ({lr_result.ci_lower:+.4f})")
ax.axvline(lr_result.ci_upper, color="darkorange", linewidth=1.5, linestyle="--", label=f"95% CI upper ({lr_result.ci_upper:+.4f})")
ax.axvline(0, color="black", linewidth=1, linestyle=":")
ax.set_xlabel("LR difference (challenger - champion)")
ax.set_ylabel("Bootstrap frequency")
ax.set_title("Bootstrap Distribution: Loss Ratio Difference")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: loss ratio by arm with CI
ax = axes[1]
arms = ["Champion", "Challenger"]
lr_vals = [lr_result.champion_estimate, lr_result.challenger_estimate]
# Approximate individual arm CIs from bootstrap (rough symmetric CI)
boot_lr_champ = np.array([
    sum(c for p, e, c in [champ_data[j] for j in rng_plot.choice(len(champ_data), size=len(champ_data), replace=True)]) /
    sum(p * e for p, e, c in [champ_data[j] for j in rng_plot.choice(len(champ_data), size=len(champ_data), replace=True)])
    for _ in range(1000)
])
ci_champ = np.percentile(boot_lr_champ, [2.5, 97.5])

bars = ax.bar(arms, lr_vals, color=["steelblue", "tomato"], alpha=0.7, width=0.4)
ax.errorbar(
    ["Champion"],
    [lr_result.champion_estimate],
    yerr=[[lr_result.champion_estimate - ci_champ[0]], [ci_champ[1] - lr_result.champion_estimate]],
    fmt="none", color="black", capsize=8, linewidth=2,
)
for bar, val in zip(bars, lr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Loss Ratio")
ax.set_title("Loss Ratio by Arm (12-month development)")
ax.set_ylim(0, max(lr_vals) * 1.3)
ax.grid(True, alpha=0.3, axis="y")
ax.text(0.5, 0.92, f"Conclusion: {lr_result.conclusion}", transform=ax.transAxes,
        ha="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.suptitle("Bootstrap Loss Ratio Test: Champion vs Challenger", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/bootstrap_lr_test.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved to /tmp/bootstrap_lr_test.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Promotion: Set Challenger as New Champion
# MAGIC
# MAGIC If the statistical tests and human review support promotion, the
# MAGIC registry records the champion change. This is a metadata update —
# MAGIC the old champion version stays in the registry permanently.

# COMMAND ----------

if lr_result.conclusion == "CHALLENGER_BETTER":
    new_champ = registry.set_champion("motor_frequency", "2.0")
    print(f"Promoted challenger to champion: {new_champ}")
    print()
    print("Registry state after promotion:")
    for mv in registry.list("motor_frequency"):
        champion_marker = " [CHAMPION]" if mv.is_champion else ""
        print(f"  {mv}{champion_marker}")
else:
    print(f"Conclusion was {lr_result.conclusion!r} — champion retained.")
    print()
    print("This is the correct outcome when evidence is insufficient or ambiguous.")
    print("The registry is unchanged. Deactivate the experiment or extend it.")
    print()
    print("Registry state:")
    for mv in registry.list("motor_frequency"):
        champion_marker = " [CHAMPION]" if mv.is_champion else ""
        print(f"  {mv}{champion_marker}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. ICOBS 6B.2.51R Audit Report
# MAGIC
# MAGIC The FCA's multi-firm review in 2023 found 83% of firms non-compliant
# MAGIC with ICOBS 6B.2.51R — the rule requiring firms to record how they
# MAGIC demonstrated no systematic tenure discrimination in renewal pricing.
# MAGIC
# MAGIC The compliance failure was almost entirely a record-keeping failure,
# MAGIC not a pricing failure. Firms had the ENBP calculations somewhere; they
# MAGIC could not produce a written record tying specific renewal quotes to
# MAGIC ENBP comparisons with model version audit trails.
# MAGIC
# MAGIC The `ENBPAuditReport` generates that record from the QuoteLogger data.
# MAGIC It is designed to be handed directly to an SMF holder or compliance
# MAGIC team — the recipient is not a data scientist.

# COMMAND ----------

reporter = ENBPAuditReport(logger)

report_md = reporter.generate(
    experiment_name=experiment.name,
    period_start="2025-07-01",
    period_end="2025-12-31",
    firm_name="Acme Motor Insurance Ltd",
    smf_holder="[Chief Pricing Officer]",
)

print(report_md)

# COMMAND ----------

# Save report to file
report_path = "/tmp/enbp_audit_report.md"
with open(report_path, "w") as f:
    f.write(report_md)

print(f"Audit report saved to: {report_path}")
print(f"Report length: {len(report_md):,} characters")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Verdict
# MAGIC
# MAGIC ### What insurance-deploy replaces
# MAGIC
# MAGIC | Before | After |
# MAGIC |--------|-------|
# MAGIC | Ad-hoc scripts per experiment, results in a shared drive | Reproducible, version-controlled framework |
# MAGIC | Random routing (`random.random() < 0.1`) — not replicable | Hash-based deterministic routing — any decision is auditable |
# MAGIC | Excel log of "which model was running when" | Append-only SQLite audit trail per quote |
# MAGIC | ENBP check done manually before renewals file | ENBP flag written at quote time, compliance rate always current |
# MAGIC | "We promoted because it looked better" | Bootstrap LR test with 95% CI and explicit promotion recommendation |
# MAGIC | Compliance attestation = a paragraph in an email | ICOBS 6B.2.51R-structured Markdown report with breach detail |
# MAGIC
# MAGIC ### When to use shadow mode vs live mode
# MAGIC
# MAGIC **Shadow mode (default):** The right starting point for every experiment.
# MAGIC Champion prices all quotes. Challenger scores in parallel, logged but invisible
# MAGIC to the customer. Zero regulatory risk. Use until you have statistical evidence.
# MAGIC
# MAGIC **Live mode:** Use only after shadow mode has produced a clear hypothesis and
# MAGIC after legal sign-off on FCA Consumer Duty (PRIN 2A) fair value implications.
# MAGIC Two customers with identical risk profiles receiving different prices simultaneously
# MAGIC requires a documented fair value justification. Shadow mode gives you the
# MAGIC evidence without the exposure.
# MAGIC
# MAGIC ### Realistic timelines for UK motor
# MAGIC
# MAGIC - Hit rate signal: 3–4 months at typical volumes
# MAGIC - Frequency signal: 9–12 months (IBNR caveat applies earlier)
# MAGIC - Loss ratio signal: 18–24 months (12 months to bind + 12 months development)
# MAGIC
# MAGIC Teams that expect a promotion decision in 60 days will be disappointed.
# MAGIC The power analysis is there to set realistic expectations before the experiment
# MAGIC starts, not to justify whatever the data shows after 6 weeks.
# MAGIC
# MAGIC ### Limitations
# MAGIC
# MAGIC - SQLite is not appropriate for multi-process concurrent writes. If your
# MAGIC   rating engine is distributed, use the adapter pattern to point QuoteLogger
# MAGIC   at a shared database.
# MAGIC - The library records ENBP values you provide. It does not calculate them.
# MAGIC   ENBP calculation per ICOBS 6B methodology is your pricing team's
# MAGIC   responsibility.
# MAGIC - The bootstrap LR test assumes policy-level independence across policies
# MAGIC   (correct) but does not model claim development correlation across policies.
# MAGIC   For portfolios with strong systemic risk (catastrophe exposure), treat the
# MAGIC   CI as optimistic.

# COMMAND ----------

# Final summary of what we demonstrated
print("=" * 60)
print("Demo complete. What was covered:")
print("=" * 60)
print()
print("1. Registry")
print(f"   - Registered champion v1.0 (GLM-Logistic, Gini={gini_champ:.3f})")
print(f"   - Registered challenger v2.0 (CatBoost, Gini={gini_chall:.3f})")
print(f"   - Hash verification on reload: OK")
print()
print("2. Experiment")
print(f"   - Shadow mode, 20% challenger routing")
print(f"   - Deterministic hash routing across {N:,} policies")
print()
print("3. Quote logging")
print(f"   - {logger.quote_count(experiment.name):,} quotes logged")
print(f"   - {len(logger.query_binds()):,} binds logged")
print(f"   - {len(logger.query_claims()):,} claim development rows logged")
print()
print("4. KPI tracking (4-tier maturity model)")
for arm in ("champion", "challenger"):
    hr_stats = tracker.hit_rate(experiment.name).get(arm, {})
    lr_stats = tracker.loss_ratio(experiment.name).get(arm, {})
    print(f"   {arm}: hit_rate={hr_stats.get('hit_rate', 0):.1%}  "
          f"LR={lr_stats.get('loss_ratio', float('nan')):.1%}")
print()
print("5. Statistical comparison")
print(f"   - Hit rate z-test:          {hr_result.conclusion}")
print(f"   - Frequency Poisson test:   {freq_result.conclusion}")
print(f"   - Bootstrap LR test:        {lr_result.conclusion}")
print(f"     p={lr_result.p_value:.4f}, 95% CI [{lr_result.ci_lower:.4f}, {lr_result.ci_upper:.4f}]")
print()
print("6. ICOBS 6B.2.51R audit report generated.")
print(f"   Report at: /tmp/enbp_audit_report.md")
