"""Run insurance-deploy tests on Databricks using the Jobs API."""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

env_path = Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip()

import urllib.request
import urllib.error

HOST = os.environ["DATABRICKS_HOST"].rstrip("/")
TOKEN = os.environ["DATABRICKS_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

TEST_SCRIPT = r"""
import sys, tempfile, random, warnings, math
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
import subprocess

# Install the library
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "insurance-deploy"])

import numpy as np
from insurance_deploy import (
    ModelRegistry, ModelVersion, Experiment, QuoteLogger,
    KPITracker, ModelComparison, ENBPAuditReport,
)

class DummyModel:
    def __init__(self, c=400.0):
        self.c = c
    def predict(self, X):
        return np.full(len(X), self.c)

tmp = Path(tempfile.mkdtemp())
passed = []
failed = []

def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print(f"PASS: {name}")
    else:
        failed.append(name)
        print(f"FAIL: {name} {detail}")

# --- Registry ---
r = ModelRegistry(tmp / "reg")
mv1 = r.register(DummyModel(400), "motor", "1.0", {"tag": "v1"})
mv2 = r.register(DummyModel(420), "motor", "2.0", {"tag": "v2"})
check("registry_register", mv1.version_id == "motor:1.0")
check("registry_metadata", mv1.metadata["tag"] == "v1")
check("registry_list", len(r.list()) == 2)
check("registry_list_filtered", len(r.list("motor")) == 2)
check("registry_get", r.get("motor", "1.0").version == "1.0")
check("registry_champion_default", r.champion("motor").version == "2.0")
r.set_champion("motor", "1.0")
check("registry_set_champion", r.champion("motor").version == "1.0")
check("registry_is_champion_flag", r.get("motor", "1.0").is_champion is True)

try:
    r.register(DummyModel(), "motor", "1.0")
    check("registry_duplicate_raises", False)
except ValueError:
    check("registry_duplicate_raises", True)

# Persistence
r2 = ModelRegistry(tmp / "reg")
check("registry_persistence", len(r2.list()) == 2)
check("registry_champion_persistence", r2.champion("motor").version == "1.0")

# Hash verification
import hashlib
Path(mv1.model_path).write_bytes(b"corrupt")
mv1_loaded = r2.get("motor", "1.0")
mv1_loaded._model = None
try:
    _ = mv1_loaded.model
    check("registry_hash_verification", False)
except ValueError:
    check("registry_hash_verification", True)

# --- Experiment routing ---
r3 = ModelRegistry(tmp / "reg3")
c1 = r3.register(DummyModel(400), "m", "1.0")
c2 = r3.register(DummyModel(420), "m", "2.0")
exp = Experiment("test_exp_routing", c1, c2, 0.10)

# Determinism
results_a = [exp.route(f"P-{i}") for i in range(1000)]
results_b = [exp.route(f"P-{i}") for i in range(1000)]
check("routing_deterministic", results_a == results_b)

# SHA-256 formula
pid = "POL-CHECK-001"
key = (pid + exp.name).encode()
digest = hashlib.sha256(key).hexdigest()
slot = int(digest[-8:], 16) % 100
expected = "challenger" if slot < 10 else "champion"
check("routing_sha256_formula", exp.route(pid) == expected)

# Approximate split
challenger_n = sum(1 for r in results_a if r == "challenger")
check("routing_split_approx", 75 <= challenger_n <= 125, f"got {challenger_n}")

# Lifecycle
exp.deactivate()
check("experiment_deactivated", not exp.is_active())
try:
    exp.route("P-0")
    check("routing_deactivated_raises", False)
except RuntimeError:
    check("routing_deactivated_raises", True)

# Invalid params
try:
    Experiment("bad", c1, c2, challenger_pct=0.0)
    check("experiment_invalid_pct_zero", False)
except ValueError:
    check("experiment_invalid_pct_zero", True)

try:
    Experiment("bad2", c1, c2, mode="bandit")
    check("experiment_invalid_mode", False)
except ValueError:
    check("experiment_invalid_mode", True)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    Experiment("live", c1, c2, mode="live")
    check("experiment_live_warning", any("Consumer Duty" in str(x.message) for x in w))

# Shadow mode: live_model always champion
exp2 = Experiment("shadow_exp", c1, c2, 0.50)
for pid in [f"Q-{i}" for i in range(20)]:
    check(f"shadow_live_model_{pid}", exp2.live_model(pid).version_id == c1.version_id)
# (only first few count for reporting)
passed_shadow = all(exp2.live_model(f"Q-{i}").version_id == c1.version_id for i in range(20))
# Already checked individually above, replace with summary
passed = [p for p in passed if not p.startswith("shadow_live_model_")]
check("shadow_live_model_always_champion", passed_shadow)

# --- Logger ---
log = QuoteLogger(tmp / "q.db")
log.log_quote("P1", "e", "champion", "m:1", 400.0, enbp=420.0, renewal_flag=True)
log.log_bind("P1", 400.0)
log.log_claim("P1", date(2024, 6, 1), 1200.0, 12)
check("logger_quote_count", log.quote_count() == 1)
q = log.query_quotes()[0]
check("logger_quote_enbp_compliant", q["enbp_flag"] == 1)
check("logger_quote_fields", q["policy_id"] == "P1" and q["arm"] == "champion")
check("logger_bind", len(log.query_binds()) == 1)
check("logger_claim", len(log.query_claims()) == 1)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    log.log_quote("P2", "e", "champion", "m:1", 450.0, enbp=430.0, renewal_flag=True)
    check("logger_enbp_breach_warning", any("ENBP breach" in str(x.message) for x in w))
q2 = [q for q in log.query_quotes() if q["policy_id"] == "P2"][0]
check("logger_enbp_breach_flag", q2["enbp_flag"] == 0)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    log.log_quote("P3", "e", "champion", "m:1", 400.0, renewal_flag=True, enbp=None)
    check("logger_no_enbp_renewal_warning", any("no enbp provided" in str(x.message) for x in w))

try:
    log.log_quote("P4", "e", "neutral", "m:1", 400.0)
    check("logger_invalid_arm", False)
except ValueError:
    check("logger_invalid_arm", True)

try:
    log.log_quote("P5", "e", "champion", "m:1", -100.0)
    check("logger_negative_price", False)
except ValueError:
    check("logger_negative_price", True)

try:
    log.log_claim("P1", date(2024, 6, 1), -50.0, 3)
    check("logger_negative_claim", False)
except ValueError:
    check("logger_negative_claim", True)

# Filter by experiment
log.log_quote("X1", "exp_other", "champion", "m:1", 400.0)
check("logger_filter_experiment", log.quote_count("e") == 3)
check("logger_no_delete_method", not hasattr(log, "delete_quote"))

# --- KPITracker ---
log2 = QuoteLogger(tmp / "q2.db")
for i in range(10):
    log2.log_quote(f"P{i}", "kpi_exp", "champion", "m:1", 400.0 + i)
for i in range(3):
    log2.log_bind(f"P{i}", 400.0 + i)
tracker = KPITracker(log2)

hr = tracker.hit_rate("kpi_exp")
check("kpi_hit_rate_quoted", hr["champion"]["quoted"] == 10)
check("kpi_hit_rate_bound", hr["champion"]["bound"] == 3)
check("kpi_hit_rate_value", abs(hr["champion"]["hit_rate"] - 0.30) < 0.01)

vol = tracker.quote_volume("kpi_exp")
check("kpi_volume_n", vol["champion"]["n"] == 10)
check("kpi_volume_mean", abs(vol["champion"]["mean_price"] - 404.5) < 0.1)

gwp = tracker.gwp("kpi_exp")
check("kpi_gwp_total", abs(gwp["champion"]["total_gwp"] - (400 + 401 + 402)) < 1)
check("kpi_gwp_mean", abs(gwp["champion"]["mean_gwp"] - 401.0) < 1)

# LR calculation
log3 = QuoteLogger(tmp / "q3.db")
log3.log_quote("P1", "lr_exp", "champion", "m:1", 400.0)
log3.log_bind("P1", 400.0)
log3.log_claim("P1", date(2024, 6, 1), 200.0, 12)
tracker3 = KPITracker(log3)
lr = tracker3.loss_ratio("lr_exp", development_months=12)
check("kpi_lr_calculation", abs(lr["champion"]["loss_ratio"] - 0.5) < 0.01)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    tracker3.loss_ratio("lr_exp", development_months=6)
    check("kpi_lr_immature_warning", any("immature" in str(x.message).lower() for x in w))

# Power analysis
pa = tracker.power_analysis("kpi_exp", target_delta_lr=0.03)
check("kpi_pa_positive_hr_n", pa["hr_required_n_per_arm"] > 0)
check("kpi_pa_positive_lr_n", pa["lr_required_n_per_arm"] > 0)
check("kpi_pa_lr_includes_dev", pa["lr_total_months_with_development"] > pa.get("lr_months_to_bind", 0))
check("kpi_pa_notes", len(pa["notes"]) > 0)

# Larger delta -> fewer required
pa2 = tracker.power_analysis("kpi_exp", target_delta_lr=0.05)
check("kpi_pa_larger_delta_fewer", pa["lr_required_n_per_arm"] > pa2["lr_required_n_per_arm"])

# --- Bootstrap LR test ---
log4 = QuoteLogger(tmp / "q4.db")
rng2 = random.Random(7)
base = datetime(2024, 1, 1, tzinfo=timezone.utc)
for i in range(300):
    pid = f"B-{i}"
    ts = base + timedelta(days=i // 5)
    p = max(100, rng2.gauss(400, 50))
    log4.log_quote(pid, "bt", "champion", "m:1", p, enbp=p+20, renewal_flag=True, timestamp=ts)
    if rng2.random() < 0.30:
        log4.log_bind(pid, p, ts + timedelta(hours=1))
        if rng2.random() < 0.08:
            log4.log_claim(pid, date(2024, 6, 1), max(0, rng2.gauss(1500, 500)), 12)
for i in range(100):
    pid = f"C-{i}"
    ts = base + timedelta(days=i // 2)
    p = max(100, rng2.gauss(410, 50))
    log4.log_quote(pid, "bt", "challenger", "m:2", p, enbp=p+20, renewal_flag=True, timestamp=ts)
    if rng2.random() < 0.28:
        log4.log_bind(pid, p, ts + timedelta(hours=1))
        if rng2.random() < 0.075:
            log4.log_claim(pid, date(2024, 6, 1), max(0, rng2.gauss(1400, 500)), 12)

t4 = KPITracker(log4)
comp = ModelComparison(t4)
result = comp.bootstrap_lr_test("bt", n_bootstrap=1000, seed=42)
check("bootstrap_valid_conclusion", result.conclusion in {"INSUFFICIENT_EVIDENCE", "CHALLENGER_BETTER", "CHAMPION_BETTER"})
check("bootstrap_not_nan", not math.isnan(result.champion_estimate))
check("bootstrap_ci_ordered", result.ci_lower <= result.ci_upper)
check("bootstrap_p_value_range", 0 < result.p_value <= 1.0)
check("bootstrap_n_correct", result.n_champion > 0 and result.n_challenger > 0)
check("bootstrap_ci_contains_diff", result.ci_lower <= result.difference <= result.ci_upper)

# Maturity warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r_immature = comp.bootstrap_lr_test("bt", n_bootstrap=100, development_months=6, seed=0)
    check("bootstrap_maturity_warning", any("development_months" in str(x.message) for x in w))
    check("bootstrap_maturity_flag", r_immature.maturity_warning is True)

# Seed reproducibility
r1 = comp.bootstrap_lr_test("bt", n_bootstrap=200, seed=99)
r2 = comp.bootstrap_lr_test("bt", n_bootstrap=200, seed=99)
check("bootstrap_seed_reproducible", abs(r1.p_value - r2.p_value) < 1e-10)

# Empty experiment
r_empty = comp.bootstrap_lr_test("nonexistent", n_bootstrap=10)
check("bootstrap_empty_insufficient", r_empty.conclusion == "INSUFFICIENT_EVIDENCE")

# Hit rate test
hr_result = comp.hit_rate_test("bt")
check("hitrate_test_name", hr_result.test_name == "hit_rate_test")
check("hitrate_diff_consistent", abs(hr_result.difference - (hr_result.challenger_estimate - hr_result.champion_estimate)) < 1e-6)
check("hitrate_ci_ordered", math.isnan(hr_result.ci_lower) or hr_result.ci_lower < hr_result.ci_upper)

# Frequency test
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    freq_result = comp.frequency_test("bt", development_months=12)
check("freq_test_name", freq_result.test_name == "frequency_test")
check("freq_valid_conclusion", freq_result.conclusion in {"INSUFFICIENT_EVIDENCE", "CHALLENGER_BETTER", "CHAMPION_BETTER"})

# Large difference -> detected
log5 = QuoteLogger(tmp / "q5.db")
rng3 = random.Random(11)
for i in range(500):
    pid = f"D-{i}"
    ts = base + timedelta(days=i // 8)
    p = max(100, rng3.gauss(400, 30))
    log5.log_quote(pid, "big_diff", "champion", "m:1", p, renewal_flag=False, timestamp=ts)
    if rng3.random() < 0.40:
        log5.log_bind(pid, p, ts + timedelta(hours=1))
for i in range(200):
    pid = f"E-{i}"
    ts = base + timedelta(days=i // 3)
    p = max(100, rng3.gauss(400, 30))
    log5.log_quote(pid, "big_diff", "challenger", "m:2", p, renewal_flag=False, timestamp=ts)
    if rng3.random() < 0.28:  # much lower conversion
        log5.log_bind(pid, p, ts + timedelta(hours=1))
t5 = KPITracker(log5)
c5 = ModelComparison(t5)
hr_big = c5.hit_rate_test("big_diff")
check("hitrate_detects_large_diff", hr_big.p_value < 0.05 or hr_big.conclusion != "INSUFFICIENT_EVIDENCE")

# --- ENBP Audit Report ---
from insurance_deploy import ENBPAuditReport
reporter = ENBPAuditReport(log4)
md = reporter.generate("bt", firm_name="Test Co Ltd", smf_holder="J Smith")
check("audit_is_string", isinstance(md, str))
check("audit_icobs_reference", "ICOBS 6B.2.51R" in md)
check("audit_has_executive_summary", "Executive Summary" in md)
check("audit_has_model_versions", "Model Versions" in md)
check("audit_has_routing_audit", "Routing Decision Audit" in md)
check("audit_has_attestation", "Attestation Statement" in md)
check("audit_has_firm_name", "Test Co Ltd" in md)
check("audit_has_smf_holder", "J Smith" in md)
check("audit_has_sha256", "SHA-256" in md)
check("audit_has_deterministic", "deterministic" in md)
check("audit_period_all_data", "All available data" in md)

md_period = reporter.generate("bt", period_start="2024-01-01", period_end="2024-12-31")
check("audit_period_label", "2024-01-01 to 2024-12-31" in md_period)

# No breaches case
log6 = QuoteLogger(tmp / "q6.db")
for i in range(10):
    log6.log_quote(f"R{i}", "clean", "champion", "m:1", 400.0, enbp=420.0, renewal_flag=True)
reporter6 = ENBPAuditReport(log6)
md6 = reporter6.generate("clean")
check("audit_no_breaches_message", "No ENBP breaches" in md6)

# Breach case
log7 = QuoteLogger(tmp / "q7.db")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    log7.log_quote("BR1", "breach_exp", "champion", "m:1", 450.0, enbp=430.0, renewal_flag=True)
reporter7 = ENBPAuditReport(log7)
md7 = reporter7.generate("breach_exp")
check("audit_breach_detail", "ENBP Breach Detail" in md7)
check("audit_breach_action_required", "Action required" in md7)

# ComparisonResult summary
check("result_summary_string", isinstance(result.summary(), str))
check("result_summary_contains_test", "bootstrap_lr_test" in result.summary())
check("result_summary_contains_conclusion", result.conclusion in result.summary())
check("result_repr", "bootstrap_lr_test" in repr(result))

print()
print("=" * 60)
print(f"Results: {len(passed)} passed, {len(failed)} failed")
if failed:
    print(f"FAILED tests: {failed}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
"""


def api_call(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{HOST}/api/2.0/{endpoint}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body[:500]}")
        raise


def run_via_script_job():
    """Use Databricks Repos or DBFS to run a Python script directly."""
    print("Uploading test script to DBFS...")
    import base64

    encoded = base64.b64encode(TEST_SCRIPT.encode()).decode()
    api_call("POST", "dbfs/put", {
        "path": "/FileStore/burning-cost/insurance-deploy-tests.py",
        "contents": encoded,
        "overwrite": True,
    })
    print("Script uploaded to DBFS")

    print("Submitting job run...")
    payload = {
        "run_name": "insurance-deploy-tests",
        "tasks": [{
            "task_key": "run_tests",
            "spark_python_task": {
                "python_file": "dbfs:/FileStore/burning-cost/insurance-deploy-tests.py",
            },
            "new_cluster": {
                "spark_version": "15.4.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 0,
                "spark_conf": {"spark.databricks.cluster.profile": "singleNode"},
                "custom_tags": {"ResourceClass": "SingleNode"},
            },
        }],
    }

    resp = api_call("POST", "jobs/runs/submit", payload)
    run_id = resp["run_id"]
    print(f"Job submitted, run_id={run_id}")

    # Poll
    for attempt in range(60):
        time.sleep(10)
        status = api_call("GET", f"jobs/runs/get?run_id={run_id}")
        state = status["state"]
        lcs = state.get("life_cycle_state", "")
        rs = state.get("result_state", "")
        print(f"  [{attempt * 10}s] {lcs} / {rs}")
        if lcs in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break

    # Get output
    try:
        output = api_call("GET", f"jobs/runs/get-output?run_id={run_id}")
        logs = output.get("logs", "")
        if logs:
            print("\n--- TEST OUTPUT ---")
            print(logs[-5000:])  # last 5000 chars
        error = output.get("error", "")
        if error:
            print(f"\nError: {error}")
    except Exception as e:
        print(f"Could not fetch output: {e}")

    return rs == "SUCCESS"


if __name__ == "__main__":
    success = run_via_script_job()
    print(f"\nTest run {'SUCCEEDED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
