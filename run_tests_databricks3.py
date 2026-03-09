"""Run insurance-deploy tests on Databricks using workspace files + notebook task."""

from __future__ import annotations

import os
import sys
import time
import json
import base64
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
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}


def api(method: str, endpoint: str, payload: dict = None, version="2.0") -> dict:
    url = f"{HOST}/api/{version}/{endpoint}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code} on {endpoint}: {body[:800]}")
        raise


NOTEBOOK_CONTENT = '''# Databricks notebook source
# MAGIC %pip install -q 'insurance-deploy==0.1.1'

# COMMAND ----------

import sys, tempfile, random, warnings, math, hashlib
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
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
    else:
        failed.append(name)
        print(f"FAIL: {name} {detail}")

# === REGISTRY TESTS ===
r = ModelRegistry(tmp / "reg")
mv1 = r.register(DummyModel(400), "motor", "1.0", {"tag": "v1"})
mv2 = r.register(DummyModel(420), "motor", "2.0", {"tag": "v2"})
check("registry_register", mv1.version_id == "motor:1.0")
check("registry_metadata", mv1.metadata["tag"] == "v1")
check("registry_list_all", len(r.list()) == 2)
check("registry_list_filtered", len(r.list("motor")) == 2)
check("registry_get", r.get("motor", "1.0").version == "1.0")
check("registry_champion_default", r.champion("motor").version == "2.0")
r.set_champion("motor", "1.0")
check("registry_set_champion", r.champion("motor").version == "1.0")
check("registry_is_champion_flag", r.get("motor", "1.0").is_champion is True)
check("registry_other_not_champion", r.get("motor", "2.0").is_champion is False)

try:
    r.register(DummyModel(), "motor", "1.0")
    check("registry_duplicate_raises", False)
except ValueError:
    check("registry_duplicate_raises", True)

try:
    r.get("motor", "99.0")
    check("registry_get_missing_raises", False)
except KeyError:
    check("registry_get_missing_raises", True)

try:
    r.champion("nonexistent")
    check("registry_champion_missing_raises", False)
except KeyError:
    check("registry_champion_missing_raises", True)

try:
    r.set_champion("motor", "99.0")
    check("registry_set_champion_missing_raises", False)
except KeyError:
    check("registry_set_champion_missing_raises", True)

# Persistence
r2 = ModelRegistry(tmp / "reg")
check("registry_persistence_count", len(r2.list()) == 2)
check("registry_persistence_champion", r2.champion("motor").version == "1.0")

# Hash verification
Path(mv1.model_path).write_bytes(b"tampered data")
mv1_r2 = r2.get("motor", "1.0")
mv1_r2._model = None
try:
    _ = mv1_r2.model
    check("registry_hash_verification", False)
except ValueError:
    check("registry_hash_verification", True)

# === EXPERIMENT ROUTING TESTS ===
r3 = ModelRegistry(tmp / "reg3")
c1 = r3.register(DummyModel(400), "m", "1.0")
c2 = r3.register(DummyModel(420), "m", "2.0")
exp = Experiment("test_exp_det", c1, c2, 0.10)

# Determinism
results_a = [exp.route(f"P-{i}") for i in range(2000)]
results_b = [exp.route(f"P-{i}") for i in range(2000)]
check("routing_deterministic", results_a == results_b)

# SHA-256 formula
pid = "POL-FORMULA-CHECK"
key = (pid + exp.name).encode()
digest = hashlib.sha256(key).hexdigest()
slot = int(digest[-8:], 16) % 100
expected = "challenger" if slot < 10 else "champion"
check("routing_sha256_formula", exp.route(pid) == expected)

# Approximate split (10% challenger)
n_chall = sum(1 for r in results_a if r == "challenger")
check("routing_split_approx", 150 <= n_chall <= 250, f"got {n_chall}")

# Different names -> different routing
exp2 = Experiment("other_name", c1, c2, 0.10)
routes1 = [exp.route(f"Q-{i}") for i in range(500)]
routes2 = [exp2.route(f"Q-{i}") for i in range(500)]
check("routing_different_names", routes1 != routes2)

# Shadow mode: live_model always champion
for pid_t in [f"S-{i}" for i in range(10)]:
    check(f"shadow_live_model", exp.live_model(pid_t).version_id == c1.version_id)

# Deactivation
exp.deactivate()
check("experiment_deactivated", not exp.is_active())
try:
    exp.route("P-0")
    check("routing_deactivated_raises", False)
except RuntimeError:
    check("routing_deactivated_raises", True)

# Validation
try:
    Experiment("bad", c1, c2, challenger_pct=0.0)
    check("experiment_invalid_pct_zero", False)
except ValueError:
    check("experiment_invalid_pct_zero", True)

try:
    Experiment("bad", c1, c2, challenger_pct=1.0)
    check("experiment_invalid_pct_one", False)
except ValueError:
    check("experiment_invalid_pct_one", True)

try:
    Experiment("bad", c1, c2, mode="bandit")
    check("experiment_invalid_mode", False)
except ValueError:
    check("experiment_invalid_mode", True)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    Experiment("live_test", c1, c2, mode="live")
    check("experiment_live_warning", any("Consumer Duty" in str(x.message) for x in w))

# === LOGGER TESTS ===
log = QuoteLogger(tmp / "q.db")
log.log_quote("P1", "e", "champion", "m:1", 400.0, enbp=420.0, renewal_flag=True)
log.log_bind("P1", 400.0)
log.log_claim("P1", date(2024, 6, 1), 1200.0, 12)
check("logger_quote_count", log.quote_count() == 1)
q = log.query_quotes()[0]
check("logger_enbp_compliant_flag", q["enbp_flag"] == 1)
check("logger_quote_policy_id", q["policy_id"] == "P1")
check("logger_quote_arm", q["arm"] == "champion")
check("logger_bind_count", len(log.query_binds()) == 1)
check("logger_claim_count", len(log.query_claims()) == 1)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    log.log_quote("P2", "e", "champion", "m:1", 450.0, enbp=430.0, renewal_flag=True)
    check("logger_breach_warning", any("ENBP breach" in str(x.message) for x in w))
q2 = [q for q in log.query_quotes() if q["policy_id"] == "P2"][0]
check("logger_breach_flag", q2["enbp_flag"] == 0)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    log.log_quote("P3", "e", "champion", "m:1", 400.0, renewal_flag=True, enbp=None)
    check("logger_missing_enbp_warning", any("no enbp provided" in str(x.message) for x in w))

q3 = [q for q in log.query_quotes() if q["policy_id"] == "P3"][0]
check("logger_missing_enbp_flag_null", q3["enbp_flag"] is None)

nb_log = QuoteLogger(tmp / "nb.db")
nb_log.log_quote("P1", "e", "champion", "m:1", 400.0, renewal_flag=False)
qnb = nb_log.query_quotes()[0]
check("logger_nb_enbp_flag_null", qnb["enbp_flag"] is None)

try:
    log.log_quote("PX", "e", "neutral", "m:1", 400.0)
    check("logger_invalid_arm", False)
except ValueError:
    check("logger_invalid_arm", True)

try:
    log.log_quote("PX", "e", "champion", "m:1", -100.0)
    check("logger_negative_price", False)
except ValueError:
    check("logger_negative_price", True)

try:
    log.log_quote("PX", "e", "champion", "m:1", 400.0, exposure=0.0)
    check("logger_zero_exposure", False)
except ValueError:
    check("logger_zero_exposure", True)

try:
    log.log_claim("PX", date(2024, 6, 1), -50.0, 3)
    check("logger_negative_claim", False)
except ValueError:
    check("logger_negative_claim", True)

try:
    log.log_bind("PX", -10.0)
    check("logger_negative_bind", False)
except ValueError:
    check("logger_negative_bind", True)

check("logger_no_delete", not hasattr(log, "delete_quote"))

# Filter by experiment
log.log_quote("X1", "other_exp", "champion", "m:1", 400.0)
check("logger_filter_exp", log.quote_count("e") == 3)  # P1, P2, P3

# === KPI TRACKER TESTS ===
log2 = QuoteLogger(tmp / "q2.db")
for i in range(10):
    log2.log_quote(f"P{i}", "kpi_e", "champion", "m:1", 400.0 + i)
for i in range(3):
    log2.log_bind(f"P{i}", 400.0 + i)
t2 = KPITracker(log2)

hr = t2.hit_rate("kpi_e")
check("kpi_hr_quoted", hr["champion"]["quoted"] == 10)
check("kpi_hr_bound", hr["champion"]["bound"] == 3)
check("kpi_hr_rate", abs(hr["champion"]["hit_rate"] - 0.30) < 0.01)

vol = t2.quote_volume("kpi_e")
check("kpi_vol_n", vol["champion"]["n"] == 10)
check("kpi_vol_mean", abs(vol["champion"]["mean_price"] - 404.5) < 0.1)

gwp = t2.gwp("kpi_e")
check("kpi_gwp_total", abs(gwp["champion"]["total_gwp"] - (400 + 401 + 402)) < 1)
check("kpi_gwp_bound_n", gwp["champion"]["bound_policies"] == 3)

# Loss ratio calculation
log3 = QuoteLogger(tmp / "q3.db")
log3.log_quote("P1", "lr_e", "champion", "m:1", 400.0)
log3.log_bind("P1", 400.0)
log3.log_claim("P1", date(2024, 6, 1), 200.0, 12)
t3 = KPITracker(log3)
lr = t3.loss_ratio("lr_e", development_months=12)
check("kpi_lr_value", abs(lr["champion"]["loss_ratio"] - 0.5) < 0.01)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    t3.loss_ratio("lr_e", development_months=6)
    check("kpi_lr_immature_warning", any("immature" in str(x.message).lower() for x in w))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    t2.frequency("kpi_e", development_months=3)
    check("kpi_freq_immature_warning", any("IBNR" in str(x.message) for x in w))

# Power analysis
pa = t2.power_analysis("kpi_e", target_delta_lr=0.03)
check("kpi_pa_hr_n_positive", pa["hr_required_n_per_arm"] > 0)
check("kpi_pa_lr_n_positive", pa["lr_required_n_per_arm"] > 0)
check("kpi_pa_notes", len(pa["notes"]) > 0)
pa_lg = t2.power_analysis("kpi_e", target_delta_lr=0.05)
check("kpi_pa_larger_delta_fewer", pa["lr_required_n_per_arm"] > pa_lg["lr_required_n_per_arm"])

# === BOOTSTRAP LR TEST ===
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
check("boot_valid_conclusion", result.conclusion in {"INSUFFICIENT_EVIDENCE", "CHALLENGER_BETTER", "CHAMPION_BETTER"})
check("boot_not_nan", not math.isnan(result.champion_estimate))
check("boot_ci_ordered", result.ci_lower <= result.ci_upper)
check("boot_p_value_range", 0 < result.p_value <= 1.0)
check("boot_ci_contains_diff", result.ci_lower <= result.difference <= result.ci_upper)
check("boot_n_positive", result.n_champion > 0 and result.n_challenger > 0)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    r_immature = comp.bootstrap_lr_test("bt", n_bootstrap=100, development_months=6, seed=0)
    check("boot_maturity_warning", any("development_months" in str(x.message) for x in w))
check("boot_maturity_flag", r_immature.maturity_warning is True)

r1s = comp.bootstrap_lr_test("bt", n_bootstrap=200, seed=99)
r2s = comp.bootstrap_lr_test("bt", n_bootstrap=200, seed=99)
check("boot_seed_reproducible", abs(r1s.p_value - r2s.p_value) < 1e-10)

r_empty = comp.bootstrap_lr_test("nonexistent", n_bootstrap=10)
check("boot_empty_insufficient", r_empty.conclusion == "INSUFFICIENT_EVIDENCE")

# Hit rate test
hr_result = comp.hit_rate_test("bt")
check("hitrate_test_name", hr_result.test_name == "hit_rate_test")
check("hitrate_diff_consistent", abs(hr_result.difference - (hr_result.challenger_estimate - hr_result.champion_estimate)) < 1e-6)
check("hitrate_ci_ordered", math.isnan(hr_result.ci_lower) or hr_result.ci_lower < hr_result.ci_upper)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    freq_result = comp.frequency_test("bt", development_months=12)
check("freq_test_name", freq_result.test_name == "frequency_test")
check("freq_valid_conclusion", freq_result.conclusion in {"INSUFFICIENT_EVIDENCE", "CHALLENGER_BETTER", "CHAMPION_BETTER"})

# Large hit rate difference detection
log5 = QuoteLogger(tmp / "q5.db")
rng3 = random.Random(11)
for i in range(600):
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
    if rng3.random() < 0.27:
        log5.log_bind(pid, p, ts + timedelta(hours=1))
t5 = KPITracker(log5)
c5 = ModelComparison(t5)
hr_big = c5.hit_rate_test("big_diff")
check("hitrate_detects_large", hr_big.conclusion != "INSUFFICIENT_EVIDENCE")

# ComparisonResult features
check("result_summary_string", isinstance(result.summary(), str))
check("result_summary_test_name", "bootstrap_lr_test" in result.summary())
check("result_repr_test_name", "bootstrap_lr_test" in repr(result))

# === ENBP AUDIT REPORT ===
reporter = ENBPAuditReport(log4)
md = reporter.generate("bt", firm_name="Test Co Ltd", smf_holder="J Smith")
check("audit_is_string", isinstance(md, str))
check("audit_icobs", "ICOBS 6B.2.51R" in md)
check("audit_exec_summary", "Executive Summary" in md)
check("audit_model_versions", "Model Versions" in md)
check("audit_routing_audit", "Routing Decision Audit" in md)
check("audit_attestation", "Attestation Statement" in md)
check("audit_firm_name", "Test Co Ltd" in md)
check("audit_smf", "J Smith" in md)
check("audit_sha256", "SHA-256" in md)
check("audit_deterministic", "deterministic" in md)
check("audit_all_data_label", "All available data" in md)

md_period = reporter.generate("bt", period_start="2024-01-01", period_end="2024-12-31")
check("audit_period_label", "2024-01-01 to 2024-12-31" in md_period)

md_from = reporter.generate("bt", period_start="2024-06-01")
check("audit_from_label", "from 2024-06-01" in md_from)

md_to = reporter.generate("bt", period_end="2024-09-30")
check("audit_to_label", "to 2024-09-30" in md_to)

# No breaches
log6 = QuoteLogger(tmp / "q6.db")
for i in range(10):
    log6.log_quote(f"R{i}", "clean", "champion", "m:1", 400.0, enbp=420.0, renewal_flag=True)
reporter6 = ENBPAuditReport(log6)
md6 = reporter6.generate("clean")
check("audit_no_breach_msg", "No ENBP breaches" in md6)

# Breach case
log7 = QuoteLogger(tmp / "q7.db")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    log7.log_quote("BR1", "breach_exp", "champion", "m:1", 450.0, enbp=430.0, renewal_flag=True)
reporter7 = ENBPAuditReport(log7)
md7 = reporter7.generate("breach_exp")
check("audit_breach_detail", "ENBP Breach Detail" in md7)
check("audit_breach_action", "Action required" in md7)
check("audit_breach_policy_id", "BR1" in md7)

# Empty experiment
reporter_empty = ENBPAuditReport(QuoteLogger(tmp / "empty.db"))
md_empty = reporter_empty.generate("empty")
check("audit_no_renewal_msg", "No renewal quotes recorded" in md_empty)

# Generated timestamp
check("audit_generated_ts", "Generated:" in md)
check("audit_utc", "UTC" in md)

# === RESULTS ===
print()
print("=" * 60)
print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
if failed:
    print(f"FAILED: {failed}")
else:
    print("ALL TESTS PASSED")
print("=" * 60)

assert not failed, f"Tests failed: {failed}"
'''


def main():
    # Upload notebook to workspace
    workspace_path = "/insurance-deploy/test-runner"
    print(f"Uploading test notebook to {workspace_path}...")

    encoded = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
    api("POST", "workspace/import", {
        "path": workspace_path,
        "content": encoded,
        "format": "SOURCE",
        "language": "PYTHON",
        "overwrite": True,
    })
    print("Notebook uploaded")

    # Submit as a job run
    print("Submitting job...")
    payload = {
        "run_name": "insurance-deploy-tests",
        "tasks": [{
            "task_key": "run_tests",
            "notebook_task": {
                "notebook_path": workspace_path,
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

    resp = api("POST", "jobs/runs/submit", payload)
    run_id = resp["run_id"]
    print(f"Job submitted: run_id={run_id}")

    for i in range(90):
        time.sleep(15)
        status = api("GET", f"jobs/runs/get?run_id={run_id}")
        state = status["state"]
        lcs = state.get("life_cycle_state", "")
        rs = state.get("result_state", "")
        elapsed = (i + 1) * 15
        print(f"  [{elapsed}s] {lcs} / {rs}")
        if lcs in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break

    # Get output
    try:
        output = api("GET", f"jobs/runs/get-output?run_id={run_id}")
        notebook_result = output.get("notebook_output", {})
        result_text = notebook_result.get("result", "")
        if result_text:
            print("\n--- NOTEBOOK OUTPUT ---")
            print(result_text[-4000:])
        error = output.get("error", "")
        if error:
            print(f"\nError: {error}")
        # Also try logs
        logs = output.get("logs", "")
        if logs:
            print("\n--- LOGS ---")
            print(logs[-3000:])
    except Exception as e:
        print(f"Could not fetch output: {e}")

    succeeded = rs == "SUCCESS"
    print(f"\nTest run: {'SUCCEEDED' if succeeded else 'FAILED'}")
    return succeeded


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
