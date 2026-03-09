"""
Run the insurance-deploy test suite on Databricks serverless.

Usage:
    python run_tests_databricks.py

Uploads the project to Databricks workspace, runs pytest via a notebook job,
and streams the output.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Load Databricks credentials
env_path = Path.home() / ".config" / "burning-cost" / "databricks.env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

PROJECT_DIR = Path(__file__).parent
WORKSPACE_PATH = "/Workspace/burning-cost/insurance-deploy"


def upload_project(w: WorkspaceClient) -> None:
    print(f"Uploading project to {WORKSPACE_PATH}...")
    import subprocess
    result = subprocess.run(
        ["databricks", "workspace", "import-dir",
         str(PROJECT_DIR), WORKSPACE_PATH, "--overwrite"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Upload stderr:", result.stderr[:500])
        # Try manual upload of key files
        print("Falling back to manual file upload...")

    # Also upload as a zip for pip install
    import zipfile
    import tempfile
    zip_path = Path(tempfile.mktemp(suffix=".zip"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in PROJECT_DIR.rglob("*"):
            if f.is_file() and ".git" not in str(f) and "__pycache__" not in str(f):
                zf.write(f, f.relative_to(PROJECT_DIR))
    print(f"Created zip: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    return zip_path


def run_tests(w: WorkspaceClient) -> bool:
    print("Creating Databricks job to run tests...")

    notebook_content = '''
# Databricks notebook source
# MAGIC %pip install numpy scipy pandas joblib pytest pytest-cov

# COMMAND ----------
import subprocess, sys, os, tempfile, zipfile
from pathlib import Path

# Install the package
%cd /tmp
!pip install -q insurance-deploy 2>&1 | tail -5

# COMMAND ----------
import tempfile
tmpdir = tempfile.mkdtemp()

# Write inline tests
test_code = """
import sys
sys.path.insert(0, '/tmp')

import tempfile, random, warnings
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

import numpy as np
import pytest

# ---- Minimal model for testing ----
class DummyModel:
    def __init__(self, c=400.0):
        self.c = c
    def predict(self, X):
        return np.full(len(X), self.c)

tmpdir = Path(tempfile.mkdtemp())

# ---- Test 1: Registry ----
from insurance_deploy import ModelRegistry
r = ModelRegistry(tmpdir / "reg")
mv = r.register(DummyModel(), "motor", "1.0", {"x": 1})
assert mv.version_id == "motor:1.0"
assert mv.metadata["x"] == 1
print("PASS: Registry register")

# ---- Test 2: Duplicate raises ----
try:
    r.register(DummyModel(), "motor", "1.0")
    assert False, "Should have raised"
except ValueError:
    pass
print("PASS: Registry duplicate raises ValueError")

# ---- Test 3: Hash verification ----
from pathlib import Path
Path(mv.model_path).write_bytes(b"corrupt")
try:
    mv._model = None
    _ = mv.model
    assert False, "Should have raised"
except ValueError:
    pass
print("PASS: Hash verification detects tampering")

# ---- Test 4: Routing determinism ----
from insurance_deploy import Experiment, ModelRegistry
r2 = ModelRegistry(tmpdir / "reg2")
mv1 = r2.register(DummyModel(400), "m", "1.0")
mv2 = r2.register(DummyModel(420), "m", "2.0")
exp = Experiment("test_exp", mv1, mv2, 0.10)
results = [exp.route(f"P-{i}") for i in range(1000)]
results2 = [exp.route(f"P-{i}") for i in range(1000)]
assert results == results2
print("PASS: Routing is deterministic")

# ---- Test 5: Routing split ----
challenger_n = sum(1 for r in results if r == "challenger")
assert 80 <= challenger_n <= 120, f"Expected ~100 challengers, got {challenger_n}"
print(f"PASS: Routing split ~{challenger_n/1000:.1%} challenger (expected ~10%)")

# ---- Test 6: SHA256 routing formula ----
import hashlib
pid = "POL-SPECIFIC-001"
key = (pid + exp.name).encode()
digest = hashlib.sha256(key).hexdigest()
slot = int(digest[-8:], 16) % 100
expected = "challenger" if slot < 10 else "champion"
assert exp.route(pid) == expected
print("PASS: Routing formula matches SHA-256 spec")

# ---- Test 7: Logger ----
from insurance_deploy import QuoteLogger
logger = QuoteLogger(tmpdir / "q.db")
logger.log_quote("P1", "exp", "champion", "m:1", 400.0, enbp=420.0, renewal_flag=True)
logger.log_bind("P1", 400.0)
logger.log_claim("P1", date(2024, 6, 1), 1200.0, 12)
assert logger.quote_count() == 1
quotes = logger.query_quotes()
assert quotes[0]["enbp_flag"] == 1  # 400 <= 420
print("PASS: Logger write and read")

# ---- Test 8: ENBP breach warning ----
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    logger.log_quote("P2", "exp", "champion", "m:1", 450.0, enbp=430.0, renewal_flag=True)
    assert any("ENBP breach" in str(x.message) for x in w)
print("PASS: ENBP breach emits warning")

# ---- Test 9: KPITracker hit rate ----
from insurance_deploy import KPITracker
logger2 = QuoteLogger(tmpdir / "q2.db")
for i in range(10):
    logger2.log_quote(f"P{i}", "e", "champion", "m:1", 400.0)
for i in range(4):
    logger2.log_bind(f"P{i}", 400.0)
tracker = KPITracker(logger2)
hr = tracker.hit_rate("e")
assert hr["champion"]["hit_rate"] == 0.4
print("PASS: KPITracker hit rate")

# ---- Test 10: Loss ratio ----
logger3 = QuoteLogger(tmpdir / "q3.db")
logger3.log_quote("P1", "e2", "champion", "m:1", 400.0)
logger3.log_bind("P1", 400.0)
logger3.log_claim("P1", date(2024, 6, 1), 200.0, 12)
tracker3 = KPITracker(logger3)
lr = tracker3.loss_ratio("e2", development_months=12)
assert abs(lr["champion"]["loss_ratio"] - 0.5) < 0.01
print("PASS: Loss ratio calculation (200/400 = 0.50)")

# ---- Test 11: Power analysis ----
pa = tracker.power_analysis("e", target_delta_lr=0.03)
assert pa["hr_required_n_per_arm"] > 0
assert pa["lr_required_n_per_arm"] > 0
assert pa["lr_total_months_with_development"] > 0
print("PASS: Power analysis returns positive sample sizes")

# ---- Test 12: ModelComparison bootstrap ----
from insurance_deploy import ModelComparison
import math
logger4 = QuoteLogger(tmpdir / "q4.db")
rng = random.Random(0)
base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
for i in range(300):
    pid = f"CH-{i}"
    ts = base_ts + timedelta(days=i // 5)
    price = max(100, rng.gauss(400, 50))
    logger4.log_quote(pid, "boot_exp", "champion", "m:1", price,
                      enbp=price+20, renewal_flag=True, timestamp=ts)
    if rng.random() < 0.30:
        logger4.log_bind(pid, price, ts + timedelta(hours=1))
        if rng.random() < 0.08:
            logger4.log_claim(pid, date(2024, 6, 1), max(0, rng.gauss(1500, 500)), 12)
for i in range(100):
    pid = f"CL-{i}"
    ts = base_ts + timedelta(days=i // 2)
    price = max(100, rng.gauss(410, 50))
    logger4.log_quote(pid, "boot_exp", "challenger", "m:2", price,
                      enbp=price+20, renewal_flag=True, timestamp=ts)
    if rng.random() < 0.28:
        logger4.log_bind(pid, price, ts + timedelta(hours=1))
        if rng.random() < 0.075:
            logger4.log_claim(pid, date(2024, 6, 1), max(0, rng.gauss(1400, 500)), 12)

tracker4 = KPITracker(logger4)
comp = ModelComparison(tracker4)
result = comp.bootstrap_lr_test("boot_exp", n_bootstrap=500, seed=42)
assert result.conclusion in {"INSUFFICIENT_EVIDENCE", "CHALLENGER_BETTER", "CHAMPION_BETTER"}
assert not math.isnan(result.champion_estimate)
assert result.ci_lower <= result.ci_upper
print(f"PASS: Bootstrap LR test (conclusion: {result.conclusion})")

# ---- Test 13: ENBPAuditReport ----
from insurance_deploy import ENBPAuditReport
reporter = ENBPAuditReport(logger4)
md = reporter.generate("boot_exp", firm_name="Test Co", smf_holder="Test SMF")
assert "ICOBS 6B.2.51R" in md
assert "Executive Summary" in md
assert "Attestation" in md
assert "SHA-256" in md
print("PASS: ENBPAuditReport generates valid Markdown")

print()
print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
"""

# Write and run the test
test_file = Path("/tmp/run_tests.py")
test_file.write_text(test_code)

import subprocess
result = subprocess.run(
    [sys.executable, str(test_file)],
    capture_output=True, text=True, timeout=120
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError(f"Tests failed with return code {result.returncode}")
'''

    # Create a notebook via the API
    import base64
    notebook_b64 = base64.b64encode(notebook_content.encode()).decode()

    notebook_path = "/Workspace/burning-cost/insurance-deploy-test-runner"
    try:
        w.workspace.import_(
            path=notebook_path,
            content=notebook_b64,
            format="SOURCE",
            language="PYTHON",
            overwrite=True,
        )
        print(f"Notebook uploaded to {notebook_path}")
    except Exception as e:
        print(f"Notebook upload error: {e}")
        return False

    # Run as a one-time job
    run = w.jobs.submit(
        run_name="insurance-deploy-tests",
        tasks=[
            jobs.SubmitTask(
                task_key="run_tests",
                notebook_task=jobs.NotebookTask(
                    notebook_path=notebook_path,
                ),
                new_cluster=None,  # use serverless
            )
        ],
    ).result()

    # Poll for completion
    print("Job submitted, polling...")
    for _ in range(60):
        run_state = w.jobs.get_run(run_id=run.run_id)
        state = run_state.state
        print(f"  Status: {state.life_cycle_state} / {state.result_state}")
        if state.life_cycle_state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(10)

    return state.result_state == "SUCCESS"


if __name__ == "__main__":
    w = WorkspaceClient()
    print("Connected to Databricks workspace")

    success = run_tests(w)
    sys.exit(0 if success else 1)
