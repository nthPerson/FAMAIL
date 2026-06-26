# Provenance & Reproducibility Bundle (Plan 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give every re-run artifact a provenance trail — a per-run manifest (git SHA, argv, seeds, edit-dir, env, determinism flags), on-disk per-stage timings, source-data byte fingerprints, and a navigable bundle index — so every headline number binds to an exact code state + dataset, and the methods/repro section writes itself.

**Architecture:** One shared helper module `baselines/_manifest.py` (no heavy imports) called at the top of each runner's `main()`; a one-shot env capture; a tiny timings appender; a data-SHA addition to the source-gen writer; and scaffolding for a figures manifest + run README. All non-GPU; piggybacks on already-scheduled runs.

**Tech Stack:** Python 3.12, stdlib (json, hashlib, subprocess, platform, datetime), torch/numpy/pandas (version probe only), pytest.

**Spec:** `docs/superpowers/specs/2026-06-25-data-cleanup-rerun-design.md` §6.1 (E19, E20, E21, E29, E30, E37, E38, E39). Branch: `data-cleanup-rerun`.

## Global Constraints
- **TDD** every helper. Pure/deterministic where possible; inject clock + git/env probes so tests don't depend on the real environment.
- **No heavy imports at module top** of `_manifest.py` (it's imported by every runner; keep torch/pandas imports lazy/inside functions so a probe failure never breaks a run).
- **Never let provenance crash a run:** every probe (git, pip freeze, torch) is wrapped — on failure it records `"unknown"`/`null`, not an exception.
- **`default=str`** on every `json.dumps` (matches the source-gen writer; lossless for numpy scalars).
- E22 (cleanup_delta) is DEFERRED to Plan 5 (needs dirty+clean experiment outputs that don't exist yet).

---

## File Structure
- **Create** `famail_temporal/baselines/_manifest.py` — `write_run_manifest`, `capture_env`, `append_timing`, `git_sha`, `sha256_file`.
- **Create** `famail_temporal/baselines/tests/test_manifest.py`.
- **Modify** `famail_temporal/data/source_generation/writer.py` — add `data_sha256` of the written outputs into `processing_metadata.json` (E30).
- **Modify** the 5 baseline runners' `main()` (`run_level1_table_v2.py`, `run_level2_table.py`, `run_weighted_bc_smoke.py`, `run_variance_suite.py`, `run_data_pareto.py`) — call `write_run_manifest(out_dir, ...)` + `append_timing` (Task 4).
- **Create** `famail_temporal/baselines/_bundle_index.py` — `register_figure(...)` (FIGURES_MANIFEST) + `write_rerun_readme(...)` scaffolding (E37, E39).

---

### Task 1: Manifest + env capture helper

**Files:**
- Create: `famail_temporal/baselines/_manifest.py`
- Test: `famail_temporal/baselines/tests/test_manifest.py`

**Interfaces:**
- Produces: `git_sha() -> tuple[str,bool]` (short sha, dirty?); `capture_env() -> dict` (python, torch, cuda, gpu_name, numpy, pandas, cudnn flags — each guarded, `"unknown"` on failure); `write_run_manifest(out_dir, *, argv, seeds, edit_dir, extra=None, now=None) -> Path` writing `out_dir/manifest.json` with `{git_sha, git_dirty, argv, seeds, edit_dir, hostname, timestamp_utc, env, **(extra or {})}`; `now` injectable for tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest.py
import json
from pathlib import Path
from famail_temporal.baselines import _manifest


def test_write_run_manifest_captures_core_fields(tmp_path):
    out = _manifest.write_run_manifest(
        tmp_path, argv=["prog", "--seeds", "0,1"], seeds=[0, 1],
        edit_dir="results/edit_X", extra={"discriminator_sha256": "abc", "gate_matched": 0.84},
        now="2026-06-26T00:00:00Z",
    )
    assert out == tmp_path / "manifest.json"
    m = json.loads(out.read_text())
    assert m["argv"] == ["prog", "--seeds", "0,1"]
    assert m["seeds"] == [0, 1]
    assert m["edit_dir"] == "results/edit_X"
    assert m["timestamp_utc"] == "2026-06-26T00:00:00Z"
    assert m["discriminator_sha256"] == "abc"        # extra merged
    assert "git_sha" in m and "git_dirty" in m
    assert "env" in m and "python" in m["env"]
    assert "hostname" in m


def test_capture_env_never_raises_and_has_keys():
    env = _manifest.capture_env()
    for k in ("python", "torch", "cuda", "gpu_name", "numpy", "pandas",
              "cudnn_deterministic", "cudnn_benchmark"):
        assert k in env  # value may be "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_manifest.py -q`
Expected: FAIL — `ModuleNotFoundError: ... _manifest`.

- [ ] **Step 3: Write minimal implementation**

```python
# _manifest.py
"""Per-run provenance: a manifest.json + env capture for every re-run artifact.
Kept import-light and crash-proof — provenance must never break a run."""
from __future__ import annotations
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_sha() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return sha, dirty
    except Exception:
        return "unknown", False


def capture_env() -> dict:
    env = {"python": platform.python_version(), "torch": "unknown", "cuda": "unknown",
           "gpu_name": "unknown", "numpy": "unknown", "pandas": "unknown",
           "cudnn_deterministic": "unknown", "cudnn_benchmark": "unknown"}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = getattr(torch.version, "cuda", "unknown")
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
        env["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        env["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:
        pass
    for mod, key in (("numpy", "numpy"), ("pandas", "pandas")):
        try:
            env[key] = __import__(mod).__version__
        except Exception:
            pass
    return env


def write_run_manifest(out_dir, *, argv, seeds, edit_dir, extra=None, now=None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha, dirty = git_sha()
    manifest = {
        "timestamp_utc": now or datetime.now(timezone.utc).isoformat(),
        "git_sha": sha, "git_dirty": dirty,
        "argv": list(argv), "seeds": list(seeds) if seeds is not None else None,
        "edit_dir": str(edit_dir) if edit_dir is not None else None,
        "hostname": socket.gethostname(), "env": capture_env(),
    }
    if extra:
        manifest.update(extra)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_manifest.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/_manifest.py famail_temporal/baselines/tests/test_manifest.py
git commit -m "feat(baselines): per-run provenance manifest + env capture (E19/E20/E38)"
```

---

### Task 2: Per-stage timings appender + file SHA

**Files:**
- Modify: `famail_temporal/baselines/_manifest.py`
- Test: `famail_temporal/baselines/tests/test_manifest.py`

**Interfaces:**
- Produces: `append_timing(path, stage, seconds, *, now=None) -> None` (appends one JSON line `{stage, seconds, timestamp}` to `path`, creating it); `sha256_file(path) -> str` (streaming SHA-256, `"missing"` if absent).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_manifest.py
def test_append_timing_writes_jsonl(tmp_path):
    p = tmp_path / "timings.jsonl"
    _manifest.append_timing(p, "stage1", 12.5, now="t1")
    _manifest.append_timing(p, "stage2", 3.0, now="t2")
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    assert lines == [
        {"stage": "stage1", "seconds": 12.5, "timestamp": "t1"},
        {"stage": "stage2", "seconds": 3.0, "timestamp": "t2"},
    ]


def test_sha256_file_stable_and_missing(tmp_path):
    f = tmp_path / "a.bin"; f.write_bytes(b"hello")
    h1 = _manifest.sha256_file(f); h2 = _manifest.sha256_file(f)
    assert h1 == h2 and len(h1) == 64
    assert _manifest.sha256_file(tmp_path / "nope.bin") == "missing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_manifest.py -k "timing or sha256" -q`
Expected: FAIL — `AttributeError: ... append_timing`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to _manifest.py
import hashlib

def append_timing(path, stage, seconds, *, now=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"stage": stage, "seconds": float(seconds),
           "timestamp": now or datetime.now(timezone.utc).isoformat()}
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def sha256_file(path) -> str:
    path = Path(path)
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_manifest.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/_manifest.py famail_temporal/baselines/tests/test_manifest.py
git commit -m "feat(baselines): timings.jsonl appender + streaming file SHA-256 (E21/E30)"
```

---

### Task 3: Source-data byte fingerprint in processing_metadata (E30)

**Files:**
- Modify: `famail_temporal/data/source_generation/writer.py`
- Test: `famail_temporal/data/source_generation/tests/test_writer_sha.py` (create)

**Interfaces:**
- Produces: after writing all outputs, `write_all_outputs` adds `metadata["data_sha256"] = {filename: sha256, ...}` for the written `.pkl` outputs (a dict), so a reproducer can verify byte-identical regeneration.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_writer_sha.py
import json, hashlib
from pathlib import Path
from famail_temporal.data.source_generation import writer as W


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_metadata_records_data_sha256(tmp_path):
    # minimal valid inputs for write_all_outputs (mirror an existing writer test's fixtures)
    # ... build the smallest payloads write_all_outputs accepts ...
    paths = W.write_all_outputs(out_dir=tmp_path, **_minimal_payloads())  # helper below
    meta = json.loads((tmp_path / "processing_metadata.json").read_text())
    assert "data_sha256" in meta
    # every recorded sha matches the file on disk
    for name, sha in meta["data_sha256"].items():
        assert _sha(tmp_path / name) == sha
    assert "pickup_dropoff_counts.pkl" in meta["data_sha256"]
```

*(Implementer: reuse the existing writer test's fixture builder for `_minimal_payloads`; if none exists, construct the smallest dicts `write_all_outputs` accepts by reading its signature. ASK if the payload shape is unclear rather than guess.)*

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_writer_sha.py -q`
Expected: FAIL — `KeyError: 'data_sha256'`.

- [ ] **Step 3: Write minimal implementation**

In `writer.py`, after all `.pkl` files are written and before writing `processing_metadata.json`, compute `data_sha256 = {p.name: sha256_file(p) for p in written_pkl_paths}` (import `sha256_file` from `famail_temporal.baselines._manifest`, or inline a small hashlib helper to avoid a cross-package import — implementer's call; prefer inline to keep source_generation independent of baselines). Add it to the metadata dict. Compute SHAs only for the data `.pkl` outputs (not the metadata json itself).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/data/source_generation/tests/test_writer_sha.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/data/source_generation/writer.py famail_temporal/data/source_generation/tests/test_writer_sha.py
git commit -m "feat(source-gen): record data_sha256 of outputs in processing_metadata (E30)"
```

---

### Task 4: Wire manifest + timings into the baseline runners

**Files:**
- Modify: `run_level1_table_v2.py`, `run_level2_table.py`, `run_weighted_bc_smoke.py`, `run_variance_suite.py`, `run_data_pareto.py`
- Test: `famail_temporal/baselines/tests/test_runner_manifest_smoke.py` (create)

**Interfaces:**
- Consumes: `write_run_manifest`, `append_timing` (Tasks 1–2).
- Produces: each runner, at the end of a successful `main()`, writes `manifest.json` into its out-dir with `argv=sys.argv`, the parsed `seeds`, `edit_dir`, and (for L1-v2/L2/weighted-BC, which compute the validation gate) `extra={"discriminator_sha256": sha256_file(ckpt), "gate_matched": gate["high_matched"], "gate_mismatched": gate["low_mismatched"]}` (E29). Each runner appends one `timings.jsonl` line `{stage: <runner-name>, seconds: <wall>}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner_manifest_smoke.py
import ast
import pathlib
RUNNERS = [
    "run_level1_table_v2.py", "run_level2_table.py", "run_weighted_bc_smoke.py",
    "run_variance_suite.py", "run_data_pareto.py",
]
BASE = pathlib.Path("famail_temporal/baselines")

def test_every_runner_calls_write_run_manifest():
    for r in RUNNERS:
        src = (BASE / r).read_text()
        assert "write_run_manifest" in src, f"{r} missing manifest wiring"
```

*(A static-source assertion is the right test here: actually invoking each GPU runner in a unit test is infeasible. The implementer must still confirm by reading each `main()` that the call is on the success path with the real out-dir + parsed args, and note that in the report.)*

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_runner_manifest_smoke.py -q`
Expected: FAIL — at least one runner lacks the call.

- [ ] **Step 3: Wire each runner**

In each runner's `main()`: record `t0 = time.time()` near the start (several already have one — reuse it); after the result JSON is written to `out_dir`, add:

```python
from famail_temporal.baselines._manifest import write_run_manifest, append_timing, sha256_file
# ... at end of main(), after out_dir is known and results written:
write_run_manifest(
    out_dir, argv=sys.argv, seeds=seeds, edit_dir=str(args.edit_dir),
    extra=_gate_extra,   # {} for runners without a gate; gate fields for L1v2/L2/weighted-BC
)
append_timing(out_dir / "timings.jsonl", "<runner-name>", time.time() - t0)
```

For L1-v2/L2/weighted-BC, build `_gate_extra = {"discriminator_sha256": sha256_file(ckpt_path), "gate_matched": float(gate["high_matched"]), "gate_mismatched": float(gate["low_mismatched"])}` where `gate`/`ckpt_path` already exist in those runners. For variance/data_pareto (no gate), `_gate_extra = {}`. Follow each runner's existing `args`/`seeds`/`out_dir` names — read them; don't assume.

- [ ] **Step 4: Run tests**

Run: `python -m pytest famail_temporal/baselines/tests/test_runner_manifest_smoke.py -q`
Expected: PASS. Also `python -c "import ast,pathlib; [ast.parse((pathlib.Path('famail_temporal/baselines')/r).read_text()) for r in [...]]"` to confirm no syntax break (or import each module).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_*.py famail_temporal/baselines/tests/test_runner_manifest_smoke.py
git commit -m "feat(baselines): emit per-run manifest + timings + gate/ckpt provenance from every runner (E19/E21/E29)"
```

---

### Task 5: Bundle index — figures manifest + run README scaffolding

**Files:**
- Create: `famail_temporal/baselines/_bundle_index.py`
- Test: `famail_temporal/baselines/tests/test_bundle_index.py`

**Interfaces:**
- Produces: `register_figure(manifest_path, *, figure_id, caption, backing_files, producing_command) -> None` (idempotent upsert by `figure_id` into a JSON list at `manifest_path`, stamping `git_sha`); `write_rerun_readme(path, *, stages) -> None` (writes a markdown index from a list of `{stage, out_dir, gate_passed}` dicts).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bundle_index.py
import json
from famail_temporal.baselines import _bundle_index as B


def test_register_figure_upserts_by_id(tmp_path):
    mf = tmp_path / "FIGURES_MANIFEST.json"
    B.register_figure(mf, figure_id="fig5", caption="dose-response",
                      backing_files=["results/wbc/sweep.json"], producing_command="python ...")
    B.register_figure(mf, figure_id="fig5", caption="dose-response v2",
                      backing_files=["results/wbc/sweep.json"], producing_command="python ...")
    rows = json.loads(mf.read_text())
    assert len(rows) == 1 and rows[0]["caption"] == "dose-response v2"
    assert "git_sha" in rows[0]


def test_write_rerun_readme(tmp_path):
    p = tmp_path / "RERUN_README.md"
    B.write_rerun_readme(p, stages=[{"stage": "L1-v2", "out_dir": "results/l1", "gate_passed": True}])
    txt = p.read_text()
    assert "L1-v2" in txt and "results/l1" in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/baselines/tests/test_bundle_index.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# _bundle_index.py
from __future__ import annotations
import json
from pathlib import Path
from famail_temporal.baselines._manifest import git_sha


def register_figure(manifest_path, *, figure_id, caption, backing_files, producing_command) -> None:
    p = Path(manifest_path)
    rows = json.loads(p.read_text()) if p.exists() else []
    rows = [r for r in rows if r.get("figure_id") != figure_id]
    sha, _ = git_sha()
    rows.append({"figure_id": figure_id, "caption": caption,
                 "backing_files": list(backing_files),
                 "producing_command": producing_command, "git_sha": sha})
    p.write_text(json.dumps(rows, indent=2, default=str))


def write_rerun_readme(path, *, stages) -> None:
    lines = ["# Cleaned Re-Run Bundle", ""]
    for s in stages:
        gate = s.get("gate_passed")
        gate_str = "" if gate is None else f" — gate {'PASSED' if gate else 'FAILED'}"
        lines.append(f"- **{s['stage']}** → `{s['out_dir']}`{gate_str}")
    Path(path).write_text("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/baselines/tests/test_bundle_index.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/_bundle_index.py famail_temporal/baselines/tests/test_bundle_index.py
git commit -m "feat(baselines): figures-manifest + rerun-README bundle index (E37/E39)"
```

---

## Self-Review
- **Spec coverage:** E19 (Task 1), E20 (Task 1 capture_env), E21 (Task 2 append_timing + Task 4 wiring), E29 (Task 4 gate/ckpt extra), E30 (Task 3 data_sha256), E37 (Task 5 register_figure), E38 (Task 1 cudnn flags in env), E39 (Task 5 write_rerun_readme). E22 deferred to Plan 5.
- **Placeholders:** Task 3's `_minimal_payloads` intentionally defers to the existing writer-test fixture (the implementer reuses it or asks) — flagged, not a silent gap. Everything else is concrete.
- **Type consistency:** `write_run_manifest`/`append_timing`/`sha256_file`/`capture_env`/`git_sha`/`register_figure`/`write_rerun_readme` signatures consistent across Tasks 1–5; runners (Task 4) consume Tasks 1–2.

## Notes
- Crash-proofing is load-bearing: a provenance probe must never abort a multi-hour run. Tests assert `capture_env` returns keys even when probes fail.
- Plan 2 lands entirely before any GPU run, so the editor/experiment runs (Plans 3–4) emit manifests + timings from the start.
