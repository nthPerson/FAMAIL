"""Shenzhen recount byte-level regression pin (D1 Task 4).

D1 Tasks 2-3 extended ``famail_temporal.analysis.supply_recount`` to sf12 and,
in doing so, touched the Shenzhen path in three ways that are *supposed* to be
behavior-neutral:

  - Task 2: an ``es.df`` -> ``es_df`` variable rename + a single hoisted
    ``_load_driver_mapping(config)`` call.
  - Task 3: a new optional ``seg_lookup=None`` parameter on
    ``apply_substitutions`` whose ``None`` default runs the *exact* prior
    SZ code path (derive the lookup from ``df`` via the SZ transition
    machinery).

Those reviews argued behavior-preservation by diff inspection. This test is
the AUTHORITATIVE empirical pin: it re-runs the whole SZ recount ``main()``
end-to-end under the CURRENT code (post Task 2/3) against the exact inputs of
the campaign's A1 run on the s10 alpha* corpus, and asserts the fresh
``supply_recount.json`` equals the committed campaign artifact byte-for-byte on
every field except two documented exclusions (``edit_dir`` = a run-specific
path, ``runtime_seconds`` = wall-clock). A mismatch is a STOP condition (it
would mean Task 2/3 changed SZ behavior -- exactly what this pin exists to
catch); do NOT relax the comparison or edit production code to make it pass.

The committed reference artifact
``.../2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered/supply_recount.json``
lives under the git-ignored ``famail_temporal/results/`` tree (an on-disk
campaign artifact, not a git blob), so this test carries a skip-when-absent
guard for fresh clones / CI. On a developer machine that has the s10 corpus it
RUNS (does not skip).

Isolation: the tool writes ``supply_recount.json`` / ``supply_recount_report.md``
into whatever ``--edit-dir`` it is pointed at, so this test copies ONLY the two
inputs ``main()`` reads from the edit dir (``histories.pkl`` and
``delta_supply_3d.npz`` -- ``build_edited_pickup_3d`` also reads only
``histories.pkl``) into a pytest ``tmp_path`` scratch dir and runs against
that. The committed results dir is never written to.

Runtime: the full recount is ~3 minutes on CPU (the pin must cover the FULL
committed artifact -- the corpus is intentionally not subset). Run in a
subprocess (matching the SF gate tests) so ``famail_temporal.config``'s
import-time CITY resolution is a clean ``shenzhen`` regardless of any sf12 test
that ran earlier in the same session, and so the wall-clock is bounded by a
subprocess timeout.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REF_DIR = (
    _REPO_ROOT
    / "famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered"
)
_EXPECTED_JSON = _REF_DIR / "supply_recount.json"
_HISTORIES = _REF_DIR / "histories.pkl"
_DELTA_SUPPLY = _REF_DIR / "delta_supply_3d.npz"
_RAW_DIR = _REPO_ROOT / "raw_data"
# main() also reads the SZ driver_index_mapping.pkl (config.SOURCE_DATA_DIR) and
# loads the cached DataBundle (config.CACHE_DIR) inside the subprocess; a machine
# missing either would FAIL at the subprocess rather than skip, so guard them too.
# Paths are the shenzhen defaults, hardcoded relative to _REPO_ROOT to match the
# other guard paths (this test module deliberately never imports config).
_DRIVER_MAPPING = _REPO_ROOT / "famail_temporal/source_data/driver_index_mapping.pkl"
_CACHE_DIR = _REPO_ROOT / "famail_temporal/cache"

# Fields that are legitimately run-specific (not part of the pinned numeric
# content) and therefore excluded from the byte-level comparison:
#   - edit_dir: absolute path of the scratch dir this run was pointed at.
#   - runtime_seconds: wall-clock; nondeterministic by construction.
_EXCLUDE_TOP = {"edit_dir", "runtime_seconds"}

_SZ_DATA_ABSENT = not (
    _EXPECTED_JSON.is_file()
    and _HISTORIES.is_file()
    and _DELTA_SUPPLY.is_file()
    and _RAW_DIR.is_dir()
    and any(_RAW_DIR.glob("taxi_record_0*_50drivers.pkl"))
    and _DRIVER_MAPPING.is_file()
    and _CACHE_DIR.is_dir()
    and any(_CACHE_DIR.glob("active_taxis_*.pkl"))
)


def _numeric_diffs(fresh, expected, path: str = "") -> list[str]:
    """Recursively diff two JSON-decoded objects, EXACTLY (float equality is
    byte-level: values must be ``==`` AND share the same ``repr``, so a
    last-ULP change is caught). Top-level keys in ``_EXCLUDE_TOP`` are skipped.
    Returns a (possibly empty) list of human-readable divergences."""
    diffs: list[str] = []
    if isinstance(fresh, dict) and isinstance(expected, dict):
        for k in sorted(set(fresh) | set(expected)):
            if path == "" and k in _EXCLUDE_TOP:
                continue
            if k not in fresh:
                diffs.append(f"{path}/{k}: missing in FRESH")
                continue
            if k not in expected:
                diffs.append(f"{path}/{k}: missing in EXPECTED")
                continue
            diffs += _numeric_diffs(fresh[k], expected[k], f"{path}/{k}")
    elif isinstance(fresh, list) and isinstance(expected, list):
        if len(fresh) != len(expected):
            return [f"{path}: list length {len(fresh)} (FRESH) != {len(expected)} (EXPECTED)"]
        for i, (a, b) in enumerate(zip(fresh, expected)):
            diffs += _numeric_diffs(a, b, f"{path}[{i}]")
    else:
        if fresh != expected or (isinstance(fresh, float) and repr(fresh) != repr(expected)):
            diffs.append(f"{path}: FRESH={fresh!r} EXPECTED={expected!r}")
    return diffs


def _diff_fixture() -> dict:
    """Small nested (dict + list) fixture for the _numeric_diffs self-test.
    A fresh copy each call so a mutation in one test can't leak into another.
    Top-level keys avoid _EXCLUDE_TOP so nothing is silently skipped."""
    return {
        "alpha": {"score": 1.0, "counts": [10, 20, 30]},
        "beta": 0.5,
    }


def test_numeric_diffs_identity_is_empty():
    """Two byte-identical structures diff to nothing (no false positives)."""
    assert _numeric_diffs(_diff_fixture(), _diff_fixture()) == []


def test_numeric_diffs_catches_float_and_int_perturbations():
    """A 1e-9 float change and a +1 int change are both caught, and the
    reported diffs name the exact nested paths (dict key + list index)."""
    # 1e-9 float perturbation on a nested dict value.
    fresh = _diff_fixture()
    fresh["alpha"]["score"] += 1e-9
    float_diffs = _numeric_diffs(fresh, _diff_fixture())
    assert float_diffs, "1e-9 float perturbation not detected"
    assert any("/alpha/score" in d for d in float_diffs), float_diffs

    # +1 int perturbation inside a nested list.
    fresh = _diff_fixture()
    fresh["alpha"]["counts"][1] += 1
    int_diffs = _numeric_diffs(fresh, _diff_fixture())
    assert int_diffs, "+1 int perturbation not detected"
    assert any("/alpha/counts[1]" in d for d in int_diffs), int_diffs


@pytest.mark.skipif(_SZ_DATA_ABSENT, reason=(
    "Shenzhen s10 campaign corpus / raw GPS / driver mapping / DataBundle cache "
    f"absent on this machine (checked {_EXPECTED_JSON}, {_HISTORIES}, "
    f"{_DELTA_SUPPLY}, {_RAW_DIR}, {_DRIVER_MAPPING}, {_CACHE_DIR})"
))
def test_sz_recount_regression(tmp_path):
    """Re-run the SZ recount on the committed s10 corpus and pin its output to
    the committed ``supply_recount.json`` byte-for-byte (all fields except
    ``edit_dir`` and ``runtime_seconds``). A nonzero diff is a STOP -- it means
    a supposedly behavior-neutral Task 2/3 change altered the SZ path; report
    the field-level diff, do not touch production code to force a pass.
    """
    # Copy ONLY the inputs main() reads from the edit dir into a scratch dir,
    # so the committed results dir is never written to.
    scratch = tmp_path / "edit"
    scratch.mkdir()
    shutil.copy2(_HISTORIES, scratch / "histories.pkl")
    shutil.copy2(_DELTA_SUPPLY, scratch / "delta_supply_3d.npz")

    env = dict(os.environ)
    env["FAMAIL_CITY"] = "shenzhen"  # the SZ default; explicit for a clean config
    proc = subprocess.run(
        [sys.executable, "-m", "famail_temporal.analysis.supply_recount",
         "--edit-dir", str(scratch)],
        cwd=str(_REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"supply_recount subprocess failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    fresh_path = scratch / "supply_recount.json"
    assert fresh_path.is_file(), (
        f"tool did not write supply_recount.json to {scratch}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    fresh = json.loads(fresh_path.read_text())
    expected = json.loads(_EXPECTED_JSON.read_text())

    # Sanity: the fresh run is genuinely the SZ path over the real corpus, not
    # an empty/degenerate output that would make the comparison vacuous.
    assert fresh.get("city") == "shenzhen", fresh
    assert fresh["substitution_stats"]["n_histories"] > 0, fresh

    diffs = _numeric_diffs(fresh, expected)
    assert not diffs, (
        "SZ recount regression pin FAILED: fresh output diverges from the "
        "committed campaign artifact "
        f"({_EXPECTED_JSON}). This means a D1 Task 2/3 change altered the "
        "Shenzhen recount behavior -- STOP, do not relax this pin or edit "
        "production code to force a pass. Field-level divergences "
        f"({len(diffs)}):\n  " + "\n  ".join(diffs)
    )
