"""Static-source smoke test: every baseline runner must call write_run_manifest.

A static-source assertion is the right test here — actually invoking each
GPU runner in a unit test is infeasible. The implementer must still confirm
by reading each main() that the call is on the success path with the real
out-dir + parsed args.
"""
import ast
import pathlib

RUNNERS = [
    "run_level1_table_v2.py",
    "run_level2_table.py",
    "run_weighted_bc_smoke.py",
    "run_variance_suite.py",
    "run_data_pareto.py",
]
BASE = pathlib.Path("famail_temporal/baselines")


def test_every_runner_calls_write_run_manifest():
    for r in RUNNERS:
        src = (BASE / r).read_text()
        assert "write_run_manifest" in src, f"{r} missing manifest wiring"
