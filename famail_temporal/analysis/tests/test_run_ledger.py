"""Tests for the campaign run-ledger helper."""
import json, re, subprocess, sys
from pathlib import Path
from famail_temporal.analysis import run_ledger as rl


def test_start_appends_launched_row_and_environment(tmp_path):
    ledger = tmp_path / "LEDGER.md"; art = tmp_path / "art"; art.mkdir()
    rc = rl.main(["start", "--queue-id", "Q9", "--cmd", "echo hi",
                  "--artifact-dir", str(art), "--config-note", "PRIMARY",
                  "--ledger", str(ledger)])
    assert rc == 0
    text = ledger.read_text()
    assert "Q9" in text and "LAUNCHED" in text and "echo hi" in text and "PRIMARY" in text
    env = json.loads((art / "environment.json").read_text())
    assert "python" in env and "torch" in env and "pip_freeze_sha256" in env


def test_finish_flips_row_and_writes_checksums(tmp_path):
    ledger = tmp_path / "LEDGER.md"; art = tmp_path / "art"; art.mkdir()
    (art / "metrics.json").write_text('{"a": 1}')
    rl.main(["start", "--queue-id", "Q9", "--cmd", "x", "--artifact-dir",
             str(art), "--ledger", str(ledger)])
    rc = rl.main(["finish", "--queue-id", "Q9", "--artifact-dir", str(art),
                  "--ledger", str(ledger)])
    assert rc == 0
    assert "DONE" in ledger.read_text()
    prov = (art / "PROVENANCE.md").read_text()
    assert "metrics.json" in prov and re.search(r"[0-9a-f]{64}", prov)


def test_frozen_gate_recorded(tmp_path):
    ledger = tmp_path / "LEDGER.md"; art = tmp_path / "art"; art.mkdir()
    rl.main(["start", "--queue-id", "Q9", "--cmd", "x", "--artifact-dir",
             str(art), "--ledger", str(ledger)])
    assert re.search(r"frozen-gate:(PASS|FAIL)", ledger.read_text())
