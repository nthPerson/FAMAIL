"""Campaign run-ledger helper (experiments-section campaign, Task 1).

Wraps every campaign run with reproducibility capture:

    python -m famail_temporal.analysis.run_ledger start \
        --queue-id Q<n> --cmd '<cmd>' --artifact-dir <dir> \
        [--config-note '<txt>'] [--ledger <path>]
    python -m famail_temporal.analysis.run_ledger finish \
        --queue-id Q<n> --artifact-dir <dir> [--ledger <path>]
    python -m famail_temporal.analysis.run_ledger env   # print environment JSON

``start`` appends a LAUNCHED row (timestamp, git SHA, frozen-editor gate,
command, config note, artifact dir) to the ledger and writes
``<artifact-dir>/environment.json``. ``finish`` flips the row to DONE with
end-time + wall-time and appends a SHA-256 checksum section for
``<artifact-dir>/*.json`` + ``*.npz`` to ``<artifact-dir>/PROVENANCE.md``.

Row format (one markdown table row per run, keyed by queue id):

| Q<id> | <status> | <start> | <end> | <wall> | <sha> | frozen-gate:<r> | <config note> | <artifact dir> | <cmd> |
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = _REPO_ROOT / "famail_temporal" / "results" / "EXPERIMENTS_RUN_LEDGER.md"

_FROZEN_PATHS = ["famail_temporal/algorithm/", "famail_temporal/evaluation/runner.py"]

_HEADER = (
    "| queue | status | start (UTC) | end (UTC) | wall | git | frozen-gate |"
    " config | artifact dir | command |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)

_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FMT)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_REPO_ROOT,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _frozen_gate() -> str:
    """PASS iff `git diff main -- <frozen paths>` is empty."""
    try:
        out = subprocess.run(
            ["git", "diff", "main", "--"] + _FROZEN_PATHS,
            capture_output=True, text=True, cwd=_REPO_ROOT,
        )
        if out.returncode == 0 and not out.stdout.strip():
            return "PASS"
        return "FAIL"
    except OSError:
        return "FAIL"


def _environment() -> dict:
    env: dict = {"python": sys.version.split()[0]}
    try:
        import torch  # noqa: PLC0415

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:  # torch absent or CUDA probe failed
        env.setdefault("torch", None)
        env.setdefault("cuda", None)
        env.setdefault("gpu_name", None)
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    env["pip_freeze"] = freeze
    env["pip_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
    return env


def _cell(text: str) -> str:
    """Escape a value for a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _row(fields: list[str]) -> str:
    return "| " + " | ".join(fields) + " |\n"


def _start(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger)
    art = Path(args.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)

    (art / "environment.json").write_text(json.dumps(_environment(), indent=2))

    row = _row([
        args.queue_id, "LAUNCHED", _utcnow(), "-", "-", _git_sha(),
        f"frozen-gate:{_frozen_gate()}", _cell(args.config_note or "-"),
        _cell(str(art)), _cell(args.cmd),
    ])
    if not ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(_HEADER)
    with ledger.open("a") as fh:
        fh.write(row)
    return 0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finish(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger)
    art = Path(args.artifact_dir)
    now = _utcnow()

    lines = ledger.read_text().splitlines(keepends=True)
    prefix = f"| {args.queue_id} | LAUNCHED | "
    idx = next(
        (i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith(prefix)),
        None,
    )
    if idx is None:
        print(f"run_ledger: no LAUNCHED row for {args.queue_id} in {ledger}",
              file=sys.stderr)
        return 1
    fields = [f.strip() for f in lines[idx].strip().strip("|").split(" | ")]
    fields[1] = "DONE"
    fields[3] = now
    try:
        start_dt = datetime.strptime(fields[2], _TS_FMT).replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(now, _TS_FMT).replace(tzinfo=timezone.utc)
        fields[4] = str(timedelta(seconds=round((end_dt - start_dt).total_seconds())))
    except ValueError:
        fields[4] = "?"
    lines[idx] = _row(fields)
    ledger.write_text("".join(lines))

    targets = sorted(list(art.glob("*.json")) + list(art.glob("*.npz")))
    section = [f"\n## Checksums ({now})\n\n"]
    for path in targets:
        section.append(f"{_sha256_file(path)}  {path.name}\n")
    prov = art / "PROVENANCE.md"
    with prov.open("a") as fh:
        fh.writelines(section)
    return 0


def _env_cmd(_args: argparse.Namespace) -> int:
    print(json.dumps(_environment(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_ledger", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="append a LAUNCHED row + environment.json")
    p_start.add_argument("--queue-id", required=True)
    p_start.add_argument("--cmd", required=True)
    p_start.add_argument("--artifact-dir", required=True)
    p_start.add_argument("--config-note", default="")
    p_start.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    p_start.set_defaults(func=_start)

    p_finish = sub.add_parser("finish", help="flip row to DONE + checksum PROVENANCE")
    p_finish.add_argument("--queue-id", required=True)
    p_finish.add_argument("--artifact-dir", required=True)
    p_finish.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    p_finish.set_defaults(func=_finish)

    p_env = sub.add_parser("env", help="print environment JSON")
    p_env.set_defaults(func=_env_cmd)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
