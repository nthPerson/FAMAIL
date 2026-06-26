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
