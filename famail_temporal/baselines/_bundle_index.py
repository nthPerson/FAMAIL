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
