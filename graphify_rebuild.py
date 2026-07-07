#!/usr/bin/env python3
"""Reproduce the FAMAIL graphify corpus routing for a manual doc refresh.

The stock `/graphify --update` re-detects with graphify's defaults, which would
route .json as shallow AST "code" and skip .csv entirely. This project instead
routes data files through the LLM *semantic* pass so the graph knows what each
file represents. Run this AFTER `graphify.detect.detect('.')` and BEFORE
extraction to re-apply that routing; it rewrites graphify-out/.graphify_detect.json.

Routing rules (kept in sync with the original build):
  * scope           : famail_temporal/ + PAPER/ only (via repo-root .graphifyignore)
  * .py             : AST (code) — unchanged
  * .json           : moved code -> document (LLM semantic)
  * .csv            : discovered + added as document (graphify's classifier drops csv)
  * size cap        : json/csv > 64 KB dropped (numeric dumps: training_curves, etc.)
  * .png            : keep ONLY PAPER/ figures (vision); drop all other project png

Usage (from the repo root, /home/robert/FAMAIL):
    python graphify_rebuild.py
Then continue the graphify pipeline (Part A AST + Part B semantic subagents +
Part C merge + build). If GEMINI_API_KEY is set, graphify can run Part B headless;
otherwise re-run the /graphify skill so the host model dispatches extraction subagents.
"""
import json, os, subprocess
from pathlib import Path
from graphify.detect import detect

CAP = 64 * 1024  # bytes


def git_tracked(patterns, cap=None):
    """Tracked files matching git pathspecs, scoped to famail_temporal/ + PAPER/.

    Uses `git ls-files`, which is independent of .graphifyignore — so the result
    data/reports we exclude from graphify's detect (to stop the hook re-AST-ing
    them) are still discovered here and re-included as semantic documents.
    """
    out = subprocess.run(['git', 'ls-files', '--', *patterns],
                         capture_output=True, text=True).stdout.splitlines()
    files = []
    for rel in out:
        rel = rel.strip()
        if not rel or not (rel.startswith('famail_temporal/') or rel.startswith('PAPER/')):
            continue
        if cap is not None and (os.path.getsize(rel) if os.path.exists(rel) else 0) > cap:
            continue
        files.append(os.path.abspath(rel))
    return files


def main():
    root = Path('.').resolve()
    r = detect(root)
    f = r['files']

    def rel(p):
        p = str(p)
        return p[len(str(root)) + 1:] if p.startswith(str(root)) else p

    # code: keep .py; drop .json (routed to semantic documents below)
    f['code'] = [p for p in f['code'] if not p.lower().endswith('.json')]

    # documents = every tracked .md + every tracked .json/.csv <= 64 KB, across
    # famail_temporal/ + PAPER/. git ls-files sees the .graphifyignore'd result
    # data too, so this re-includes it semantically while the hook leaves it alone.
    md   = git_tracked(['*.md'])
    data = git_tracked(['*.json', '*.csv'], cap=CAP)
    docs = sorted(set(md + data))
    f['document'] = docs

    # images: PAPER/ figures only
    f['image'] = [p for p in f['image'] if '/PAPER/' in p or rel(p).startswith('PAPER/')]

    allf = f['code'] + f['document'] + f['paper'] + f['image'] + f['video']
    r['total_files'] = len(allf)
    r['total_words'] = sum(
        len(Path(p).read_text(encoding='utf-8', errors='ignore').split())
        for p in docs if os.path.exists(p))

    Path('graphify-out/.graphify_detect.json').write_text(json.dumps(r, ensure_ascii=False))
    print(f"routed: code(py)={len(f['code'])} docs={len(docs)} "
          f"(md={sum(1 for p in docs if p.endswith('.md'))} "
          f"json={sum(1 for p in docs if p.endswith('.json'))} "
          f"csv={sum(1 for p in docs if p.endswith('.csv'))}) "
          f"images={len(f['image'])} total={r['total_files']}")


if __name__ == '__main__':
    main()
