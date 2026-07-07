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


def not_ignored(paths):
    if not paths:
        return []
    res = subprocess.run(['git', 'check-ignore', '--no-index', *paths],
                         capture_output=True, text=True)
    ignored = set(res.stdout.split('\n'))
    return [p for p in paths if p not in ignored]


def main():
    root = Path('.').resolve()
    r = detect(root)
    f = r['files']

    def rel(p):
        p = str(p)
        return p[len(str(root)) + 1:] if p.startswith(str(root)) else p

    # 1) .json: code -> document
    code_json = [p for p in f['code'] if p.lower().endswith('.json')]
    f['code'] = [p for p in f['code'] if not p.lower().endswith('.json')]

    # 2) .csv: discover in scope, keep only git-tracked (respects .gitignore exceptions)
    csv_all = [os.path.join(dp, fn)
               for base in ('famail_temporal', 'PAPER')
               for dp, _, fns in os.walk(base) for fn in fns
               if fn.lower().endswith('.csv')]
    csv_keep = not_ignored(csv_all)

    # 3) 64 KB cap on data docs
    def under_cap(paths):
        return [p for p in paths if (os.path.getsize(p) if os.path.exists(p) else 0) <= CAP]
    data_docs = under_cap(code_json) + under_cap(csv_keep)

    # 4) images: PAPER/ only
    f['image'] = [p for p in f['image'] if '/PAPER/' in p or rel(p).startswith('PAPER/')]

    # 5) rebuild documents + totals
    f['document'] = list(f['document']) + data_docs
    allf = f['code'] + f['document'] + f['paper'] + f['image'] + f['video']
    r['total_files'] = len(allf)
    r['total_words'] = sum(
        len(Path(p).read_text(encoding='utf-8', errors='ignore').split())
        for p in f['document'] if os.path.exists(p))

    Path('graphify-out/.graphify_detect.json').write_text(json.dumps(r, ensure_ascii=False))
    print(f"routed: code={len(f['code'])} docs={len(f['document'])} "
          f"(md={sum(1 for p in f['document'] if p.endswith('.md'))} "
          f"json={sum(1 for p in f['document'] if p.endswith('.json'))} "
          f"csv={sum(1 for p in f['document'] if p.endswith('.csv'))}) "
          f"images={len(f['image'])} total={r['total_files']}")


if __name__ == '__main__':
    main()
