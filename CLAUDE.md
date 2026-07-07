## graphify

This project has a **prebuilt** knowledge graph at `graphify-out/` (~3,290 nodes · 152 communities) covering **`famail_temporal/` + `PAPER/`** — Python code (AST), documentation, result data (JSON/CSV), and PAPER figures. Other top-level repo dirs are intentionally out of scope. Interactive view: `graphify-out/graph.html`; audit: `graphify-out/GRAPH_REPORT.md`.

Use it for codebase questions — architecture, "where is X", "how does X connect to Y", which result/figure backs a claim — instead of broad grep or reading many files to orient:
- `graphify query "<question>"` — returns a scoped subgraph (prefer this for orientation)
- `graphify path "<A>" "<B>"` — relationship between two concepts (e.g. `path "TrajectoryModifier" "F_causal"`)
- `graphify explain "<concept>"` — focused explanation of one node
- `graphify-out/GRAPH_REPORT.md` — only for broad architecture review when query/path/explain aren't enough

Reading specific files directly (to edit or debug known code) is fine — the graph is for *orientation*, not a gate on every read.

Freshness:
- **Code** stays current automatically via the git post-commit/post-checkout hook (all branches, AST-only, no API cost).
- **Docs / result data / figures** (the LLM-extracted layer) refresh only on a full rebuild. This project uses custom routing (JSON/CSV → semantic, PAPER figures only, 64 KB cap on data files) — to reproduce it, re-run the `/graphify` skill and apply `graphify_rebuild.py` (repo root). A plain `graphify update .` will NOT reproduce that routing.
