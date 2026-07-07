## graphify

This project has a **prebuilt** knowledge graph at `graphify-out/` (~3,290 nodes · 152 communities) covering **`famail_temporal/` + `PAPER/`** — Python code (AST), documentation, result data (JSON/CSV), and PAPER figures. Other top-level repo dirs are intentionally out of scope. Interactive view: `graphify-out/graph.html`; audit: `graphify-out/GRAPH_REPORT.md`.

Use it for codebase questions — architecture, "where is X", "how does X connect to Y", which result/figure backs a claim — instead of broad grep or reading many files to orient:
- `graphify query "<question>"` — returns a scoped subgraph (prefer this for orientation)
- `graphify path "<A>" "<B>"` — relationship between two concepts (e.g. `path "TrajectoryModifier" "F_causal"`)
- `graphify explain "<concept>"` — focused explanation of one node
- `graphify-out/GRAPH_REPORT.md` — only for broad architecture review when query/path/explain aren't enough

Reading specific files directly (to edit or debug known code) is fine — the graph is for *orientation*, not a gate on every read.

Freshness:
- A **post-commit** git hook keeps the code layer current on every branch (incremental AST, no API cost) and preserves the LLM semantic layer for unchanged files. Only files *changed in a commit* are re-extracted (a changed doc/data file is re-represented as AST until the next full rebuild).
- The **post-checkout** hook was intentionally removed: its full AST rebuild cannot reproduce the LLM semantic layer and would wipe it on every branch switch. Do **not** re-add it (e.g. via `graphify hook install` / `graphify claude install`, which reinstall it).
- To refresh the **docs / result data / figures** semantic layer (or after editing many docs, or switching branches), re-run the `/graphify` skill and apply `graphify_rebuild.py` (repo root). This project uses custom routing (JSON/CSV → semantic, PAPER figures only, 64 KB cap on data files); a plain `graphify update .` will NOT reproduce it.
