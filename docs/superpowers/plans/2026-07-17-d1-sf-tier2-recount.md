# D1 — SF Tier-2 Recount Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `supply_recount.py --city sf12` produces a validated tier-2 distinct-taxi recount of the SF corpus, feeding `channel_decomposition --tier2-grid` per spec `docs/superpowers/specs/2026-07-17-d1-sf-tier2-recount-design.md` (Robert-approved 2026-07-17).

**Architecture:** ONE new component — an SF raw-ping adapter producing the exact DataFrame schema the existing `recount_tier2()` consumes — plus city-conditional wiring. Counting semantics, substitution replay, grid comparison, and reporting are reused untouched. Two hard gates (G-repro MAE 0.0, G-match 100%) pin correctness; the SZ path carries a byte-level regression test.

**Tech Stack:** Python/pandas/numpy; existing `famail_temporal.analysis.supply_recount` + `channel_decomposition`; SF pipeline under `famail_temporal/second_dataset/data/source_generation/`.

## Global Constraints
- **Mirror, don't reinterpret** (spec §1): only the adapter is new; the SF grid transform, occupancy/seeking semantics, and supply-grid construction are IMPORTED or replicated verbatim from the SF source-generation pipeline — never re-derived. Count presence from pings AS THEY ARE (no interpolation; spec Risks — SF GPS gaps up to ~18.6 cells are a known property).
- **Gates before results** (spec §2): G-repro (unedited recount == production `active_taxis_3d` under `FAMAIL_CITY=sf12`, MAE 0.0) and G-match (100% history→raw) must PASS before the edited recount is read. Nonzero G-repro MAE = STOP, diagnose, surface — do not proceed or tune toward agreement.
- **SZ regression:** after all changes, `supply_recount --edit-dir <s10> ` must reproduce the committed `s10 .../supply_recount.json` numbers exactly.
- **Decision rule** (spec §4): SF tier-2 supply CI excludes 0 → §4.7 upgrade; otherwise report-as-measured + IMMEDIATE surface to Robert (his pre-committed Reading-A reassessment trigger). Never smooth.
- Era: SF corpus = `famail_temporal/results/2026-07-11T11-31-55_supply_lift_a10_sf12_filtered` (fingerprint 1,330+629). All runs ledger-wrapped (`D1-RECOUNT`, `D1-CHAN`).
- No editor changes; `famail_temporal/algorithm/` + `evaluation/runner.py` untouched. Tests: `famail_temporal/analysis/tests/` (or the module's existing test home — discover in Task 1). Suite stays green.
- GPU: the h-chain owns the GPU until it drains (WBC-N12 last, ETA overnight 07-17→18); the recount is CPU-heavy and may run beside it ONLY if host RAM allows (guarded-companion doctrine, MemAvailable ≥ 20 GB) — otherwise queue after.

### Task 1: Discovery + SF ping adapter (read-first task)
**Files:** Create `famail_temporal/analysis/sf_recount_adapter.py`; Test `famail_temporal/analysis/tests/test_sf_recount_adapter.py` (create dir/init if absent, mirroring the repo's test conventions).
**Interfaces produced:** `load_sf_pings(raw_dir: Path) -> pd.DataFrame` returning EXACTLY the schema `recount_tier2()`/`apply_substitutions()` consume for SZ (Task-1 Step 1 documents that schema from the SZ loader — column names, dtypes, driver keying, time indexing, cell coords via the sf12 32×30 transform, occupancy/seeking flag).
- [ ] Step 1: Read `supply_recount.py`'s SZ raw-load + df construction (main() lines ~330-360, `_segment_rows_by_driver`, `recount_tier2`) and WRITE DOWN the consumed schema in the module docstring of the new adapter file.
- [ ] Step 2: Read the SF source-generation pipeline (`famail_temporal/second_dataset/data/source_generation/` — locate ping loading, lat/lon→grid transform, occupancy handling, and HOW the production sf12 `active_taxis_3d` was built). Record file:line anchors in the docstring. If the production grid derivation cannot be located, STOP: status NEEDS_CONTEXT (G-repro is unprovable without it).
- [ ] Step 3: Failing unit test — `load_sf_pings` on a tiny slice (first N rows of one taxi's source file) returns the documented schema (columns/dtypes assertions + grid coords within (32,30) bounds + occupancy flag ∈ {0,1}).
- [ ] Step 4: Implement the adapter by importing/replicating the pipeline's own transform (comment each replicated line with its source anchor). Run test → green.
- [ ] Step 5: Commit `feat(d1): SF ping adapter (Task 1)`.

### Task 2: G-repro gate — unedited recount reproduces the production SF grid
**Files:** Modify `famail_temporal/analysis/supply_recount.py` (city-conditional raw load only); Test add `test_sf_g_repro_gate` (integration, skips when SF data absent — must PASS on this machine, not skip).
- [ ] Step 1: Wire `--city sf12`: replace the deferred-stub branch with `load_sf_pings`; `_SUPPORTED_CITIES = {"shenzhen", "sf12"}`; everything downstream unchanged.
- [ ] Step 2: Failing test: recount the UNEDITED sf12 corpus (`FAMAIL_CITY=sf12`, substitutions disabled/empty) and assert grid == `bundle.active_taxis_3d` with MAE 0.0 (exact; use the module's own `_grid_compare`).
- [ ] Step 3: Run. If MAE ≠ 0.0: STOP per Global Constraints (diagnose against the source-generation anchors from Task 1; report findings; do NOT adjust the adapter to force agreement without understanding).
- [ ] Step 4: Commit `feat(d1): sf12 recount path + G-repro gate (Task 2)`.

### Task 3: G-match — substitution replay on the SF edit dir
**Files:** Test only (`test_sf_g_match_gate`), unless replay needs an SF branch (it should not — `apply_substitutions` is city-agnostic once the df schema matches; if it does need one, smallest diff + explain).
- [ ] Step 1: Failing test: load `<sf12_filtered>/histories.pkl`, replay against the adapter's df, assert 100% matched (n_matched == n_histories; the SZ machinery's own counters).
- [ ] Step 2: Run to green (or STOP + surface if matching < 100% — that would contradict the corpus's provenance).
- [ ] Step 3: Commit `feat(d1): SF substitution replay gate (Task 3)`.

### Task 4: SZ regression + suite
- [ ] Step 1: Re-run `python -m famail_temporal.analysis.supply_recount --edit-dir famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered` into a scratch copy (do NOT overwrite the committed artifacts — use a temp copy of the edit dir or diff in-memory) and assert its summary numbers equal the committed `supply_recount.json` exactly.
- [ ] Step 2: Whole analysis+baselines test suites green.
- [ ] Step 3: Commit `test(d1): SZ recount regression pin (Task 4)`.

### Task 5: Production run (ledger-wrapped; controller may execute directly)
- [ ] Step 1: `python -m famail_temporal.analysis.supply_recount --city sf12 --edit-dir famail_temporal/results/2026-07-11T11-31-55_supply_lift_a10_sf12_filtered --persist-grids` under ledger id `D1-RECOUNT` (env `FAMAIL_CITY=sf12`; check RAM headroom if the h-chain still runs).
- [ ] Step 2: `python -m famail_temporal.analysis.channel_decomposition --edit-dir <same> --bootstrap 2000 --seed 0 --tier2-grid <same>/S_tier2_after.npz` under `D1-CHAN` (confirm the tool is city-correct under FAMAIL_CITY=sf12 — it reads the bundle).
- [ ] Step 3: Apply the DECISION RULE from Global Constraints; either way, record the outcome verbatim in the ledger config-note.

### Task 6: Slot-in + curation (controller)
- [ ] Step 1: Per the decision rule: §4.7 upgrade (replace "tier-1 accounting only, a lower bound... not plumbed" with the two-tier statement + numbers) OR the surfaced-to-Robert path. Update the Reading-B provenance comment either way.
- [ ] Step 2: Curate `sf12_a10_supply_recount.json` + refreshed `sf12_a10_channel_decomposition.json` to a10/; DATA_INVENTORY rows; SF_FRAMING_UPDATE.md decision-block note; memory update.
- [ ] Step 3: Gates (latexmk + lint, exit codes checked, render if §4.7 reflows) → commit `paper+campaign(D1): ...`.

## Self-Review
Spec coverage: §1→T1/T2 wiring, §2 gates→T2/T3, §3 outputs→T5, §4 rule→T5/T6, §5 non-goals→T4 SZ pin + no-editor constraint. Placeholders: T1 is an explicit read-first task with named anchors and a NEEDS_CONTEXT stop condition; adapter code cannot be pre-written without the discovery — outcome pinned by G-repro MAE 0.0, the strongest possible test. Types: adapter df schema is defined as "exactly what recount_tier2 consumes," documented in T1S1 before any implementation.
