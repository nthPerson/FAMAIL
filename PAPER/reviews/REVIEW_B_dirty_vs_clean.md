I'll synthesize the adversarial review for Component B. The data is comprehensive and well-verified, so I can produce the report directly.

# ADVERSARIAL REVIEW — COMPONENT B (Stuck-GPS Cleanup, Dirty-vs-Clean)

## (1) OVERALL VERDICT

**The cleanup work is methodologically sound and the core findings are honest. The defects are presentation/labeling, not correctness.**

The detector is principled, the removal is surgical, the dirty-vs-clean comparison is genuinely apples-to-apples, and the headline F_spatial recovery (+0.0213) is correctly attributed to sink removal (scoring code byte-identical across runs; demographic-independence confirmed to 17 digits). Every load-bearing number reproduces against source JSONs. The "changed no conclusion" claim holds at the substantive level.

What's wrong is the **reader-facing framing in the PAPER deliverable**: a silent 3-feature/4-feature mismatch that creates an unexplained 0.08 F_causal gap, a stale cleanup caption, a removal-rate denominator that invites a ~5x misread, and a README headline that reads as a 416% paradox. None of these is a numerical error, but all four are reviewer-bait that must be fixed before KDD submission.

**Recommendation: SHIP after fixing the 4 confirmed labeling issues below. No re-runs required (option (a) on the feature-set fix is optional).**

---

## (2) CONFIRMED MUST-FIX ISSUES

### MF-1 — Cleanup-delta tables are silently 3-feature inside a 4-feature-labeled deliverable *(important)*
The cleanup tables report 3-feature F_causal (clean edited **0.8193**) while the paper headline is 4-feature (edited **0.7409**, raw **0.7253**) — a **0.08 gap** on the *same* clean editor run. The three cleanup files carry no feature-set label; README.md:50 has the breadcrumb but the tables in isolation don't. A reviewer cross-referencing will hit an irreconcilable discrepancy.
- **Fix:** Add a self-describing header note to `dataset_summary.md` and `experiment_cleanup_delta.md` stating F_causal uses the 3-feature set, that conclusions are feature-set-invariant (cite `comparison_3v4.md`), and that absolute values are NOT comparable to the 4-feature headline. Add a feature-set provenance column/comment to `cleanup_delta_editor.csv`. **Cleaner (optional):** regenerate all three tables from the 4-feature dirty/clean runs so the whole deliverable is uniformly 4-feature. Note F_spatial is feature-independent (identical 0.10342706 across both sets).

### MF-2 — removal_rate denominator invites a "~39% of seeking removed" misread *(important)*
`removal_rate` (0.4975 dirty / 0.3895 clean) is `n_removed / total_extracted` (seeking **+** driving), but the table shows it adjacent only to `total_seeking_extracted`. The true seeking-only fractions are **~91% / ~90%**, and seeking extraction itself fell 214,286→133,091. The 38.95% figure is correct as an *overall* rate but is not "39% of seeking."
- **Fix:** In `dataset_summary.py` (blocks ~lines 24-35; md rows ~71-76), add `total_extracted` (and ideally `total_driving_extracted`) rows so the denominator is visible, and rename to `overall_removal_rate` with caption "= n_removed / total_extracted (seeking+driving)". Do **not** publish `n_removed/total_seeking` as "seeking removed" — `n_removed` spans driving trajectories too (`counts_by_category`), so that ratio over-attributes.

### MF-3 — Stale hardcoded cleanup caption: "6 drivers, cell (28,52) removed" *(minor, but reader-facing)*
`experiment_delta.py:231` hardcodes a caption describing the *superseded* Meeting-40 diagnostic (6 drivers / cell (28,52)). The actual filter removed **10 cells across 9 drivers** (106,677 pickups), and (28,52) is **not** among the flagged cells — directly contradicting `dataset_summary.md` (n_sink_cells=10). Propagates into both the PAPER md and the results copy.
- **Fix:** Replace the literal at `experiment_delta.py:231` with text derived from `processing_metadata.json` (`stuck_gps_sinks`): interpolate `len(flagged_cells)`, distinct plate count, and `n_pickups_removed` so it can't drift again. Re-render both md copies.

### MF-4 — README headline reads as a 416% paradox; redistribution residual undisclosed *(minor, but reader-facing)*
README.md:16-17 says sink (29,53) "alone accounted for **+0.0885** of the F_spatial recovery (net global **+0.0213**)" — the local gain is 416% of the net, with no mention that ~88% was offset by a **−0.0783** redistribution across non-sink cells (1/N renormalization). The full decomposition IS disclosed in `sink_f_spatial_decomposition.md`, but the README pairing reads as an unexplained paradox.
- **Fix:** Reword to disclose the mechanism: "(29,53) recovered +0.0885 locally; ~88% of that was offset by a −0.0783 re-baselining across non-sink cells under the 1/N-shifted decomposition, netting +0.0213 global." Do not present +0.0885 as "the recovery."

### MF-5 (partial) — Unweighted "edited" WBC arm flips sig→n.s.; verdict prose omits it *(minor)*
The unweighted `edited` arm went from p=0.03125 (dirty) to p=0.4375 (clean); the verdict logic (`experiment_delta.py:275`) filters to `edited_w*` so the blanket "PRESERVED — weighted arms stay significant" never flags the flip. **This actually strengthens the story** (only upweighting recovers fairness; vanilla transfer is null) — but should be surfaced, not hidden. *Partial:* the table row already shows both p-values, and the paper's L2 conclusion never rested on this arm's significance, so it's a transparency gap, not a conclusion reversal.
- **Fix:** Have the renderer append a note when the unweighted arm's significance flips while direction is preserved (both negative), or tighten the prose to "weighted (wN) arms stay significant; unweighted edited stays directionally negative but n.s. under clean (n=6 floor)." No re-run needed.

---

## (3) MINOR / OBSERVATIONS

**Worth a one-line fix:**
- **config.py:86 — two factual errors in the calibration comment:** "~32.7% of all pickups" is actually **46.4%** (106,677/230,069); and "every group dropoff_ratio ≤ 0.001" is violated by the headline sink (29,53) at 0.0010465. Reword to ~46.4% of weekday raw pickups and "≤ 0.0011" (or "far below the 0.02 threshold"). These feed the paper narrative, so correct them.
- **"share of global shift" column sums to 468%, not 100%** (`sink_f_spatial_decomposition.md`) — it's `delta/net_global_shift`, a multiple, not a share. Rename to "delta / net global shift (×)" or normalize among sinks and report the residual separately. The residual IS disclosed in the footer (honest), so this is cosmetic over-reading.
- **L2 dirty "n.s." rests on the n=5 discreteness floor:** dirty p=0.0625 with 5/5 negative diffs is the Wilcoxon floor, not a comfortable margin. Acknowledge it was directionally unanimous; clean moved it to a genuinely mixed-sign null. Conclusion (no positive vanilla transfer) holds either way.
- **Dirty variance baseline is in an untracked dir** (`baselines/variance_suite/`, git status `??`). The dirty variance column (f_causal −0.0011) isn't reproducible from a clean checkout. Commit it (or snapshot the source JSONs alongside `PAPER/data/`).

**Defensible-as-is, document only:**
- **Hybrid guard is a hard equality assertion** — brittle to any legitimate data/threshold/library drift (will hard-crash). Acceptable for a frozen paper artifact (pins reproducibility), but document that `STUCK_GPS_EXPECTED_CELLS` is dataset+threshold-specific and any change requires a report-only dry-run (`expected_cells=None`) before updating the set — don't treat the crash as a bug.
- **Detection assumes binary 0/1 indicator + per-driver temporal sort** — both satisfied in the production path (event_stream sorts before filtering; diff-based masks ignore non-unit jumps). Low priority: add `assert df['passenger_indicator'].isin([0,1]).all()` at filter entry to fail loudly if encoding ever changes.
- **7 new editing-loop config keys** appear in the clean run but are all downstream of `metrics_before`; the F_spatial-before baseline (the headline) is insulated. The AFTER metrics/edit-counts are not strictly single-variable (data + runner refactor), worth a caveat.
- **git_dirty=true on both editor runs** — recorded SHAs don't fully pin executed code. Residual risk is low (cleanup work was in source_generation/runner.py, not spatial.py/grid.py); the persisted `grid_before.pkl` files back the numbers independently. Cite the byte-identical scoring SHA-diff as primary evidence, treat git_dirty as a documented caveat.
- **Heatmap diff-panel scale** is set by the +0.0885 headline cell, rendering the 9 other sink recoveries near-white. Cosmetic; consider percentile-clip/symlog or annotate the scale source.

**Verified-correct (adversarial concerns refuted):**
- Conjunctive detector (n_pickups≥1000 AND dropoff_ratio<0.02) spares real hubs (test_real_hub passes; all 10 flagged cells have dropoff_ratio ≤ 0.00105).
- min_pickups=1000 sits in an 8.8× bimodal gap (2969→339); identical flagged set for any threshold in [340, 2969]; stable 500–2000 plateau.
- +1 coordinate offset internally consistent; guard compares like-for-like; no off-by-one.
- Removal is coordinate-keyed (6-decimal, ~0.1m), not cell-level — co-located real pickups retained (negligible collateral).
- Per-sink F_spatial method (channel-0 nansum over active t) provably sums to F_spatial; reproduces 0.0822/0.1034 to float32.
- (x-1,y-1) conversion, South-at-bottom orientation, sink circling all correct.
- PAPER copies byte-identical to source results (no stale-artifact drift in the JSONs).
- The +0.0213 shift is NOT code-drift-contaminated (spatial.py/grid.py/attribution.py byte-identical across SHAs); F_spatial is structurally demographic-independent.
- The dirty-vs-clean comparison is genuinely apples-to-apples at the 3-feature level (same editor config, matched seeds, n_active=34,524 unchanged).

---

**Bottom line:** No numerical or methodological errors found in Component B. Five reader-facing labeling defects (MF-1 through MF-5) must be fixed before submission — all are documentation/renderer edits requiring no re-runs. The cleanup is sound and, once relabeled, honestly presented; the unweighted-arm flip and the redistribution residual actually *strengthen* the narrative once disclosed.

Relevant files: `PAPER/tables/dataset_summary.md`, `PAPER/tables/experiment_cleanup_delta.md`, `PAPER/tables/cleanup_delta_editor.csv`, `PAPER/README.md` (lines 16-17, 50), `PAPER/tables/sink_f_spatial_decomposition.md`, `famail_temporal/analysis/experiment_delta.py` (line 231, ~275), `famail_temporal/analysis/dataset_summary.py` (~lines 24-42, 67-76), `famail_temporal/source_generation/config.py` (line 86), `famail_temporal/source_data/processing_metadata.json`, `famail_temporal/baselines/variance_suite/` (untracked).