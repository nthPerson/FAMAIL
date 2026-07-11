# Cross-arm comparison table (PENDING — lands with the GPU runs)

**Status: ⏳ deferred until the three perturbation arms (ifgsm / fgsm / random-jitter) run on the GPU.**

This directory will hold the assembled **6-row comparison table** — raw / FAMAIL / ifgsm / fgsm /
random-jitter / demographic-oversampling — produced by
`famail_temporal/baselines/assemble_baseline_table.py` over the arm dirs plus the hand-authored
raw/FAMAIL stub files (headline numbers transcribed, never recomputed). Columns: Fidelity-A, gate,
Fidelity-B(JS), ΔF_causal, ΔF_spatial, adjacency-violation %, mean final_p, n.

Already in place:
- The demographic-oversampling arm's `metrics.json` writes the assembler's exact schema — ingestion is
  covered by `test_arm_metrics_ingest_into_baseline_table` (its fidelity cells render as "—" / "by
  construction"; see `../demographic-oversampling/FINDINGS.md` §5).
- The exact assembly command is step 3 of the Mission-3 run-book in
  `famail_temporal/baselines/STATUS.md`.
