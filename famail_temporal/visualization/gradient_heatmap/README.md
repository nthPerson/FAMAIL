# Gradient Heatmap Explorer

Interactive Streamlit tool to inspect per-cell objective-function gradients
(`F_spatial`, `F_causal`), per-cell attribution, and pickup concentration on the
48×90 Shenzhen grid, one of 24 hourly slices at a time, with district boundaries.

## Setup

```bash
# 1. One-time precompute (CPU; needs the famail_temporal preprocess cache)
python -m famail_temporal.visualization.gradient_heatmap.precompute

# 2. Launch the app
streamlit run famail_temporal/visualization/gradient_heatmap/app.py
```

## Controls
- **Quantity:** Gradient · Attribution · Concentration
- **Term:** F_spatial · F_causal · F_fidelity · Combined · Spatial+Causal
- **Hour:** 0–23 slider + prev/next
- **Display:** |magnitude|, shared-vs-per-slice scale, percentile clip, district
  boundaries, concentration contour overlay, concentration side panel, α sliders
- **Export:** "Download publication PNG"

## Notes
- Orientation: y_grid horizontal (West→East), x_grid vertical with **South at the
  bottom** — verified against geography and the ArcGIS screenshot; guarded by
  `geometry.assert_canonical_orientation`.
- `F_fidelity` has no per-cell spatial gradient (≈0 by construction) and renders
  flat. `Combined ≡ Spatial+Causal` at the per-cell level.
- The cache (`cache/gradient_viz_bundle.npz`) is a derived artifact (gitignored);
  rerun precompute after a dataset change.
