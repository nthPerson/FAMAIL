# Teaser design archive (2026-07-16)

Rejected/superseded candidates from the teaser design rounds. The chosen
design lives one level up as `figure-1.tex` (integrated into
`sections/01_introduction.tex` as `fig:teaser`); iterate there, not here.

| File | What it was |
|---|---|
| `teaser-c4.tex` | Round 1, Concept 4: results-chart teaser — external-metric dumbbells (DP/DI/Theil + under-served service with the pinned trim-only marker) bridged to the weighted-BC dose-response panel. Rejected: reads as a results figure, not a motivation figure. |
| `teaser-c3.tex` | Round 1, Concept 3: three stacked chart panels (service-gap bars / dumbbells / dose-response). Rejected for the same reason. |
| `teaser-v1.tex` | Round 2, Variant 1 as first rendered ("the city, the data, two futures" + loop-back arrow). CHOSEN, then promoted to `../figure-1.tex` with Robert's edits (explicit strip-(c) title, rerouted imitate-as-is arrow, renamed model boxes). Kept here as the pre-edit snapshot. |
| `teaser-v2.tex` | Round 2, Variant 2: loop-only "break the cycle" diagram. Runner-up; smaller (~0.12 page) but weaker on the data-worth-preserving requirement. |
| `gen_coords.py` | Coordinate generator for the c3/c4 dose-response panels (reads `PAPER/supply-lift/data/a10/shz_a10_weighted_bc_paired_stats.json`, emits TikZ coords). Only relevant to c3/c4. NOTE: its repo-root discovery (`parents[3]`) assumed it sat in `paper/figures/figure-1/`; run it from a checkout of that location or fix the path if revived. |
| `preview-*.png` | 200 dpi renders of each candidate as reviewed. |

Numbers provenance: each `.tex` carries `% src:` comments; all values are
alpha*-era (0.1, 0.8, 0.1), s10 corpus, from `PAPER/supply-lift/data/a10/`.
