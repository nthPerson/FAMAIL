# Dataset Cleanup Summary (E31)

`removal_rate = n_removed / total_extracted`, where `total_extracted = seeking + driving` trajectories. The rate is over **all** extracted trajectories, not seeking alone — do not read it as the fraction of *seeking* trips removed (that fraction is ~90%).

| Metric | Dirty | Clean | Delta |
|--------|-------|-------|-------|
| n_removed | 195,840 | 119,290 | -76,550 |
| removal_rate (= n_removed / total_extracted) | 0.4975 | 0.3895 | -0.1080 |
| total_extracted (seeking + driving) | 393,670 | 306,269 | -87,401 |
| total_seeking_extracted | 214,286 | 133,091 | -81,195 |
| total_driving_extracted | 179,384 | 173,178 | — |
| n_sink_cells | — | 10 | — |
| phantom_pickups_removed | — | 106,677 | — |
