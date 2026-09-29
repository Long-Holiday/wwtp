# RPGV v2 ablations — evaluation split: test

Missing results are `-`. Incomplete budgets are excluded from means and deltas.

| protocol | variant | seed | status | last_val_iter | best_iter | val_iou | delta_val_iou | iou | boundary_f1 | hd95_px | island_pixels | holes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | full | 42 | evaluated | 40000 | 38000 | 78.4041 | 0.0000 | 78.5600 | 17.1286 | 253.8957 | 5265922 | 507 |
| full | no_contour | 42 | planned | - | - | - | - | - | - | - | - | - |
| full | no_frequency_validation | 42 | planned | - | - | - | - | - | - | - | - | - |
| full | no_rgr | 42 | planned | - | - | - | - | - | - | - | - | - |
| full | no_structure_regularization | 42 | planned | - | - | - | - | - | - | - | - | - |
| full | unweighted_fusion | 42 | planned | - | - | - | - | - | - | - | - | - |

## Seed statistics

Sample SD (n−1); one seed has no SD. Deltas pair each seed with full in the same protocol.

| protocol | variant | n / planned | Val IoU mean ± SD | paired n | ΔVal IoU mean ± SD |
| --- | --- | --- | --- | --- | --- |
| full | full | 1 / 1 | 78.4041 ± - | 1 | 0.0000 ± - |
| full | no_contour | 0 / 1 | - ± - | 0 | - ± - |
| full | no_frequency_validation | 0 / 1 | - ± - | 0 | - ± - |
| full | no_rgr | 0 / 1 | - ± - | 0 | - ± - |
| full | no_structure_regularization | 0 / 1 | - ± - | 0 | - ± - |
| full | unweighted_fusion | 0 / 1 | - ± - | 0 | - ± - |
