# RPGV-v2 quick-screen report

> Screening results reuse weights, a mini validation set, and a reduced crop. Do not report them as independent full-budget ablations.

## 512-crop short fine-tune

| Rank | Variant | Foreground IoU | Dice | Delta vs Full |
|---:|---|---:|---:|---:|
| 1 | progressive_rgr | 71.3886 | 83.3061 | +0.2793 |
| 2 | progressive_contour | 71.1981 | 83.1763 | +0.0888 |
| 3 | full | 71.1093 | 83.1157 | +0.0000 |
| 4 | progressive_base | 71.0301 | 83.0615 | -0.0793 |
| 5 | progressive_dfgv | 70.8146 | 82.9140 | -0.2947 |
| 6 | progressive_weighted | 70.3467 | 82.5924 | -0.7626 |

## Selected for official confirmation

`progressive_rgr`, `progressive_contour`

## Official confirmation command

```bash
python scripts/run_rpgv_v2_ablations.py run --variants progressive_rgr progressive_contour --seeds 42 --max-iters 100000 --work-root work_dirs/rpgv_v2_quick_screen/official_confirmation
```
