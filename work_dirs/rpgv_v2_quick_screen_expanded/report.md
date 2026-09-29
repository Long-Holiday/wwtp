# RPGV-v2 quick-screen report

> Screening results reuse weights, a mini validation set, and a reduced crop. Do not report them as independent full-budget ablations.

## Zero-shot mini-val

| Rank | Variant | Foreground IoU | Dice | Delta vs Full |
|---:|---|---:|---:|---:|
| 1 | full | 78.2324 | 87.7870 | +0.0000 |
| 2 | progressive_contour | 78.2324 | 87.7870 | +0.0000 |
| 3 | progressive_weighted | 78.0586 | 87.6774 | -0.1739 |
| 4 | progressive_rgr | 77.9272 | 87.5945 | -0.3053 |
| 5 | progressive_dfgv | 77.9212 | 87.5907 | -0.3112 |
| 6 | progressive_base | 77.8600 | 87.5520 | -0.3724 |

## 512-crop short fine-tune

| Rank | Variant | Foreground IoU | Dice | Delta vs Full |
|---:|---|---:|---:|---:|
| 1 | progressive_contour | 70.8739 | 82.9546 | +0.0527 |
| 2 | full | 70.8212 | 82.9185 | +0.0000 |
| 3 | progressive_base | 70.6289 | 82.7866 | -0.1923 |
| 4 | progressive_rgr | 70.2346 | 82.5151 | -0.5865 |
| 5 | progressive_dfgv | 70.0462 | 82.3849 | -0.7750 |
| 6 | progressive_weighted | 69.9556 | 82.3222 | -0.8655 |

## Selected for official confirmation

`progressive_contour`, `progressive_base`

## Official confirmation command

```bash
python scripts/run_rpgv_v2_ablations.py run --variants progressive_contour progressive_base --seeds 42 --max-iters 100000 --work-root work_dirs/rpgv_v2_quick_screen_expanded/official_confirmation
```
