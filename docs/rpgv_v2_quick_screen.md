# RPGV-v2 快速消融筛选

这套流程用于快速决定“哪些想法值得跑正式实验”，不会改写或复用正式消融目录：

1. 从验证集按前景覆盖率四分位固定抽取 64 张图；
2. 所有变体加载同一个 Full checkpoint，进行零训练消融；
3. 选出前 3 个变体，以 512×512 crop、3000 iter 做短微调；
4. 根据短微调结果选出 1–2 个方案，生成独立 100k 正式训练命令。

快速结果复用了权重、缩小了验证集和训练分辨率，不能作为独立、完整预算的论文消融结果。

## 一条命令跑完快速流程

```bash
docker compose run --rm wwtp python scripts/run_rpgv_v2_quick_screen.py all \
  --checkpoint work_dirs/rpgv_v2_joint_fixed/best_binary_Foreground_IoU_iter_64000.pth \
  --variants progressive \
  --work-root work_dirs/rpgv_v2_quick_screen \
  --mini-val-size 64 \
  --top-k 3 \
  --official-top-k 2
```

如果已经进入项目容器，可以去掉命令开头的 `docker compose run --rm wwtp`。

`all` 只执行 mini-val、零训练评估和短微调。它会生成正式训练脚本，但不会默认启动耗时很长的 100k 实验。

主要输出：

- `mini_val.json`：固定样本、原始数据索引和前景覆盖率；
- `zero_shot_summary.json`：零训练消融排名；
- `finetune_summary.json`：512 crop 短微调排名；
- `report.md`：两阶段对比表和最终入选方案；
- `run_official_confirmation.sh`：正式确认命令。

## 分阶段运行

仅生成或检查固定 mini-val：

```bash
docker compose run --rm wwtp python scripts/run_rpgv_v2_quick_screen.py prepare \
  --work-root work_dirs/rpgv_v2_quick_screen
```

只做零训练消融：

```bash
docker compose run --rm wwtp python scripts/run_rpgv_v2_quick_screen.py zero-shot \
  --checkpoint work_dirs/rpgv_v2_joint_fixed/best_binary_Foreground_IoU_iter_64000.pth \
  --variants paper \
  --work-root work_dirs/rpgv_v2_quick_screen
```

手工指定短微调候选项：

```bash
docker compose run --rm wwtp python scripts/run_rpgv_v2_quick_screen.py finetune \
  --checkpoint work_dirs/rpgv_v2_joint_fixed/best_binary_Foreground_IoU_iter_64000.pth \
  --variants paper \
  --candidates no_rgr no_contour \
  --work-root work_dirs/rpgv_v2_quick_screen
```

仅根据已有短微调结果重新生成正式命令：

```bash
docker compose run --rm wwtp python scripts/run_rpgv_v2_quick_screen.py official \
  --work-root work_dirs/rpgv_v2_quick_screen \
  --official-top-k 2
```

确认排名后，运行生成的脚本：

```bash
docker compose run --rm wwtp \
  bash work_dirs/rpgv_v2_quick_screen/run_official_confirmation.sh
```

也可以显式添加 `--run-official`，让 runner 在快速筛选完成后立即开始正式实验。

## 常用调整

- 显存充足：`--train-batch-size 4`；
- 显存不足：`--train-batch-size 1`；
- 更激进筛选：`--train-iters 1500 --mini-val-size 32`；
- 提高短微调可信度：`--train-iters 5000 --mini-val-size 128`；
- 从中断的短微调恢复：添加 `--resume-finetune`；
- 多 GPU 并行时，为每个进程指定不同的 `CUDA_VISIBLE_DEVICES` 和 `--work-root`。

正式确认仍通过 `scripts/run_rpgv_v2_ablations.py` 在全分辨率、全验证集、独立初始化下运行；快速筛选目录不会被正式汇总脚本读取。
