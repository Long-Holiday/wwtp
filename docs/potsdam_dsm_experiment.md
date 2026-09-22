# Potsdam 官方 DSM 替代 Depth Anything 实验

此实验使用 `Potsdam.zip` 内的 `1_DSM_normalisation.zip`，选取每张瓦片的
`dsm_potsdam_XX_YY_normalized_lastools.jpg`。这 38 张单通道 6000×6000 图是
数据集随附的归一化 DSM，编码为 8 位 JPEG；**不保留原始 `1_DSM.rar` 中 TIFF
的浮点或高位深精度**。训练时深度输入是这些灰度值除以 255。原 Depth Anything
实验的数据和输出均不修改。

转换脚本沿用 `prepared/potsdam/manifest.json` 的 20/4/14 瓦片划分及 512
切片坐标，写入新目录 `prepared/potsdam/dsm_geometry/{train,val,test}`。
每个 `.npz` 仍含 `depth` 和 `reliability`：后者在瓦片内为 255，边缘补齐区为
0。该值表示已知 DSM 的覆盖区域，不是 Depth Anything 的多视图一致性分数。
模型结构、随机种子、增强、优化器、40k 训练步数和 14 张测试瓦片的评估指标
继承自 `potsdam_rpgv_full_geometry.py`。因此实验差异包括深度来源及其对应的
可靠度先验。

## 命令

从仓库根目录运行。各命令相互独立；只运行 `status` 不会使用 GPU。

```bash
python3 scripts/run_potsdam_rpgv_dsm.py prepare
python3 scripts/run_potsdam_rpgv_dsm.py status
python3 scripts/run_potsdam_rpgv_dsm.py train
python3 scripts/run_potsdam_rpgv_dsm.py eval
```

`prepare` 只执行 CPU 数据转换。若转换中断，使用下面的命令在 Docker 内续跑；
已有有效 `.npz` 不会被覆盖。转换前可加 `--dry-run` 检查官方 ZIP 与 RGB
切片清单：

```bash
docker compose run --rm --no-deps wwtp python tools/remote_sensing/prepare_potsdam_dsm.py --dry-run
docker compose run --rm --no-deps wwtp python tools/remote_sensing/prepare_potsdam_dsm.py --resume
```

训练输出专属 `work_dirs/remote_sensing/potsdam_rpgv_full_dsm/`；评估输出专属
`work_dirs/remote_sensing/potsdam_rpgv_full_dsm_eval/`，包含 `metrics.json`。
评估入口默认读取 DSM 训练的 `iter_40000.pth`，不会读取 Depth Anything 的
checkpoint。报告时使用 `full/mIoU5`、`full/mF15` 和 `full/OA5`，与现有 Potsdam
测试一致。可先加 `--dry-run` 查看训练或评估命令。

当前若有模型占用 GPU，先完成 `prepare`；在 GPU 空闲时再显式运行 `train`
和 `eval`。这些脚本不会停止已有进程。
