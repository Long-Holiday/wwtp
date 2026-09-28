# Potsdam 完整瓦片实验

## 数据与划分

实验读取 `data/remote_sensing/prepared/potsdam/2_Ortho_RGB/2_Ortho_RGB/` 中的
38 张原始 RGB PNG，标注直接读取官方彩色 TIFF。原始 6000×6000、0.05 m/像素瓦片
在读取时整体缩至 1536×1536，保留每张 300 m×300 m 的场景范围；实验网格的有效
分辨率是 0.1953125 m/像素。指标在这个共同网格上计算，不能当作原始 5 cm 网格的
ISPRS 官方排行榜成绩。所有模型使用相同的空间处理与划分：

| 集合 | 瓦片 | 数量 |
| --- | --- | ---: |
| Train | 官方参与者标注中除 `2_10`、`7_10` 外的瓦片 | 22 |
| Val | `2_10` | 1 |
| Test | 官方历史测试瓦片，使用 `5_Labels_all` 的公开完整参考 | 14 |

`7_10` 因标注问题排除。黑色无效像素忽略；白、蓝、青、绿、黄、红依次对应
impervious surface、building、low vegetation、tree、car、clutter。五类指标
排除 clutter，验证集 mIoU 只用于保存最佳权重，测试集不参与模型选择。

## 模型与训练

| 模型 | 配置 |
| --- | --- |
| DeepLabV3+ | `configs/remote_sensing/potsdam_original_deeplabv3plus.py` |
| Mask2Former | `configs/remote_sensing/potsdam_original_mask2former.py` |
| SegFormer | `configs/remote_sensing/potsdam_original_segformer.py` |
| UNetFormer | `configs/remote_sensing/potsdam_original_unetformer.py` |
| RPGV v2 + Depth Anything V2 | `configs/remote_sensing/potsdam_original_rpgv_v2_depth_anything.py` |
| RPGV v2 + 官方 nDSM | `configs/remote_sensing/potsdam_original_rpgv_v2_ndsm.py` |

RPGV v2 的两组实验使用完全相同的六分类模型。此模型将二分类 RPGVNetV2 的
加法多尺度解码与单个有界轮廓修正头移植到多分类模型；几何校正、可靠度验证与
全局上下文继承原有多分类 RPGV 实现。Depth Anything V2 对整张影像缩放后的视图
生成伪深度和多视图一致性可靠度。官方几何分支读取数据集提供的
`normalized_lastools.jpg` nDSM，而非绝对高程 DSM。这些文件是 6000×6000 的
8 位归一化地表高度图；生成脚本将其直接缩放至 1536×1536，保留 0–255 数值含义，
不再做逐瓦片百分位归一化。深度来源不同之外，其余设置一致。

所有模型单卡 batch size 1、AMP、随机种子 42、固定 3000 iterations，每 250
iterations 在 `2_10` 验证。**不使用早停**。最终用验证 mIoU 最佳的 checkpoint
在 14 张测试瓦片评估六类 mIoU 和五类 mIoU/mF1/OA。

## 命令

```bash
# 官方 nDSM 的 37 张实验瓦片可在 CPU 上准备，不占用训练 GPU
docker compose run --rm --no-deps wwtp python tools/remote_sensing/prepare_potsdam_full_ndsm.py

# 当前训练结束后，由你手动执行：生成 Depth Anything 深度，
# 随后按顺序训练并评估六个模型；GPU 忙碌时命令会直接报错
python3 scripts/run_potsdam_original_experiments.py all

# 查看进度
python3 scripts/run_potsdam_original_experiments.py status
```

如果只想先生成 Depth Anything 深度，运行
`python3 scripts/run_potsdam_original_experiments.py prepare-depth`。逐模型运行使用
`python3 scripts/run_potsdam_original_experiments.py train --model MODEL`、
`python3 scripts/run_potsdam_original_experiments.py eval --model MODEL`，
其中 `MODEL` 是配置文件去掉 `potsdam_original_` 后的名称。结果写入
`work_dirs/remote_sensing/potsdam_original_*_eval/metrics.json`，六组全部完成后
汇总为 `work_dirs/remote_sensing/potsdam_original_results.json`。已有旧版 512
补丁实验的 checkpoint 和指标不混入此次实验。
