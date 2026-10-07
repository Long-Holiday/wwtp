# WWTP 遥感语义分割实验

基于 MMSegmentation 1.2.2，统一 WWTP、UAVid2020 和 LoveDA 的模型、训练与评估入口。

主流模型固定为 **HRNet-W18、UNetFormer-R18、SegNeXt-S、SegFormer-B2、DeepLabV3+-R50、Mask2Former-Swin-T、HDNet**。`untformer` 指 UNetFormer，`deeplab` 指 DeepLabV3+。训练批处理、测试批处理和合成冒烟测试使用同一份模型清单：`wwtpseg/experiments/__init__.py`。

RPGV v1、v2、v3、v4、v5、v5.1 的实现、阶段配置、消融、初始化与数值测试全部保留。公开数据集另保留已有 RPGV RGB／完整几何／v2 多分类适配；v3–v5.1 原生二分类实现继续用于 WWTP，未把它们改名成多分类实验。

## 项目结构

```text
configs/_base_/       模型、1024 数据管线、运行与训练协议
configs/experiments/  WWTP 七个主流模型与 RPGV v1/v2 阶段配置
configs/remote_sensing/{uavid,loveda}/  相同的七个主流模型与已有 RPGV 多分类适配
configs/{ablations,ablations_v2,v3,v4,v5,v51}/  全部 RPGV 版本及消融
wwtpseg/              数据、模型、指标、运行时与统一模型清单
tools/                单实验 train/test、预测导出、数据准备及模型测试
scripts/              主流模型批处理与 RPGV 实验编排
tests/                官方划分、标签编码和配置回归测试
docs/                 数据协议与 RPGV 结构说明
docker/               新版训练镜像
```

历史日志、审计、结果、中间可视化与原实验报告归档到本地分支 **`experiment-logs/2026-10-07`**，索引为该分支的 `experiment_archive/README.md` 与 `index.json`。主分支不再追踪 `work_dirs/`、`analysis/`；数据和权重也不入 Git。没有改写已有提交历史。归档分支需按需要另行推送。

```bash
git show experiment-logs/2026-10-07:experiment_archive/README.md
git show experiment-logs/2026-10-07:EXPERIMENT_REPORT.md
```

## Docker 环境

默认升级到 **PyTorch 2.5.1 / CUDA 12.4 / cuDNN 9 devel**，保留 MMEngine 0.10.7、MMCV 2.1.0、MMDetection 3.3.0、MMSeg 1.2.2。MMSeg 1.2.2 的官方版本约束是 MMCV `<2.2.0`；较新 PyTorch 没有对应的 MMCV 2.1 官方轮子，因此镜像从源码编译真实 CUDA 算子，不放宽版本检查、不用旧 ABI 轮子。镜像不再安装 Mamba CUDA 扩展。源码编译首次耗时较长，后续 Docker 层可复用。

```bash
docker compose build
docker compose run --rm wwtp python tools/smoke_test.py --device cuda --backward
```

`TORCH_CUDA_ARCH_LIST` 默认 `8.6;8.9+PTX`（RTX 3050／3090／4090 等），其他显卡构建时覆盖；`MAX_JOBS=2` 限制编译内存。Compose 缓存卷替代原来写死的个人 home 路径。

提供已知兼容的旧环境回退，使用官方预编译 MMCV 轮子：

```bash
PYTORCH_IMAGE=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel \
MMCV_FROM_SOURCE=0 WWTP_IMAGE=wwtp-mmseg:torch2.1.2-cu121 docker compose build
# 运行时仍用同一 WWTP_IMAGE 选择回退镜像
WWTP_IMAGE=wwtp-mmseg:torch2.1.2-cu121 docker compose run --rm wwtp python tools/smoke_test.py
```

依赖源可用 `PIP_INDEX_URL` 覆盖；Linux 上需要主机网络构建时可设置 `BUILD_NETWORK=host`。Docker daemon 的拉取代理与容器内依赖下载代理需分别配置，后者可通过 Docker 预定义的 `HTTP_PROXY`／`HTTPS_PROXY` 构建参数传入。

**验证状态：按本次要求只修改代码与文件，镜像构建已停止，未运行 Docker 模型／训练测试。新版 MMCV 源码编译兼容性、实际吞吐与 24GB 显卡显存占用均待在目标环境验证。** 下文列出后续验证入口；不代表本次已经通过这些测试。

## 数据与训练协议

WWTP 默认根目录 `wwtp_semantic_dataset_1m/`，**1024×1024 整图、1 m/px**，沿用 v5 的光度扰动和精确翻转，不在线放大到 2048、不随机裁小块。误用非 1024 数据会明确报错。已有原始 WWTP 数据可用 `tools/resample_wwtp_dataset.py` 转换；影像面积采样、掩膜最近邻，划分保持不变。

公开集仅使用 **UAVid2020、LoveDA**，本项目不下载数据。准备工具只读取官方解压目录。LoveDA 严格 `2522/1669/1796` 张；UAVid2020 严格 `200/70/150` 帧，按官方 sequence 保留 split，检查每个 sequence 十帧、划分间无 sequence 重叠。准备失败不会生成可用于训练的完整数据目录。详细目录与类别编码见 [数据协议](docs/datasets.md)。

```bash
docker compose run --rm wwtp python tools/remote_sensing/prepare.py loveda \
  --source data/remote_sensing/raw/loveda --output data/remote_sensing/prepared/loveda
docker compose run --rm wwtp python tools/remote_sensing/prepare.py uavid \
  --source data/remote_sensing/raw/uavid --output data/remote_sensing/prepared/uavid
```

LoveDA 保留官方 1024 整图；UAVid 仅在官方训练 sequence 内切成 1024 块，边缘补零、标签补 255，不丢失边缘像素。验证与官方测试保留完整原始帧，用 1024／stride 768 滑窗拼接。**200 是原始训练帧数，不是切片后的样本数。** 不随机拆帧、不合并 train/val。

七个主流模型使用统一预算：batch 2 × 累积 4 = 有效 batch 8，20,000 个数据加载迭代 = 5,000 次优化器更新；每 1,000 迭代验证，不默认早停。保留各模型优化器参数组；SegNeXt 和 Mask2Former 使用 FP32，其余默认 AMP。WWTP 指标含前景、边界与拓扑，边界容差 1.5 px=1.5 m；公开多分类集使用 mIoU/mDice/mFscore，按验证 mIoU 选权重。

## 训练与测试

```bash
# 单模型：train/test 始终使用同一份配置
docker compose run --rm wwtp python tools/train.py configs/experiments/hrnet.py
docker compose run --rm wwtp python tools/train.py configs/remote_sensing/uavid/hrnet.py
docker compose run --rm wwtp python tools/train.py configs/remote_sensing/loveda/hrnet.py

# 批处理：先查看所有七个模型的计划
docker compose run --rm wwtp python scripts/run_experiments.py train --dataset wwtp --dry-run
docker compose run --rm wwtp python scripts/run_experiments.py train --dataset loveda
docker compose run --rm wwtp python scripts/run_experiments.py test --dataset loveda

# 恢复原优化器、随机状态和调度器
docker compose run --rm wwtp python tools/train.py configs/experiments/hrnet.py --resume

# 评估指定 checkpoint
docker compose run --rm wwtp python tools/test.py configs/experiments/hrnet.py \
  work_dirs/wwtp/hrnet/iter_20000.pth --split val
```

默认输出为 `work_dirs/<dataset>/<model>/`。`--cfg-options` 可覆盖 batch、累积、workers 等；变更 batch 时同步调整累积和迭代预算，保持有效 batch 与更新数。多卡批处理用 `--gpus N`。

训练入口统一启用 TF32，数据加载使用 persistent workers 与 pin memory。删除训练循环与确定性设置的全局 monkey patch，恢复 MMEngine 原生断点续训语义。AMP 模型可用 `--amp-dtype bfloat16`，设备必须支持 BF16；FP32 配置不被该参数强制切换。测试/恢复从完整 checkpoint 加载时关闭骨干预训练下载。

UAVid/LoveDA 官方测试标签不公开，`tools/test.py` 的公开集配置明确评估官方 **val**。测试集预测用独立导出入口，不生成伪测试分数：

```bash
docker compose run --rm wwtp python tools/remote_sensing/predict.py \
  configs/remote_sensing/loveda/segformer.py work_dirs/loveda/segformer/BEST.pth \
  --output work_dirs/loveda/segformer/submission
```

LoveDA 提交沿用官方预测脚本：导出 **0–6 ID**，以原图片编号直接保存到输出目录；原始真值的 1–7 编号只用于读取训练标签。UAVid 导出与官方颜色匹配的 RGB mask，目录为 `seq*/Labels/<原帧名>.png`。

## RPGV

RPGV 的结构开关、损失和初始化规则继续保留。v1/v2 阶段配置改用相同 1024 WWTP 根目录与整图增强；v3 锚点初始化和 v5.1 权重迁移仍需运行原初始化工具。阶段训练预算按各版本原方案保留；严格等预算对照可用 `configs/v5/controls/`，不要把阶段实验与七个统一预算基线宣称为等预算比较。

```bash
docker compose run --rm wwtp python tools/generate_pseudo_geometry.py wwtp_semantic_dataset_1m --device cuda
docker compose run --rm wwtp bash scripts/train_rpgv_v5.sh
docker compose run --rm wwtp python tools/smoke_test.py --include-rpgv --backward --device cuda
```

伪几何文件为 `pseudo_geometry/<split>/<image_stem>.npz`，与准备后的 1024 图像逐像素对齐。所有 RPGV 模型仍在 `wwtpseg/models/segmentors/`，消融和数值测试入口仍可直接调用。

## 小规模验证

无需数据集和预训练权重下载：

```bash
docker compose run --rm wwtp python -m unittest discover -s tests -p 'test_experiment_protocol.py' -v
docker compose run --rm wwtp python tools/smoke_test.py --dataset wwtp --backward --device cuda
docker compose run --rm wwtp python tools/smoke_test.py --dataset uavid --backward --device cuda
docker compose run --rm wwtp python tools/smoke_test.py --dataset loveda --backward --device cuda
docker compose run --rm wwtp python tools/smoke_runner.py --device cuda
```

模型测试使用 64 像素、batch 2，检查前向、反向、有限梯度和预测类别/尺寸；Runner 测试执行 2 步训练、保存、测试，再恢复至 4 步。它们验证代码路径，不代表完整训练质量或最终 1024 分辨率显存要求。

模型组件来源：[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2)、[UNetFormer/GeoSeg](https://github.com/WangLibo1995/GeoSeg)、[HDNet](https://github.com/danfenghong/ISPRS_HD-Net)。HDNet 的 GPL-3.0 许可证保留在 `wwtpseg/edge_baselines/`。
