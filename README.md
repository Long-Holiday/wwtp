# WWTP 遥感语义分割实验

基于 MMSegmentation 1.2.2，统一 WWTP、UAVid2020 和 LoveDA 的模型、训练与评估入口。

主流模型固定为 **HRNet-W18、UNetFormer-R18、SegNeXt-S、SegFormer-B2、DeepLabV3+-R50、Mask2Former-Swin-T、HDNet**。`untformer` 指 UNetFormer，`deeplab` 指 DeepLabV3+。训练批处理、测试批处理和合成冒烟测试使用同一份模型清单：`wwtpseg/experiments/__init__.py`。

RPGV v1、v2、v3、v4、v5、v5.1 的实现、阶段配置、消融、初始化与数值测试全部保留，并新增 v6 的可变形深度校正、频域几何和区域级双向对比学习。公开数据集另保留已有 RPGV RGB／完整几何／v2 多分类适配；v3–v6 原生二分类实现继续用于 WWTP，未把它们改名成多分类实验。

## 项目结构

```text
configs/
  _base_/             共享模型、1024 数据管线、运行与训练协议
  wwtp/               WWTP 七个主流模型
  uavid/              UAVid 同一组七个主流模型及已有 RPGV 多分类配置
  loveda/             LoveDA 同一组七个主流模型及已有 RPGV 多分类配置
  rpgv/{v1,v2,v3,v4,v5,v51,v6}/  各版本阶段、消融与对照配置
wwtpseg/
  models/segmentors/  RPGV 各版本、hdnet/、multiclass/ 多分类适配
  datasets/           数据集与数据变换
  evaluation/         边界和拓扑指标
  engine/             训练运行时
  experiments/        训练、测试共用的模型清单
tools/
  train.py, test.py   单实验训练与评估入口
  data/              数据准备、WWTP 重采样与伪几何生成
  evaluation/        预测导出、可视化、拓扑分析与结果汇总
  rpgv/              RPGV 初始化、诊断、性能分析和消融评估
scripts/
  run_experiments.py  七个主流模型统一批处理
  rpgv/              RPGV 阶段训练与消融编排
tests/
  test_experiment_protocol.py  官方划分、标签与配置回归
  smoke_test.py, smoke_runner.py  模型与训练流程冒烟测试
  rpgv/              RPGV 各版本数值测试与训练冒烟测试
docs/                数据协议与 RPGV 结构说明
docker/              训练镜像构建文件
```

历史日志、审计、结果、中间可视化与原实验报告归档到本地分支 **`experiment-logs/2026-10-07`**，索引为该分支的 `experiment_archive/README.md` 与 `index.json`。主分支不再追踪 `work_dirs/`、`analysis/`；数据和权重也不入 Git。没有改写已有提交历史。归档分支需按需要另行推送。

```bash
git show experiment-logs/2026-10-07:experiment_archive/README.md
git show experiment-logs/2026-10-07:EXPERIMENT_REPORT.md
```

## Docker 环境

默认使用适合本项目 OpenMMLab 依赖的成熟组合：**PyTorch 2.1.2 / torchvision 0.16.2 / CUDA 12.1 / cuDNN 8 runtime**，配套 MMEngine 0.10.7、MMCV 2.1.0、MMDetection 3.3.0、MMSeg 1.2.2。选择依据是依赖兼容性和安装便利性：MMSeg 1.2.2 [要求 MMCV `<2.2.0`](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/__init__.py)，OpenMMLab 提供对应 PyTorch 2.1 / CUDA 12.1 的 [MMCV 2.1 官方预编译包](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html)。默认直接安装包含 CUDA 算子的 wheel，无需源码编译；使用 runtime 基础镜像，减少 CUDA 编译工具链的体积和维护成本。

```bash
docker compose build
docker compose run --rm wwtp python tests/smoke_test.py --device cuda --backward
```

新镜像构建并通过冒烟测试后，确认旧容器不再使用旧版镜像，再清理本项目原来的镜像：

```bash
docker image rm wwtp-mmseg:torch2.5.1-cu124
```

默认镜像名为 `wwtp-mmseg:torch2.1.2-cu121`，可用 `WWTP_IMAGE` 自定义镜像名，构建与运行时保持一致。基础镜像和 MMCV wheel 下载地址固定为同一套兼容组合，移除原来的源码编译开关、显卡架构和编译并行度参数。默认针对 Linux x86_64 NVIDIA GPU 环境，宿主机需配置支持 CUDA 12.1 的 NVIDIA 驱动和 NVIDIA Container Toolkit。Compose 使用命名缓存卷，避免写死个人 home 路径。

构建使用 Docker Hub、Ubuntu 官方仓库、PyPI 原始源与 OpenMMLab 官方下载站，不使用镜像源。APT／pip 下载缓存保存在 BuildKit 中，不进入最终镜像；基础 Python 依赖、MMCV wheel、项目依赖与代码分层缓存。MMCV 强制使用二进制包，找不到匹配的 wheel 时直接报错，避免意外回退到耗时的源码编译。

构建使用 Docker 默认网络，执行 `docker compose build` 即可，无需指定网络模式、代理地址或端口。

公共依赖版本在 `requirements/core.txt` 统一维护，包含与基础镜像一致的 PyTorch／torchvision 固定版本；项目依赖在 `requirements/runtime.txt`。构建期间以公共依赖清单约束 MMCV，避免依赖解析意外升级 PyTorch 或 NumPy。镜像构建包含依赖一致性检查、CUDA 扩展导入检查与项目模块导入检查，不启动训练。

实际吞吐与 24GB 显卡在完整 1024 分辨率训练时的显存占用需在目标机器验证；下文提供无需数据集和预训练权重的小规模验证入口。

## 数据与训练协议

WWTP 默认根目录 `wwtp_semantic_dataset_1m/`，**1024×1024 整图、1 m/px**，沿用 v5 的光度扰动和精确翻转，不在线放大到 2048、不随机裁小块。误用非 1024 数据会明确报错。已有原始 WWTP 数据可用 `tools/data/resample_wwtp_dataset.py` 转换；影像面积采样、掩膜最近邻，划分保持不变。

公开集仅使用 **UAVid2020、LoveDA**，本项目不下载数据。准备工具只读取官方解压目录。LoveDA 严格 `2522/1669/1796` 张；UAVid2020 严格 `200/70/150` 帧，按官方 sequence 保留 split，检查每个 sequence 十帧、划分间无 sequence 重叠。准备失败不会生成可用于训练的完整数据目录。详细目录与类别编码见 [数据协议](docs/datasets.md)。

```bash
docker compose run --rm wwtp python tools/data/prepare_public.py loveda \
  --source data/remote_sensing/raw/loveda --output data/remote_sensing/prepared/loveda
docker compose run --rm wwtp python tools/data/prepare_public.py uavid \
  --source data/remote_sensing/raw/uavid --output data/remote_sensing/prepared/uavid
```

LoveDA 保留官方 1024 整图；UAVid 仅在官方训练 sequence 内切成 1024 块，边缘补零、标签补 255，不丢失边缘像素。验证与官方测试保留完整原始帧，用 1024／stride 768 滑窗拼接。**200 是原始训练帧数，不是切片后的样本数。** 不随机拆帧、不合并 train/val。

七个主流模型使用统一预算：batch 2 × 累积 4 = 有效 batch 8，20,000 个数据加载迭代 = 5,000 次优化器更新；每 1,000 迭代验证，不默认早停。保留各模型优化器参数组；SegNeXt 和 Mask2Former 使用 FP32，其余默认 AMP。WWTP 指标含前景、边界与拓扑，边界容差 1.5 px=1.5 m；公开多分类集使用 mIoU/mDice/mFscore，按验证 mIoU 选权重。

## 训练与测试

```bash
# 单模型：train/test 始终使用同一份配置
docker compose run --rm wwtp python tools/train.py configs/wwtp/hrnet.py
docker compose run --rm wwtp python tools/train.py configs/uavid/hrnet.py
docker compose run --rm wwtp python tools/train.py configs/loveda/hrnet.py

# 批处理：先查看所有七个模型的计划
docker compose run --rm wwtp python scripts/run_experiments.py train --dataset wwtp --dry-run
docker compose run --rm wwtp python scripts/run_experiments.py train --dataset loveda
docker compose run --rm wwtp python scripts/run_experiments.py test --dataset loveda

# 恢复原优化器、随机状态和调度器
docker compose run --rm wwtp python tools/train.py configs/wwtp/hrnet.py --resume

# 评估指定 checkpoint
docker compose run --rm wwtp python tools/test.py configs/wwtp/hrnet.py \
  work_dirs/wwtp/hrnet/iter_20000.pth --split val
```

默认输出为 `work_dirs/<dataset>/<model>/`。`--cfg-options` 可覆盖 batch、累积、workers 等；变更 batch 时同步调整累积和迭代预算，保持有效 batch 与更新数。多卡批处理用 `--gpus N`。

训练入口统一启用 TF32，数据加载使用 persistent workers 与 pin memory。删除训练循环与确定性设置的全局 monkey patch，恢复 MMEngine 原生断点续训语义。AMP 模型可用 `--amp-dtype bfloat16`，设备必须支持 BF16；FP32 配置不被该参数强制切换。测试/恢复从完整 checkpoint 加载时关闭骨干预训练下载。

UAVid/LoveDA 官方测试标签不公开，`tools/test.py` 的公开集配置明确评估官方 **val**。测试集预测用独立导出入口，不生成伪测试分数：

```bash
docker compose run --rm wwtp python tools/evaluation/predict_public.py \
  configs/loveda/segformer.py work_dirs/loveda/segformer/BEST.pth \
  --output work_dirs/loveda/segformer/submission
```

LoveDA 提交沿用官方预测脚本：导出 **0–6 ID**，以原图片编号直接保存到输出目录；原始真值的 1–7 编号只用于读取训练标签。UAVid 导出与官方颜色匹配的 RGB mask，目录为 `seq*/Labels/<原帧名>.png`。

## RPGV

RPGV 的结构开关、损失和初始化规则继续保留。v1/v2 阶段配置改用相同 1024 WWTP 根目录与整图增强；v3 锚点初始化和 v5.1 权重迁移仍需运行原初始化工具。阶段训练预算按各版本原方案保留；严格等预算对照可用 `configs/rpgv/v5/controls/`，不要把阶段实验与七个统一预算基线宣称为等预算比较。

```bash
docker compose run --rm wwtp python tools/data/generate_pseudo_geometry.py wwtp_semantic_dataset_1m --device cuda
docker compose run --rm wwtp bash scripts/rpgv/train_rpgv_v5.sh
docker compose run --rm wwtp python tests/smoke_test.py --include-rpgv --backward --device cuda
```

伪几何文件为 `pseudo_geometry/<split>/<image_stem>.npz`，与准备后的 1024 图像逐像素对齐。所有 RPGV 模型仍在 `wwtpseg/models/segmentors/`，消融和数值测试入口仍可直接调用。

### RPGV v6

v6 将伪深度经过几何独立的可变形卷积校正后，做 FP32 互补频带分解；高频空间聚类原型与对应区域的 RGB 外观原型进行双向 InfoNCE，低频与 RGB 进行受限 token 的交叉注意力，最终经可靠度门控融入 RGB 特征。无深度或 Q=0 时精确回退到 RGB 路径，不需要 CLIP 权重或文本编码器。

```bash
docker compose run --rm wwtp bash scripts/rpgv/train_rpgv_v6.sh
docker compose run --rm wwtp python -m unittest tests.rpgv.test_rpgv_v6 -v
docker compose run --rm wwtp python tests/smoke_runner.py \
  --config configs/rpgv/v6/rpgv_v6.py --device cpu
```

默认从 ImageNet RGB 骨干初始化联合训练，不直接续用 v5.1 完整权重。提供 `configs/rpgv/v6/no_contrastive.py` 和 `rgb_only.py` 两个消融，训练预算与 v5 相同。架构、损失、可靠度过滤、使用方式及已知限制见 [v6 设计说明](docs/rpgv_v6_design.md)。

## 小规模验证

无需数据集和预训练权重下载：

```bash
docker compose run --rm wwtp python -m unittest discover -s tests -p 'test_experiment_protocol.py' -v
docker compose run --rm wwtp python tests/smoke_test.py --dataset wwtp --backward --device cuda
docker compose run --rm wwtp python tests/smoke_test.py --dataset uavid --backward --device cuda
docker compose run --rm wwtp python tests/smoke_test.py --dataset loveda --backward --device cuda
docker compose run --rm wwtp python tests/smoke_runner.py --device cuda
```

模型测试使用 64 像素、batch 2，检查前向、反向、有限梯度和预测类别/尺寸；Runner 测试执行 2 步训练、保存、测试，再恢复至 4 步。它们验证代码路径，不代表完整训练质量或最终 1024 分辨率显存要求。

模型组件来源：[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2)、[UNetFormer/GeoSeg](https://github.com/WangLibo1995/GeoSeg)、[HDNet](https://github.com/danfenghong/ISPRS_HD-Net)。HDNet 的 GPL-3.0 许可证保留在 `wwtpseg/models/segmentors/hdnet/`。
