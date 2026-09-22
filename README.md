# WWTP 遥感影像语义分割基线

本项目基于 MMSegmentation 1.2.2，为污水处理厂（WWTP）二分类遥感语义分割提供统一、可复现、便于扩展的实验骨架。已接入以下模型：

- U-Net（S5-D16）
- DeepLabV3+（ResNet-50）
- HRNet（W18）
- SegFormer（MiT-B2）
- SegNeXt（MSCAN-S + LightHamHead）
- Mask2Former（Swin-T）
- UNetFormer（ResNet-18）
- RS-Mamba（Tiny，八方向 selective scan）
- RPGV-Net（MiT-B2 + 可靠性感知伪几何校正与双频验证）
- CBR-Net、HD-Net（独立边界模型对比配置，见 [使用说明](docs/edge_baselines.md)）

数据、模型、训练策略、评价指标和命令入口彼此解耦。新增模型或消融实验通常只需增加一个注册模块和一份继承配置，不需要修改训练器。

## 目录结构

```text
configs/
├── _base_/
│   ├── datasets/             # WWTP 数据与增强
│   ├── models/               # 单一职责的模型定义
│   ├── schedules/            # 优化器与训练周期
│   └── default_runtime.py    # 日志、checkpoint、随机种子
└── experiments/              # 8 个可直接运行的实验
wwtpseg/
├── datasets/                 # WWTPDataset
├── evaluation/               # 前景、边界、Hausdorff 指标
└── models/                   # 自定义 backbone 与 RPGV-Net 分割器
tools/                        # train / test / smoke_test / 数据校验
docker/                       # 固定 CUDA/PyTorch/OpenMMLab 环境
```

## 环境

推荐镜像基于 `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`，固定核心版本如下：

| 组件 | 版本 |
| --- | --- |
| PyTorch | 2.1.2 + CUDA 12.1 |
| MMSegmentation | 1.2.2 |
| MMCV | 2.1.0 |
| MMEngine | 0.10.7 |
| MMDetection | 3.3.0（Mask2Former 组件） |
| timm | 0.9.16 |
| mamba-ssm | 1.2.2 |

使用 `devel` 而非 `runtime` 镜像是因为 RS-Mamba 的 fused selective-scan 需要 CUDA 编译工具链。宿主机只需要合适版本的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。

```bash
docker compose build
```

镜像构建时会编译 Mamba CUDA 扩展，首次构建时间较长。若 GPU 架构不在默认的 `7.0;7.5;8.0;8.6;8.9+PTX` 中，可这样覆盖：

```bash
docker compose build --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
```

## 数据校验与冒烟测试

仓库默认数据路径是 `wwtp_semantic_dataset/`，也可通过环境变量 `WWTP_DATA_ROOT` 指向其他位置。掩膜是调色板 PNG；加载时必须保留调色板索引 0/1，项目使用 MMSeg 默认的 Pillow annotation backend 处理这一点。

先校验全部 3044 对文件、尺寸和标签值：

```bash
docker compose run --rm wwtp \
  python tools/validate_dataset.py /workspace/wwtp_semantic_dataset
```

再运行所有模型的极小合成输入 loss 前向、真实训练/验证样本读取、位置增强和指标不变量测试：

```bash
docker compose run --rm wwtp python tools/smoke_test.py
```

需要额外验证反向传播时：

```bash
docker compose run --rm wwtp \
  python tools/smoke_test.py --backward
```

冒烟测试会临时缩小 RS-Mamba 的宽度和深度，并禁用预训练权重下载；正式训练配置不会被修改。无 CUDA 时 RS-Mamba 会使用很慢但可微的 PyTorch 参考扫描，它仅适合 32/64 像素测试，完整训练必须使用 fused CUDA selective-scan。

RPGV 的 CUDA FP16 数值回归（含三阶段 AdamW 更新、8 步梯度累积、饱和熵与
1024 分辨率损失反向测试）：

```bash
docker compose run --rm wwtp python tools/test_rpgv_numerics.py -v
```

三阶段更新测试使用 64 像素合成输入，不下载预训练权重；无 CUDA 时会明确跳过
GPU 用例。训练现在会在非有限前向 loss 出现时立即报出阶段与损失项，避免继续
更新并保存无效权重。如果旧 checkpoint 已含 NaN/Inf，或 AMP `loss_scaler.scale`
已为 0，不能用 `--resume` 恢复它。应选用已验证正常的断点，或从第一阶段在新
工作目录重新训练（保留旧目录用于排查）：

```bash
docker compose run --rm -e RPGV_WORK_ROOT=work_dirs/rpgv_staged_amp_fixed wwtp \
  bash scripts/train_rpgv_stages.sh
```

## 训练

### RPGV-Net 伪几何预处理

RPGV-Net 使用离线冻结的 Depth Anything V2，不会在分割训练时更新或重复运行深度模型。先为 train/val/test 生成整图归一化的伪深度和增强一致性可靠度：

```bash
docker compose run --rm wwtp \
  python tools/generate_pseudo_geometry.py \
  /workspace/wwtp_semantic_dataset \
  --device cuda
```

默认对 2048 图像使用 1024、25% 重叠的局部块，与 518 全图预测做 scale-shift 对齐后以 Hann 权重拼接；随后用恒等、水平/垂直翻转及尺度变换的预测方差生成可靠度。结果以 16-bit 深度和 8-bit 可靠度压缩写入 `wwtp_semantic_dataset/pseudo_geometry/<split>/<stem>.npz`。可用 `WWTP_PSEUDO_ROOT` 指定其他目录。

生成后按 RGB 预训练、几何预训练和联合微调三个阶段训练：

```bash
docker compose run --rm wwtp \
  bash scripts/train_rpgv_stages.sh
```

三个阶段分别训练 40k、20k 和 40k iterations，脚本会验证并自动传递阶段间 checkpoint。可通过 `RPGV_WORK_ROOT` 修改工作目录。`configs/experiments/rpgv_net.py` 保留为不经过分阶段预训练的直接联合训练消融配置。

RPGV-Net 的训练 crop 为 1024，batch size 1 并累积 8 步；推理采用 crop 1024、stride 768 的 Hann 加权滑窗。数据管线在局部增强前保留 512 全图 thumbnail，用共享 MiT-B2 生成 FiLM token；颜色增强只作用于 RGB，后续 resize/rotate/crop/flip 对 RGB、深度和可靠度同步执行。几何分支使用自顶向下 FPN，使四级几何编码器都能从辅助分割得到监督；高频边界残差在 1/4 融合，深层区域残差在 1/16 融合，解码后再以 1/2 RGB 细节分支精修边界。几何训练阶段以 30% 概率注入噪声、模糊、块缺失、scale-shift 或全零深度，并生成不作为模型输入的软有效度监督；联合阶段另以 15% 概率将整幅几何可靠度置零，用于维持 RGB-only 退化能力。

单卡训练一个模型：

```bash
docker compose run --rm wwtp \
  python tools/train.py configs/experiments/segformer.py \
  --work-dir work_dirs/segformer
```

例如训练新增的 SegNeXt-MSCAN-S：

```bash
docker compose run --rm wwtp \
  python tools/train.py configs/experiments/segnext.py \
  --work-dir work_dirs/segnext
```

多卡训练：

```bash
docker compose run --rm wwtp \
  torchrun --nproc-per-node=4 tools/train.py \
  configs/experiments/deeplabv3plus.py \
  --launcher pytorch --work-dir work_dirs/deeplabv3plus
```

依次训练全部不依赖伪几何预处理的基线模型：

```bash
docker compose run --rm -e GPU_COUNT=1 wwtp bash scripts/train_all.sh
```

断点恢复：

```bash
python tools/train.py configs/experiments/unet.py \
  --work-dir work_dirs/unet --resume
```

所有参数都可从命令行覆盖，适合快速资源适配：

```bash
python tools/train.py configs/experiments/hrnet.py \
  --cfg-options train_dataloader.batch_size=2 \
                train_cfg.max_iters=20000 \
                train_cfg.val_interval=1000
```

默认训练增强依次包含 0.5–1.5 倍随机缩放、随机任意角度旋转、前景感知位置裁剪、水平/垂直/对角翻转和光度扰动。针对目标集中在原图中央的位置偏置，正样本有 80% 概率把前景质心放到 512×512 裁剪窗口内 15%–85% 的随机位置；其余 20% 正样本和全部负样本仍使用均匀随机裁剪，以保留纯背景与困难上下文。该裁剪在前景非常小时也会尽量保留目标，避免增强后正样本大量退化为负样本。

默认训练 40k iterations，每 2k iterations 验证并按 `binary/Foreground_IoU` 保存最佳权重。验证和测试不使用随机增强，在原始 2048×2048 影像上进行 512×512、stride 384 的滑窗推理，避免直接缩小影像导致小目标和边缘评价失真。RS-Mamba 默认 batch size 1、梯度累积 4 次，以维持有效 batch size 4。

预训练权重会在首次正式训练时自动下载。若运行环境完全离线，请事先缓存权重，或者用配置覆盖相应 `init_cfg=None`；UNetFormer 的 timm encoder 可覆盖为 `model.backbone.pretrained=False`。

## 测试

```bash
docker compose run --rm wwtp \
  python tools/test.py configs/experiments/segformer.py \
  work_dirs/segformer/best_binary_Foreground_IoU_iter_*.pth
```

结果同时写入终端和对应 `work_dirs/<model>/` 日志。

WWTP 全模型 val/test 黑白掩码和逐图对比图的生成方式见 [推理对比图说明](docs/wwtp_inference_gallery.md)。

## 指标定义

MMSeg 原生 `IoUMetric` 输出 `aAcc`、`mIoU`、`mAcc`、`mDice`、`mFscore` 等常规指标；自定义 `BinaryBoundaryMetric` 额外输出：

| 指标 | 含义 |
| --- | --- |
| `Foreground_IoU` | 仅类别 1（污水厂）的全局 IoU |
| `Dice` / `F1` | 前景像素 Dice/F1，二者数值相同 |
| `Precision` / `Recall` | 前景像素查准率/查全率 |
| `Boundary_F1` / `BFScore` | 预测与真值轮廓在 3 px（1.5 m）容差内的全局边界 F1 |
| `Boundary_Precision/Recall` | 边界匹配精度与召回率 |
| `Hausdorff_px/m` | 每张图对称 Hausdorff distance 的均值 |
| `HD95_px/m` | 每张图 95% Hausdorff distance 的均值，降低单个离群点影响 |

重叠与 F1 类指标以百分数输出，距离指标保留像素或米。两张空掩膜的 HD 为 0；仅一张为空时，以图像对角线作为有限最差惩罚。全空负样本不人为抬高 Boundary F1，但错误预测出的边界会降低 Boundary Precision。

## 新模型与消融实验

新增 MMSeg 已有模型，只需在 `configs/_base_/models/` 写模型配置，并在 `configs/experiments/` 组合四类 base config。新增自定义模块时：

1. 在 `wwtpseg/models/` 中实现并用 `@MODELS.register_module()` 注册；
2. 在 `wwtpseg/models/__init__.py` 导出；
3. 在模型 base config 中用注册名引用；
4. 运行 `tools/smoke_test.py --models <实验名>`。

消融配置建议继承目标实验，仅覆盖一个变量。例如对 UNetFormer 移除辅助头：

```python
_base_ = ['./unetformer.py']
model = dict(auxiliary_head=None)
```

边缘改进模型可把边界分支、边界损失和结构模块分别注册成独立组件，再用配置开关组合；现有指标无需修改即可横向比较定位与轮廓完整性。

数据增强同样采用注册组件与配置解耦。位置偏置消融可继承实验配置，把训练 pipeline 中 `RandomForegroundCrop` 的 `foreground_prob` 改为 `0.0`；其他参数不变即可与普通随机裁剪公平比较。

RPGV-Net 的结构消融集中在 `configs/ablations/`，通过 `component_cfg` 旁路组件且
保持 checkpoint 参数结构不变。正式对比使用同一三阶段协议完整重训：

```bash
bash scripts/train_rpgv_ablations.sh
python tools/summarize_rpgv_ablations.py work_dirs/rpgv_ablations
```

若要复用已经训练好的统一阶段二 checkpoint，仅重跑各变体的阶段三：

```bash
bash scripts/train_rpgv_ablations_from_stage2.sh
python tools/summarize_rpgv_ablations.py work_dirs/rpgv_ablations_stage3
```

开关语义、单因素控制方式、选择部分实验和快速阶段三诊断方法见
[`docs/model_design.md`](docs/model_design.md#十六消融实验)。

## 实现来源

- MMSeg 原生模型和配置遵循 [OpenMMLab MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 1.2.2 的组件接口。
- SegNeXt 使用 MMSeg 原生 MSCAN-S 骨干和 LightHamHead，并加载官方 MSCAN-S ImageNet 预训练权重。
- UNetFormer 模块依据论文及 [GeoSeg 官方实现](https://github.com/WangLibo1995/GeoSeg) 的 global-local attention、weighted fusion 和 feature refinement 结构接入 MMSeg。
- RS-Mamba 模块依据论文及 [官方 RS-Mamba 实现](https://github.com/NJU-LHRS/Official_Remote_Sensing_Mamba) 的八方向扫描、RSM block 与 U 形解码结构接入 MMSeg；CUDA 扫描由 `mamba-ssm` 提供。

模型对比时请保留同一数据划分、裁剪尺度、训练 iterations 和评价脚本，并同时记录参数量、显存、吞吐与是否加载预训练权重，避免只比较最终精度。
