# WWTP 遥感语义分割实验

基于 MMSegmentation 1.2.2，统一 WWTP、UAVid2020、LoveDA 和 DeepGlobe Road Extraction 的模型、训练与评估入口。

主流模型固定为 **HRNet-W18、UNetFormer-R18、SegNeXt-S、SegFormer-B2、DeepLabV3+-R50、Mask2Former-Swin-T、HDNet**。`untformer` 指 UNetFormer，`deeplab` 指 DeepLabV3+。训练批处理、测试批处理和合成冒烟测试使用同一份模型清单：`wwtpseg/experiments/__init__.py`。

RPGV v1、v2、v3、v4、v5、v5.1 的实现、阶段配置、消融、初始化与数值测试全部保留，并新增 v6 的可变形深度校正、频域几何和区域级双向对比学习。公开数据集另保留已有 RPGV RGB／完整几何／v2 多分类适配，并提供独立的 v6 多分类适配；v3–v5.1 原生二分类实现继续用于 WWTP，未把它们改名成多分类实验。

## 项目结构

```text
configs/
  _base_/             共享模型、1024 数据管线、运行与训练协议
  wwtp/               WWTP 七个主流模型
  uavid/              UAVid 同一组七个主流模型及已有 RPGV 多分类配置
  loveda/             LoveDA 同一组七个主流模型及已有 RPGV 多分类配置
  deepglobe/          道路二分类：七基线与完整 RPGV v6/v7
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

公共依赖版本在 `requirements/core.txt` 统一维护，包含与基础镜像一致的 PyTorch／torchvision 固定版本；项目依赖在 `requirements/runtime.txt`，其中 `tensorboard` 供 `TensorboardVisBackend` 写事件文件和提供查看曲线用的 `tensorboard` 命令。构建期间以公共依赖清单约束 MMCV，避免依赖解析意外升级 PyTorch 或 NumPy。镜像构建包含依赖一致性检查、CUDA 扩展导入检查与项目模块导入检查，不启动训练。

实际吞吐与 24GB 显卡在完整 1024 分辨率训练时的显存占用需在目标机器验证；下文提供无需数据集和预训练权重的小规模验证入口。

## 数据与训练协议

WWTP 默认根目录 `wwtp_semantic_dataset_1m/`，**1024×1024 整图、1 m/px**，沿用 v5 的光度扰动和精确翻转，不在线放大到 2048、不随机裁小块。误用非 1024 数据会明确报错。已有原始 WWTP 数据可用 `tools/data/resample_wwtp_dataset.py` 转换；影像面积采样、掩膜最近邻，划分保持不变。

公开集支持 **UAVid2020、LoveDA、DeepGlobe Road Extraction**，本项目不下载数据。准备工具读取已解压目录。LoveDA 严格 `2522/1669/1796` 张；UAVid2020 严格 `200/70/150` 帧，按官方 sequence 保留 split，检查每个 sequence 十帧、划分间无 sequence 重叠。DeepGlobe 严格使用官方有标签 train 的 6226 对原图，构建固定本地 train/val/test，详见下节。准备失败不会生成可用于训练的完整数据目录。详细目录与类别编码见 [数据协议](docs/datasets.md)。

```bash
docker compose run --rm wwtp python tools/data/prepare_public.py loveda \
  --source data/remote_sensing/raw/loveda --output data/remote_sensing/prepared/loveda
docker compose run --rm wwtp python tools/data/prepare_public.py uavid \
  --source data/remote_sensing/raw/uavid --output data/remote_sensing/prepared/uavid
```

LoveDA 保留官方 1024 整图；UAVid 仅在官方训练 sequence 内切成 1024 块，边缘补零、标签补 255，不丢失边缘像素。验证与官方测试保留完整原始帧，用 1024／stride 768 滑窗拼接。**200 是原始训练帧数，不是切片后的样本数。** 不随机拆帧、不合并 train/val。

正式对比（七基线、WWTP RPGV v6及其消融、公开集已有RPGV多分类适配）统一使用 `configs/_base_/schedules/baseline_20k.py`：**FP32 OptimWrapper、AdamW（lr=6e-4、betas=0.9/0.999、eps=1e-8、weight decay=0.01）、norm参数不衰减，其余参数同学习率/衰减规则，无骨干或头部专属倍率；L2梯度裁剪max_norm=10**。batch 2 × 累积 4 = 单卡有效batch 8，30,000个数据加载迭代 = 7,500次优化器更新；前1,000迭代LinearLR从1e-6倍升至基础LR，随后PolyLR（power=0.9、eta_min=0）。每1,000迭代验证与保存，不早停，seed=42、deterministic=False，自动学习率缩放关闭。FP32覆盖原AMP模型，吞吐与显存需求需重新测量。

同一数据集内保持相同划分、1024输入、RGB增强、采样及选权重规则：WWTP按验证前景IoU，UAVid/LoveDA按验证mIoU，DeepGlobe按本地验证道路IoU（`roadIoU`）。WWTP指标含前景、边界与拓扑，边界容差1.5px=1.5m；公开集使用mIoU/mDice/mFscore，DeepGlobe额外报告道路IoU/F1。**统一的是外部训练协议，模型原生辅助/匹配/几何损失及必要输入保留**。标准骨干使用各自ImageNet预训练；HDNet没有现成预训练骨干，保留随机初始化并单独披露，不能宣称预训练条件完全相同。

## 训练与测试

```bash
# 单模型：train/test 始终使用同一份配置
docker compose run --rm wwtp python tools/train.py configs/wwtp/hrnet.py
docker compose run --rm wwtp python tools/train.py configs/uavid/hrnet.py
docker compose run --rm wwtp python tools/train.py configs/loveda/hrnet.py

# 一键训练并评估：每个模型训练成功后，自动评估验证集选出的最佳权重
# 七基线：HRNet、UNetFormer、SegNeXt、SegFormer、DeepLabV3+、Mask2Former、HDNet
python scripts/run_experiments.py train-test --dataset wwtp --suite baselines
# RPGV v6 完整模型
python scripts/run_experiments.py train-test --dataset wwtp --suite v6
# 五个 v6 消融：RGB-only、无对比学习、encoder fusion、legacy contour、cluster32
python scripts/run_experiments.py train-test --dataset wwtp --suite v6-ablations
# 全套 13 个实验（上述三个命令的合集，不要在同一输出目录重复运行）
python scripts/run_experiments.py train-test --dataset wwtp --suite all
# 只查看计划，无需安装训练依赖，也不会生成数据或启动训练
python scripts/run_experiments.py train-test --dataset wwtp --suite all --dry-run
# 断点恢复全套实验；仅恢复同一版本、同一配置、同一训练协议的 checkpoint
python scripts/run_experiments.py train-test --dataset wwtp --suite all --resume

# Docker 中将上述 python 命令前加 docker compose run --rm wwtp
# 公开集支持七基线和完整 v6/v7；消融、all/v7-all 组仅适用于 WWTP
# UAVid/LoveDA 自动评估官方 val，不报告无标签 test 的指标
python scripts/run_experiments.py train-test --dataset loveda --suite baselines
python scripts/run_experiments.py train-test --dataset uavid --suite v6

# 原有分开训练/评估入口继续支持：先查看所有七个模型的计划
docker compose run --rm wwtp python scripts/run_experiments.py train --dataset wwtp --dry-run
docker compose run --rm wwtp python scripts/run_experiments.py train --dataset loveda
docker compose run --rm wwtp python scripts/run_experiments.py test --dataset loveda

# 恢复原优化器、随机状态和调度器
docker compose run --rm wwtp python tools/train.py configs/wwtp/hrnet.py --resume

# 评估指定 checkpoint
docker compose run --rm wwtp python tools/test.py configs/wwtp/hrnet.py \
  work_dirs/wwtp/hrnet/iter_20000.pth --split val
```

一键入口复用现有配置与正式训练策略，不更改预算或超参数；串行执行，任何训练、选权重或评估失败都会停止，不会静默跳过。默认输出为 `work_dirs/<dataset>/<model>/`，评估指标保存到该目录的 `eval/metrics.json`；消融目录名为 `rpgv_v6_<消融名>`。用 `--work-root` 指定新的整组输出根目录；非空训练目录需显式 `--resume`。`--models` 可选择子集，与 `--suite` 互斥；不指定时仍默认七基线。`train-test` 中额外训练参数（如 `--no-pretrained`）只传给训练，显式 `--cfg-options` 同时传给训练和评估；WWTP与DeepGlobe默认评估有标签test，UAVid/LoveDA默认评估官方val；也可指定 `--split val`。一键 v6/消融会检查并补齐所需的伪几何，已有完整缓存则不加载深度模型；首次生成需要生成工具的依赖与 Depth Anything 权重。数据根目录用 `WWTP_DATA_ROOT`（兼容 `WWTP_V5_DATA_ROOT`）、`UAVID_DATA_ROOT`、`LOVEDA_DATA_ROOT`、`DEEPGLOBE_DATA_ROOT` 指定，伪几何目录用 `WWTP_PSEUDO_ROOT` / `PUBLIC_PSEUDO_ROOT` 指定；真实运行会按配置覆盖后的 pipeline 解析生成路径，dry-run 的几何路径仅展示环境变量默认值。即使 RGB-only 消融，其现有 pipeline 也要求伪几何。此批处理入口的 v6 目录与旧 shell 入口的 `work_dirs/rpgv_v6_siglip_1m` 独立，不会自动复用旧权重。

`--cfg-options` 仍可覆盖配置，但公平比较必须对整组实验应用相同策略，并使用相同GPU数；N卡默认全局有效batch为8N，不能与单卡结果混为等预算。变更batch时同步调整累积和迭代预算，保持全局有效batch、图像曝光与更新数。多卡批处理用 `--gpus N`。新协议必须重新训练并使用新的 `--work-dir`，旧策略的checkpoint/optimizer状态不能直接resume或混入正式结果。

训练入口统一启用TF32，数据加载使用persistent workers与pin memory；FP32 wrapper并不代表禁用TF32的严格FP32矩阵运算。删除训练循环与确定性设置的全局monkey patch，使用MMEngine原生断点续训语义。`--amp-dtype`仅适用于历史或显式AMP配置，正式FP32配置不接受该选项。测试/恢复从完整checkpoint加载时关闭骨干预训练下载。

UAVid/LoveDA 官方测试标签不公开，`tools/test.py` 的公开集配置明确评估官方 **val**。测试集预测用独立导出入口，不生成伪测试分数：

```bash
docker compose run --rm wwtp python tools/evaluation/predict_public.py \
  configs/loveda/segformer.py work_dirs/loveda/segformer/BEST.pth \
  --output work_dirs/loveda/segformer/submission
```

LoveDA 提交沿用官方预测脚本：导出 **0–6 ID**，以原图片编号直接保存到输出目录；原始真值的 1–7 编号只用于读取训练标签。UAVid 导出与官方颜色匹配的 RGB mask，目录为 `seq*/Labels/<原帧名>.png`。

### DeepGlobe Road Extraction

自行获取 [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)，解压至新的原始数据目录。必需 `train/<id>_sat.jpg` 与 `train/<id>_mask.png` 共 6226 对；官方无标签 `valid/`（1243张）和 `test/`（1101张）可选。**不能用无标签官方 valid/test 计算本地指标。**

默认将有标签 train 按原图 ID 固定 seed=42 拆成 **4982/622/622** 张本地 train/val/test，整图保持 1024×1024；标签黑色→背景0，白色→道路1，白色255不作为ignore。划分和来源记录在 `manifest.json`，训练入口校验成员互斥与完整覆盖；这是本地随机原图划分，不声称地理空间隔离，结果不可冒充官方排行榜成绩。`--split-seed`、`--split-ratios TRAIN VAL TEST` 可自定义划分，正式对比必须使用同一份 prepared 数据；修改后应准备到新目录。

```bash
docker compose run --rm wwtp python tools/data/prepare_public.py deepglobe \
  --source data/remote_sensing/raw/deepglobe --output data/remote_sensing/prepared/deepglobe

# 九模型：按本地 val 的道路IoU选best，再评估本地test
# 先用 --dry-run 检查命令；删除该参数才会真正启动训练
docker compose run --rm wwtp python scripts/run_experiments.py train-test \
  --dataset deepglobe --models hrnet unetformer segnext segformer deeplabv3plus mask2former hdnet rpgv_v6 rpgv_v7 \
  --dry-run
# 也可分组运行（不要与上述正式运行使用相同的非空输出目录）
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset deepglobe --suite baselines
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset deepglobe --suite v6
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset deepglobe --suite v7
```

新增配置在 `configs/deepglobe/`，七基线及RPGV v6/v7均继承同一30k外部训练策略。根目录由 `DEEPGLOBE_DATA_ROOT` 指定；Docker Compose 使用容器内 `/workspace/...` 路径。RPGV复用 `PUBLIC_PSEUDO_ROOT`（默认 prepared 目录下的 `pseudo_geometry`），批量入口自动准备 train/val/test 所需缓存。默认按 `roadIoU` 保存 `best_roadIoU_iter_*.pth`，指标为全集合累计像素统计的百分数道路IoU/F1及mIoU/mDice/mFscore；道路指标零分母定义为0，不对逐图IoU取平均。本地test不参与选权重，显式 `--split val` 可只评估验证集。

原始官方valid/test如已提供，准备工具分别导入 `img_dir/official_val` 和 `img_dir/official_test`，与本地test完全分离。导出黑白 `<id>_mask.png`（非本地评估）示例：

```bash
# RPGV导出前需为目标官方split生成伪几何；基线不需要此步
docker compose run --rm wwtp python tools/data/generate_pseudo_geometry.py \
  data/remote_sensing/prepared/deepglobe --image-dir img_dir --splits official_test

docker compose run --rm wwtp python tools/evaluation/predict_public.py \
  configs/deepglobe/rpgv_v7.py work_dirs/deepglobe/rpgv_v7/BEST.pth \
  --split official_test --output work_dirs/deepglobe/rpgv_v7/official_predictions
# --split official_val 可导出官方valid；不指定时默认official_test
```

新数据集不会自动插入 WWTP 的 `run_pipeline.sh` 训练队列。真实完整数据训练仍需由用户启动。

### 训练可视化（TensorBoard）

所有实验配置都继承 `configs/_base_/default_runtime.py`，其中 `vis_backends` 同时启用 `LocalVisBackend`（原有文本日志）和 `TensorboardVisBackend`；`tools/train.py`、`tools/test.py` 和 `scripts/run_experiments.py` 都无需额外参数。事件文件写入每次运行自己的目录：

```text
work_dirs/<dataset>/<model>/<时间戳>/vis_data/events.out.tfevents.*
```

两个 backend 共用运行目录下的同一个 `vis_data/`：`LocalVisBackend` 写 `scalars.json`、`vis_image/` 与配置快照，TensorBoard 的 `events.out.tfevents.*` 直接落在这一层，没有 `TensorboardVisBackend/` 之类的子目录（backend 的 `name` 只是 visualizer 内部的注册键，不参与路径拼接）。

时间戳来自 MMEngine 的运行目录：重新训练的 run 各自独立。`--resume` 也**不会**复用原时间戳目录，MMEngine 通过 `work_dir/last_checkpoint` 找到上次权重继续训练，但新的日志目录仍带新时间戳，所以恢复后的曲线会作为第二个 run 出现在 TensorBoard 里，与前半段互补。

训练会自动启动 TensorBoard 服务：`tools/train.py`（以及 `scripts/run_experiments.py`、`scripts/rpgv/*.sh` 里的每一次训练）在 rank 0 上拉起一个 detached 的 tensorboard 进程，logdir 默认取 work_dir 的根目录（即 `work_dirs`），URL 直接写进训练日志：

```text
TensorBoard: http://localhost:6006/ (logdir: /workspace/work_dirs)
```

服务在训练结束后继续运行，后续训练发现端口已被占用就直接复用，所以一次训练拉起的页面能连续看到整批实验的曲线。`--no-tensorboard` 关闭自动启动，`--tensorboard-port`（默认 6006）和 `--tensorboard-logdir` 覆盖端口与目录（例如只看 WWTP：`--tensorboard-logdir work_dirs/wwtp`）；`scripts/run_experiments.py` 在 train-test 模式下会把这些参数透传给 `tools/train.py`。端口被别的服务占用时只会在日志里提示跳过，不影响训练。

- **Docker**：自动启动的服务监听容器内的 `0.0.0.0:6006`，训练命令要写成 `docker compose run --rm -p 6006:6006 wwtp ...` 才能在宿主机打开。不想改训练命令时，也可以用独立容器查看曲线（它有自己的网络命名空间，与训练容器不冲突）：

```bash
docker compose run --rm -p 6006:6006 wwtp \
  tensorboard --logdir work_dirs --host 0.0.0.0 --port 6006
```

- **AutoDL / conda**：conda 环境不在镜像里，依赖要装进环境本身——`bash scripts/setup_autodl.sh` 会按 `requirements/runtime.txt` 装上 `tensorboard==2.14.1`，已有环境补一条 `pip install tensorboard==2.14.1`（缺它训练会在启动时直接报 `ImportError`）。面板自带的 tensorboard 默认占用 6006 且只看 `/root/tf-logs`，先按 AutoDL 官方做法停掉它（`ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9`）再启动训练，自动拉起的服务就接管 6006 并 watch `work_dirs`，面板链接会直接显示这批曲线；不想动面板进程时用 `--tensorboard-port 6007` 另起一个。

远程服务器上先用 `ssh -L 6006:localhost:6006 <host>` 建立本地转发，再访问同一端口。曲线内容：

- **Scalars**：训练 `loss`、`lr`、`data_time`、`time`，每 50 迭代记录，数值与文本日志一致；验证指标每 `val_interval`（正式协议为 1000 迭代）记录一次。WWTP 为前景 IoU、边界与拓扑指标，UAVid/LoveDA 为 mIoU/mDice/mFscore，DeepGlobe 额外记录 roadIoU/roadF1，并按 roadIoU 选权重，与 `save_best` 选择依据一致，因此最优 checkpoint 就是曲线峰值处。
- **Images**：默认关闭。mmseg 1.x 的 `SegVisualizationHook` 默认 `draw=False`，只打一条 warning 就返回，验证期不绘图、也不生成 `vis_image/`。需要时显式打开：`tools/train.py <config> --cfg-options default_hooks.visualization.draw=True`（`scripts/run_experiments.py` 同样透传 `--cfg-options`），打开后每 `interval`（默认 50）个验证样本画一张叠加了预测与真值的 1024 原分辨率图，同时写入 TensorBoard 与 `vis_data/vis_image/`。
- **Text**：本次运行的完整配置，便于核对训练协议。

`--logdir work_dirs` 按目录递归分组，run 名就是相对路径（含末尾的 `vis_data`），可直接叠加多个模型或消融曲线；`tools/test.py` 的评估目录（`<work_dir>/eval/<时间戳>/vis_data`）同样会写事件文件，但只含配置文本、没有曲线。TensorBoard 只读取已记录的数值，不参与训练，不改变优化器、调度器、数据增强或评测协议。首次使用要先装上 tensorboard——镜像重新 `docker compose build`，conda 环境 `pip install tensorboard==2.14.1`——否则 `TensorboardVisBackend` 会在运行时初始化失败，训练和测试启动时直接抛 `ImportError`，不会退化成只写本地日志。

## RPGV

RPGV的结构开关、损失和初始化规则继续保留。v1/v2阶段配置使用相同1024 WWTP根目录与整图增强；v3锚点初始化和v5.1权重迁移仍需原初始化工具。**v1–v5.1保留历史阶段/迁移协议，不纳入当前七基线+v6的统一策略正式对比**；`configs/rpgv/v5/controls/` 仅是历史等预算对照，不能把“等预算”当成“相同训练策略”。v6及其五个消融直接复用公共正式协议，不需要额外阶段训练。

```bash
docker compose run --rm wwtp python tools/data/generate_pseudo_geometry.py wwtp_semantic_dataset_1m --device cuda
docker compose run --rm wwtp bash scripts/rpgv/train_rpgv_v5.sh
docker compose run --rm wwtp python tests/smoke_test.py --include-rpgv --backward --device cuda
```

伪几何文件为 `pseudo_geometry/<split>/<image_stem>.npz`，与准备后的 1024 图像逐像素对齐。所有 RPGV 模型仍在 `wwtpseg/models/segmentors/`，消融和数值测试入口仍可直接调用。

### RPGV v6

优化版 v6 使用 SDPA MiT-B2，在 decoder F8/F4 注入几何残差；高频局部/区域细节不再被对比有效性阈值关闭，低频进行受限 token 交叉注意力。可靠度归一化 Gaussian 分解减少 Q 孔洞伪边缘，独立的 SDF 辅助头和无固定幅度限制的 logit 残差头负责轮廓细化。默认不关闭几何样本、不冻结骨干；无深度或 Q=0 时精确回退到 RGB 路径，不需要 CLIP/SigLIP 权重或文本编码器；RGB/几何区域对齐使用带可学习尺度与偏置的 SigLIP-style sigmoid 损失，单个有效区域也参与学习。

```bash
docker compose run --rm wwtp bash scripts/rpgv/train_rpgv_v6.sh
docker compose run --rm wwtp python -m unittest tests.rpgv.test_rpgv_v6 \
  tests.rpgv.test_mit_sdpa tests.rpgv.test_contour_targets -v
docker compose run --rm wwtp python tests/smoke_runner.py \
  --config configs/rpgv/v6/rpgv_v6.py --device cpu
docker compose run --rm wwtp python tools/rpgv/profile_rpgv_v6.py \
  --output work_dirs/profiles/v6_train.json
```

默认从 ImageNet RGB 骨干初始化联合训练，仍为1024整图、batch2×累积4、7,500次优化器更新；启用激活重计算，SDF目标在训练数据worker生成。新目录为 `work_dirs/rpgv_v6_siglip_1m`。**旧 InfoNCE v6、历史 v6/v5.1 完整权重不能直接 resume 本版 SigLIP 目标**，新增 `logit_bias` 参数且优化器参数集合改变，只恢复本版相同配置的checkpoint。除 `no_contrastive.py` / `rgb_only.py` 外，还提供 `encoder_fusion.py`、`legacy_contour.py`、`cluster32.py` 单因素消融。训练方式、WSL GPU启动替代命令、性能实测及已知限制见 [v6 设计说明](docs/rpgv_v6_design.md)。

### RPGV v7

v7 沿用统一外部训练策略，添加独立的“几何基元—空间关系—局部组合”结构路径：固定 64 个几何软节点（不平坦节点被筛掉），两层 128 维、4 头关系注意力，三层对比监督（区域、关系、组合）与结构扰动排序损失，结构图残差在 F16/F8 反馈到分割解码；F4 仍由 v6 几何与轮廓残差负责细节。GT 不参与构图或扰动；v6 checkpoint 不能 resume 完整 v7。保留辅助 SDF、边界带、校正正则；新损失按有效节点/边/组合归一化，并在日志记录有效样本数、负例数与组合分差，不进入总损失。

v7 消融仅研究完整 v7 的模块与训练目标（仅 WWTP），不混入版本演进对比。默认 `v7-ablations` 精简为六项，加完整 v7 共七组：

- 整体/前向机制：`rgb_only.py`、`fixed_grid.py`、`no_feedback.py`。
- 核心监督归因：`no_relation_supervision.py`、`no_composition.py`、`no_structure_aux.py`。分别关闭关系目标组（含测量回归）、组合排序、结构辅助分割；其他目标保持不变。

`no_hierarchical_supervision.py`、`no_node_supervision.py`、`no_message_passing.py` 保留为按需分析，不进入默认队列。前两项分别检查整体层次目标和区域监督；图推理项 **只与 `no_composition.py` 配对比较**，在两者共同关闭组合目标后，将关系注意力替换为节点独立 MLP，区域、关系及结构辅助监督均保留。无序 MLP 池化无法区分当前的内容置换，不能把此项与完整 v7 的差值单独归因于图消息传递，也不能称为仅去掉边属性。默认六项不单独证明区域目标或图消息传递的贡献。

统一外部协议不变：30,000 microsteps / 7,500 次优化器更新。`v7-all` 为原七基线、v6、完整 v7 和六个消融，共 15 项；其中 v6 不是消融对照。旧 `no_contrastive.py`、`no_relational_supervision.py`、`no_edges.py`、`no_perturbation.py`、`rgb_capacity.py` 保留原语义，但与上述三项按需分析一样只通过显式 `--models` 选择，不进入默认队列、不迁移旧结果。`rgb_capacity` 是 RGB 灰度/单位支持替代外部几何，并非严格纯参数容量控制。默认只有 `rgb_only` 不加载伪几何；历史 `rgb_capacity` 也不加载。公开集支持七基线和完整 v7。

```bash
python scripts/run_experiments.py train-test --dataset wwtp --suite v7-all --dry-run
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset wwtp --suite v7
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset wwtp --suite v7-ablations
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset loveda --suite v7
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset uavid --suite v7
docker compose run --rm wwtp bash scripts/rpgv/train_rpgv_v7.sh
docker compose run --rm wwtp python -m unittest tests.rpgv.test_rpgv_v7 \
  tests.rpgv.test_rpgv_v6 tests.test_run_experiments tests.test_experiment_protocol -v
docker compose run --rm wwtp python tests/smoke_runner.py --config configs/rpgv/v7/rpgv_v7.py --device cpu
docker compose run --rm wwtp python tools/rpgv/profile_rpgv_v7.py \
  --output work_dirs/profiles/v7_train.json
```

结构路径使用的相对伪深度由现有 `Depth Anything` 离线生成，本身含额外预训练先验；不把它当作真实三维信息。设计、损失公式、扰动判据、消融含义、合成诊断与已知限制见 [v7 设计说明](docs/rpgv_v7_design.md)。

## 小规模验证

无需数据集和预训练权重下载。DeepGlobe 数据、配置、指标、调度与RPGV前后向，以及合成数据短步训练/选best/test，可在现有镜像中用CPU测试（不需要GPU透传）：

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace \
  -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=2 \
  wwtp-mmseg:torch2.1.2-cu121 python -m unittest \
  tests.test_deepglobe_data tests.test_deepglobe_training \
  tests.test_run_experiments tests.test_experiment_protocol -v
docker run --rm -v "$PWD:/workspace" -w /workspace \
  -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 \
  wwtp-mmseg:torch2.1.2-cu121 python tests/smoke_test.py --dataset deepglobe --backward --device cpu
```

以下现有GPU测试命令仍可使用：

```bash
docker compose run --rm wwtp python -m unittest discover -s tests -p 'test_experiment_protocol.py' -v
docker compose run --rm wwtp python tests/smoke_test.py --dataset wwtp --backward --device cuda
docker compose run --rm wwtp python tests/smoke_test.py --dataset uavid --backward --device cuda
docker compose run --rm wwtp python tests/smoke_test.py --dataset loveda --backward --device cuda
docker compose run --rm wwtp python tests/smoke_runner.py --device cuda
```

模型测试使用 64 像素、batch 2，检查前向、反向、有限梯度和预测类别/尺寸；Runner 测试执行 2 步训练、保存、测试，再恢复至 4 步。它们验证代码路径，不代表完整训练质量或最终 1024 分辨率显存要求。

模型组件来源：[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2)、[UNetFormer/GeoSeg](https://github.com/WangLibo1995/GeoSeg)、[HDNet](https://github.com/danfenghong/ISPRS_HD-Net)。HDNet 的 GPL-3.0 许可证保留在 `wwtpseg/models/segmentors/hdnet/`。
