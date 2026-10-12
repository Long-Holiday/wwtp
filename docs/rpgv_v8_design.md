# RPGV v8：RGB-only 端到端高斯几何分割

## 1. 定位与版本边界

v8 从单张正射 RGB 预测可学习的高斯基元，并通过可微正射溅射、节点关系和解码器反馈共同训练。**训练、验证、推理都只有三个图像通道，不读取 Depth Anything、伪深度或可靠度缓存。** 默认仍有 ImageNet 骨干预训练，这是 RGB 先验，不是深度教师。

这不是经典多视角 3D Gaussian Splatting 重建：没有多视角图像、相机位姿、真实高度、逐场景优化、深度排序或 alpha-over 遮挡合成。高度是由分割任务学习的相对潜变量，不能解释为米制高度；覆盖度不是经标定的几何可靠度，高度混合方差也不是预测误差。

- `RPGVNetV8`：WWTP，内部单 logit BCE/Dice，预测输出两类。
- `RPGVRemoteSensingV8`：LoveDA 7 类、UAVid 8 类、DeepGlobe 2 类 categorical CE。
- 从 `BaseSegmentor` 独立实现，不实例化 v7 的深度校正、频率拆分和跨表示对比头。
- v1–v7 实现与旧实验套件不替换；**v7 checkpoint 不能作为 v8 resume**。只能恢复相同模型配置、参数集合和训练协议的 v8 checkpoint。

## 2. 前向结构

```text
三通道图像 → MiT-B2/SDPA 四尺度特征
                ├─ RGB 空间解码路径 ───────────────────┐
                └─ 高斯参数预测 → 可微空间支持 A       │
                                  ├─ 局部特征溅射 → F8/F4
                                  └─ 节点关系推理 → 结构溅射 → F16/F8
                                                        ↓
                                               残差轮廓细化 → 分割
```

数据预处理保留原始 BGR 0–255，模型内部一次性 BGR→RGB/ImageNet 归一化。三通道以外的输入直接拒绝，不能静默忽略旧伪几何通道。

### 2.1 高斯预测

四尺度 RGB 特征经过投影、尺寸对齐及卷积融合后，在**多个分辨率的规则网格**上自适应池化。默认 `grid_sizes=(4, 8, 16)`，逐级 clamp 到结构栅格后拼成一个节点集，得到最多 16+64+256 = **336** 个锚点（改造前为单一 `grid_size=8` 的 64 个）；很小的输入会相应地按级缩小网格。默认节点维度128。

多尺度的作用是容量分配：窄渠、管道等细窄目标由细级基元承担，大面积水域由粗级基元承担，而不是让所有目标平摊同一份基元预算。这也让 `single_scale_dense` 消融有意义——它用单一 20×20 网格给出 **400** 个锚点，比默认的 336 还多，因此默认模型胜出时不能只用"节点更多"解释（§5）。

每个节点预测中心 `mu`、Cholesky 因子 `L`、幅值 `alpha`、相对高度 `z` 与内容特征 `f`。坐标使用像素中心 `[-1,1]`，x/y 轴分别按图像宽/高归一化。`step` 是**逐节点**张量，每个锚点携带自己层级的格距 `(2/grid_w, 2/grid_h)`：

- `reach = step * exp(center_log_scale)`，`mu = grid_center + tanh(offset) * reach`。中心不再被限制在自己的网格单元内；可达域始终是以锚点为中心、半径 `reach` 的圆盘，不做图像边缘硬裁剪。`center_log_scale` 是初值为 0 的标量，`exp(0) == 1.0` 精确，因此**初值处中心与改造前逐位相同**，训练只能放宽可达范围。
- 因为 `step` 逐节点，同一个全局标量自然给出尺度自适应的搜索范围：粗基元的中心可达远、细基元可达近。
- `L00, L11 = (softplus(raw_diag) + 1e-4) * step`；`L10 = raw_offdiag * step_y`，`L01=0`。
- `Sigma = L L^T`，始终正定；非对角项允许学习方向与各向异性，不使用各向同性初始化时会有零梯度的独立旋转角。
- `alpha = softplus(raw_alpha) + 1e-4`，为正核幅值，不要求它是物理不透明度概率。
- `z = tanh(raw_height)`，不具有物理尺度保证。

`center_log_scale` 在所有变体下都存在同一个 `state_dict` 键；`fixed_gaussians` 与 `bounded_centers` 下它是 `requires_grad=False` 的 buffer 而非 Parameter，以免留下一个永远拿不到梯度的死参数。参数集合只增加这一个标量与边描述子的一列（默认 +129）。

### 2.2 正射高斯溅射

对结构栅格坐标 x：

```text
w_i(x) = alpha_i exp[-0.5 ||L_i^-1 (x-mu_i)||²]
A_i(x) = w_i(x) / sum_j w_j(x)
F(x)   = sum_i A_i(x) f_i
D(x)   = sum_i A_i(x) z_i
V(x)   = sum_i A_i(x) (z_i-D(x))²
C(x)   = 1-exp[-sum_i w_i(x)]
```

`w` 是峰值核，没有高斯概率密度的 determinant 前因子。按节点维 log-softmax 等价计算 A，使用二维三角求解避免逐像素矩阵求逆。高斯核、归一化及方差用 FP32；double 输入保留精度以便 gradcheck。实现对基元顺序不敏感。

覆盖度 C 与高度 D、方差 V **实际输入局部融合和高斯辅助头**：否则统一缩放幅值可能被归一化抵消，高度也可能成为未使用的输出。它们不作为可靠度门控去抑制高斯路径。

默认结构栅格长边最多64，保持宽高比。仅建立一次 `B×N×H_gW_g` 支持，再复用到节点池化及两次特征渲染；在 F4/F8/F16 对渲染后的特征图插值，不在1024原图生成64张高斯支持图。

### 2.3 节点与关系

节点特征由预测内容、按 A 空间归一化池化的 RGB 特征及高斯形状描述的投影相加后 LayerNorm 得到。**像素渲染沿节点归一化；节点池化另沿空间归一化**。

默认两层、四头稠密关系注意力。每对节点的9维描述为：相对中心2维、x/y尺度对数比2维、相对高度1维、协方差相关系数差1维、空间支持余弦重叠1维、中心平方距离1维、**软接触1维**。边 MLP 产生逐头注意力偏置，配合内容 q/k 与 value 完成消息传递。

软接触与 v7 的同一测量量同源（`rpgv_v7_modules.py:140-145`）：在结构栅格上取四邻接方向的软支持交叉量之和，除以两节点软面积的几何均值。这里的差别是**全程保留梯度**——v7 在 `torch.no_grad()` 下用冻结观测计算它，v8 的接触依赖于学到的中心与协方差，因此是受分割目标塑造的量，而不是外部给定的量。它是本版本唯一同时跨尺度、又可直接归因到几何参数的边描述子。

接触量恒在 `[0, 2]` 内，不存在数值爆炸：`A` 在每个像素沿节点维归一化故 `sum_i area_i` 恰为栅格像素数，而 `sum_k A_i(x) A_j(x+delta) <= min(area_i, area_j) <= sqrt(area_i * area_j)`，所以每个方向归一化后不超过 1。`1e-12` 的 clamp 与 v7 保持一致以方便公式对照，但它不会实际生效。

这些是可学习几何属性，不是 v7 的独立观测深度；中心、协方差、相对高度、重叠与接触均保留梯度。没有离散 top-k 邻接或 GT 构图。关系节点再用同一 A 溅射成结构图。关闭消息传递时不构造关系网络，因此接触也不会被计算。

### 2.4 解码与轮廓

复用 `SpatialGeometryDecoderV7` 的纯接口，不复用 v7 segmentor：
- 关系更新后的节点特征注入 F16/F8；
- 更新前节点特征与 D/C/V 注入 F8/F4；
- 融合是普通可学习残差，无零初始化关闭门、可靠度阈值或梯度 detach。

复用 `ResidualContourHeadV6` 的粗语义、RGB 细节残差与辅助 SDF。UAVid 的每个滑窗独立预测局部高斯，使用 Hann 窗融合语义 logits；不拼接不同 crop 的潜在高度作为全局高度。

## 3. 训练目标与梯度

默认：

```text
L = L_final + 0.3 L_coarse + 0.1 L_sdf + 0.05 L_boundary_band
    + 0.1 L_gaussian_dense + 0.05 L_gaussian_node
```

- 主/粗分割：WWTP masked BCE+Dice；公共集 categorical CE。
- SDF 与边界带：保留 ignore-aware 轮廓目标，排除未知标签邻域形成的伪边界。
- 高斯稠密辅助：关系溅射特征与 D/C/V 预测标签，不直接读取 RGB skip。将有效 one-hot 类别计数和有效比例分别 area pooling 后归一化，采用 soft-target BCE/CE。
- 节点辅助：按当前 A 与有效标注比例聚合类别分布，使用有效支持质量加权 soft-target BCE/CE。**完整目标及监督权重 detach，前向池化、溅射与关系路径不 detach。**

忽略标签255只用于监督掩码，不影响高斯前向结构。所有标签忽略时，每项损失为连接计算图的精确零。无有效标注的节点自然不计损失，不设置额外最小质量硬阈值。

不保留深度校正惩罚、RGB–深度对比、v7测量关系回归、组合排序；不增加高度真值回归、深度教师、冻结阶段或默认几何正则集合。

## 4. 配置与公平训练

四个主配置：
- `configs/rpgv/v8/rpgv_v8.py`
- `configs/loveda/rpgv_v8.py`
- `configs/uavid/rpgv_v8.py`
- `configs/deepglobe/rpgv_v8.py`

独立继承 v8 model、标准 RGB dataset、`baseline_20k.py` 和 runtime，不继承旧版本配置链。所有 split 及公共无标签导出管线都没有 `LoadPseudoGeometry`。

外部协议与现有正式对比相同：1024输入、batch2×累积4、AdamW lr6e-4、普通参数decay0.01/归一化参数0、FP32 wrapper、梯度裁剪10、1000 iter warmup、poly0.9、30000 microsteps/7500更新、seed42、相同增强和选权重策略。训练入口沿用项目的 TF32 设置，不将其称为严格禁用 TF32 的纯 FP32。辅助目标差异是模型设计，不是优化器差异。

| 数据集 | 选权重 | 本地评估 | 推理 |
|---|---|---|---|
| WWTP | binary/Foreground_IoU | test | whole |
| LoveDA | mIoU | 官方val | whole |
| UAVid | mIoU | 官方val | slide1024/stride768 |
| DeepGlobe | roadIoU | 本地有标签test | whole |

### 命令

```bash
# 无依赖 dry-run；不会出现 generate_pseudo_geometry 命令
python scripts/run_experiments.py train-test --dataset wwtp --suite v8-all --dry-run

# 正式训练由使用者按需启动，不会因添加模型而自动运行
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset wwtp --suite v8
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset wwtp --suite v8-ablations
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset loveda --suite v8
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset uavid --suite v8
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset deepglobe --suite v8

# 公共无标签预测也不需要深度缓存
docker compose run --rm wwtp python tools/evaluation/predict_public.py \
  configs/loveda/rpgv_v8.py work_dirs/loveda/rpgv_v8/BEST.pth \
  --output work_dirs/loveda/rpgv_v8/submission
```

公共数据集仍要求符合现有 manifest/来源/划分验证；v8 不是绕过数据协议的入口。

## 5. 核心消融

`v8-ablations` 仅适用于 WWTP：

| 配置 | 变化 |
|---|---|
| `rgb_only` | 不构造任何高斯预测、关系、融合与高斯辅助头，保留 RGB 解码与轮廓 |
| `fixed_gaussians` | 固定中心/协方差；幅值、高度、内容与关系仍学习，只检验空间支持形状学习 |
| `no_message_passing` | 不构造关系网络，原始节点直接溅射，保留两项高斯辅助；接触也不再计算 |
| `no_gaussian_aux` | 不构造 dense/node 辅助头，保留主/粗分割、SDF和边界带 |
| `bounded_centers` | `center_log_scale` 固定为 0，中心可达半径退回一个自身层级的格距；只检验放宽中心自由度本身 |
| `no_contact` | 只从边描述子去掉接触维（9→8），关系网络与其余目标不动；只检验该几何关系量 |
| `single_scale_dense` | 单一 20×20 网格 = **400** 个锚点，比默认多尺度的 336 还多；检验增益来自尺度多样性而非节点容量 |

后三项是与本次机制一一对应的单因素对照。`bounded_centers` 与 `no_contact` 只改一个标量/一列权重，参数量与完整模型几乎相同；`single_scale_dense` 的节点数严格更多，因此它对完整模型是保守方向的对照。`fixed_gaussians` 的语义随本次改动自动更新为"每个锚点的协方差 = 0.8 × 自身层级格距"，不要改回共享格距。

关闭模块时不留下未使用的训练参数（`fixed_gaussians`/`bounded_centers` 下 `center_log_scale` 是 buffer）。`v8-all` = 七基线 + 完整 v8 + 七消融，共15组；不改变 `all`/`v7-all`，也不沿用 v7 superiority gate。`fixed_gaussians` 与 `rgb_only` 分别移除了形状头与整个高斯路径，参数集合不同，不能与完整模型互相 resume 优化器；`bounded_centers`、`no_contact`、`no_gaussian_aux`、`no_message_passing` 的 `state_dict` 键集合与完整模型一致。

## 6. Docker 验证与限制

使用已有 `wwtp-mmseg:torch2.1.2-cu121`；源码只读绑定覆盖镜像旧快照，测试数据/权重写入临时容器目录，不安装宿主机依赖：

```bash
docker run --rm --mount type=bind,src="$PWD",dst=/workspace,readonly \
  -w /workspace -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  wwtp-mmseg:torch2.1.2-cu121 \
  python -m unittest tests.rpgv.test_rpgv_v8 tests.test_run_experiments \
  tests.test_experiment_protocol tests.test_deepglobe_training -v

docker run --rm --mount type=bind,src="$PWD",dst=/workspace,readonly \
  -w /workspace -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  wwtp-mmseg:torch2.1.2-cu121 \
  python tests/smoke_runner.py --config configs/rpgv/v8/rpgv_v8.py --device cpu
```

GPU Docker 映射参照 [已有 WSL 说明](docker_validation.md)，无需新 CUDA 扩展。可用 `tests/smoke_test.py --dataset <dataset> --models rpgv_v8 --backward --device cuda` 检查各数据集路径。

验收包括解析高斯参考计算、double gradcheck、基元置换、退化输入、主损失对所有高斯参数族的非零有限梯度、参数干预改变 logits、ignore/all-ignore、二分类与2/7/8类、消融无死参数、滑窗与checkpoint恢复；本次另加多尺度锚点在各级 clamp 下的节点数、中心可达域随层级与 `center_log_scale` 的缩放、接触与独立移位参考实现一致且保留梯度、边偏置输入维随接触开关变化。

资源诊断（无真实数据、无预训练下载）：

```bash
python tools/rpgv/profile_rpgv_v8.py --device cpu --size 64 --batch-size 1 --steps 1
python tools/rpgv/profile_rpgv_v8.py --size 1024 --batch-size 2 --stages \
  --output work_dirs/profiles/v8_train.json
python tools/rpgv/profile_rpgv_v8.py --mode infer --size 1024 --batch-size 1 \
  --output work_dirs/profiles/v8_infer.json
# (网格规模, 接触开关) 的成本矩阵，含每格节点数
python tools/rpgv/profile_rpgv_v8.py --mode matrix --size 1024 --batch-size 2 \
  --output work_dirs/profiles/v8_matrix.json
```

该工具**不能**复用 `profile_rpgv_v6.synthetic_batch`：后者产出 5 通道图像，会被 `_normalize_bgr` 拒绝。它自带 3 通道合成 batch，并报告逐参数梯度 L2（因此 `center_log_scale` 的梯度可见）、各阶段首次完成时间与峰值显存、以及推理模式峰值。

**这些验证只证明实现可运行、可学习，不证明有用三维恢复或分割精度提升。** 64×64/batch2 smoke 不能代表1024正式训练显存或吞吐；64×64结构栅格也不能保证独立表达所有窄小目标，细节仍依赖RGB路径。精度收益需要真实数据完整训练，并与 RGB-only/固定空间高斯对照。多尺度把默认节点数从 64 提到 336，关系注意力是稠密 N×N，因此显存与吞吐的变化必须由 profiler 在 1024/batch2 上实测，不能由 64×64 smoke 推断；`single_scale_dense` 的 400 节点比默认更重，是刻意为对照留出的余量。

### 本次架构升级验证（2026-10-12）

环境：`wwtp-mmseg:torch2.1.2-cu121`，源码只读绑定到 `/workspace`，宿主机 Python 与依赖不动。

- `tests.rpgv.test_rpgv_v8 tests.test_run_experiments tests.test_experiment_protocol tests.test_deepglobe_training` 共 **70 项通过，1 项跳过**（CUDA 用例，`CUDA_VISIBLE_DEVICES=` 为空），约 45 秒。
- `tests/smoke_runner.py` 对 `rpgv_v8`、`no_contact`、`bounded_centers`、`single_scale_dense` 四个配置均通过生产训练→保存→测试→恢复；`smoke_test.py` 的 v8 路径通过。
- dry-run 核对：`v8` 1/1、`v8-ablations` 7/7、`v8-all` 15/15，三者均 0 次 `generate_pseudo_geometry`。
- CPU 合成 profiler（64×64、batch1）：参数 25,175,144，`center_log_scale` 初值梯度 L2 为 1.9e-3（非零），阶段顺序为 encoder → predictor → graph → structure_fusion[0..1] → local_fusion[1,0] → decoder → refiner。
- **与改造前的数值等价性**（同一 seed，`grid_sizes=(8,)` 且 `center_log_scale=0`，`git archive` 取改动前代码做对照）：31 组预测器参数张量与输入 `encoded`/`seeds` 全部逐位相同，产出的 `centers` 与 `cholesky` 逐位相同；其余张量差异为 `nn.Linear` 的 1-5 ULP（隐层 ≤8.9e-8、`nodes` ≤5.4e-7）。因此准确表述是**机器精度内等价**，不是全链路逐位相同；该差异来自两个构建方式不同的进程间的浮点派发效应（已验证与张量内存布局无关），不影响任何研究结论。
- 未启动真实数据训练，未在 1024/batch2 上测过显存与吞吐；这些验证不证明分割精度提升。
