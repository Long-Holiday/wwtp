# RPGV v8.1：RGB-only 内容自适应高斯与显式关系消息

## 1. 定位与版本边界

v8.1 是 v8 的定向修正版，不改变版本定位：**训练、验证、推理都只有三通道 RGB**，仍不读取 Depth Anything、伪深度或可靠度缓存。相对 v8，它只做四件事：

1. 恢复 v7 在最终预测上的**区域一致性**与**有符号边界**监督；
2. WWTP 二分类恢复 v7 的**前景/背景平衡边界带**；
3. 高斯支持从“纯空间核”改为**高斯先验 + 内容相似度**，覆盖率仍只由高斯质量决定；
4. 关系层把边描述**重新接入消息内容**，并让 F8/F4 的局部融合读取**高分辨率 RGB**。

- `RPGVNetV81`：WWTP，内部单 logit BCE/Dice，预测输出两类。
- `RPGVRemoteSensingV81`：LoveDA 7 类、UAVid 8 类、DeepGlobe 2 类 categorical CE。
- v8 及 v1–v7 实现与实验套件不替换；**v8 checkpoint 不能作为 v8.1 resume**，只能恢复相同配置的 v8.1 checkpoint。
- 不是多视角 3DGS 重建：没有真实高度、相机位姿、逐场景优化、深度排序或遮挡合成。潜高度、覆盖度与方差仍是任务内相对量。

核心文件：

| 文件 | 职责 |
|---|---|
| `wwtpseg/models/utils/rpgv_v81_modules.py` | 内容感知溅射、含内容键的高斯预测器、带边值的关系层、高分辨率局部融合 |
| `wwtpseg/models/utils/rpgv_v81_losses.py` | 最终配对监督、平衡边界带、稠密 soft Dice |
| `wwtpseg/models/segmentors/rpgv_net_v81.py` | 完整二分类模型与损失装配 |
| `wwtpseg/models/segmentors/multiclass/model_v81.py` | 多分类适配（复用滑窗推理） |

## 2. 前向改动

```text
三通道图像 → MiT-B2/SDPA 四尺度特征
                ├─ RGB 空间解码路径 ─────────────────────────────┐
                └─ 高斯参数 + 内容键 → 支持 A = softmax(log w + s·cos(q,k)) 
                                  ├─ 局部特征溅射 → F8/F4（拼 C1/C2 → 局部融合）
                                  └─ 节点关系推理（边偏置 + 边值）→ 结构溅射 → F16/F8
                                                                  ↓
                                                        残差轮廓细化 → 分割
```

### 2.1 内容感知支持

支持图由高斯峰核与内容相似度共同决定，节点维 softmax：

```text
log w_i(x) = log alpha_i - 0.5 ||L_i^-1 (x-mu_i)||²
A_i(x)      = softmax_i [ log w_i(x) + s · cos(q(x), k_i) ]
```

- `q(x) = normalize(W_q · E(x))`、`k_i = normalize(W_k · seed_i)` 都用 1×1 线性映射，`s = content_scale` 默认 2.0。
- **覆盖率仍只由高斯质量决定**：`C(x) = 1 - exp(-Σ_j w_j(x))`，不含内容项。内容相似度不参与覆盖度，也不会被解释为可靠度。
- 高度、方差仍由 `A` 加权得到，因此内容项会改变高度聚合，但不改变几何参数本身的定义。
- `content_logits` 在结构栅格上计算（默认长边最多 64），复用于节点池化与两次渲染，不在原图分辨率展开。
- `content_aware=False` 时不构造两个内容映射，`state_dict` 键集合与 v8 完全一致。

### 2.2 关系消息

v8 的关系属性只进入注意力偏置。v8.1 恢复 v7 的消息形式：

```text
m_i = Σ_j a_ij V h_j  +  Σ_j a_ij W_e e_ij
a   = softmax_j( q_i·k_j/√d + bias(e_ij) )
```

实现先把**低维边描述**按同一注意力权重聚合，再投影到头维度（`Σ_j a_ij` 与投影可交换）：

```text
relation_i = Σ_j a_ij e_ij        # 边描述维（默认 9）
m_i        = Σ_j a_ij V h_j + W_e^(head) relation_i
```

因此不需要为每条边展开 128 维张量，显存与稠密 `N×N×C` 的边值实现不同，数学上逐位等价（`tests/rpgv/test_rpgv_v81.py::test_relation_aggregation_matches_explicit_edge_value_and_gradients` 同时校验输出与 `edge_value` 梯度）。`use_edge_value=False` 时该层不加参数、不构造边值分支。

### 2.3 局部细节融合

v8 的 `local` 与 `structure` 都来自同一张最多 64 的结构栅格。v8.1 的 F8/F4 局部融合直接读取骨干的 C2/C1 特征：

```text
f_local = _resize(W_rgb · C_k, size)     # 高分辨率 RGB 细节
f_gauss = _resize(W_gauss · local, size) # 高斯局部特征 + 高度/覆盖度/方差
f_out   = f + update([f, f_local, f_gauss])
```

`local_detail=False` 时回退为 v8 的 `GaussianFusionV8` 路径（`local_fusion` 的类型在两种模式下不同，参数集合也不同）。结构分支（关系更新后节点渲染 → F16/F8）保持 v8 语义，没有可靠度阈值或梯度 detach。

## 3. 训练目标

默认（WWTP 二分类）：

```text
L = L_final + 0.3 L_coarse + 0.1 L_sdf + 0.05 L_boundary_band_balanced
    + 0.05 L_region_consistency + 0.1 L_final_boundary
    + 0.1 L_gaussian_dense + 0.05 L_gaussian_node + 0.05 L_gaussian_dense_dice
```

| 项 | 来源 | 说明 |
|---|---|---|
| `L_region_consistency` 0.05 | 恢复自 v7/v6 | 同一 GT 类别内相邻像素预测概率差，不跨越 GT 边界 |
| `L_final_boundary` 0.1 | 恢复自 v7/v6 | 跨类别相邻像素的**有符号**跳变，与 GT 跳变同号，因此平滑抹掉窄目标不能降低成本 |
| `L_boundary_band_balanced` 0.05 | 对齐 v7 | 二分类下前景/背景 BCE 分别求均值再各半加权；多分类仍按出现类别平均，与 v8 相同 |
| `L_gaussian_dense_dice` 0.05 | 新增，仅二分类 | 稠密高斯 logits 对**软类别比例**目标的 soft Dice，与原有 `L_gaussian_dense`（soft BCE 0.1）相加 |

`L_final`、`L_coarse`、`L_sdf`、高斯 dense/node 监督与 v8 相同（主/粗仍为 0.5 BCE + 0.5 Dice）。WWTP 的配对损失使用 `region_and_boundary_loss`（sigmoid 概率），多类使用与 v6/v7 相同的 categorical 版本，二者在同一组输入上数值一致（测试内交叉校验）。忽略标签 255 不参与任何监督；全忽略样本每项损失为连接计算图的精确零。

## 4. 配置与公平训练

四个主配置：

- `configs/rpgv/v81/rpgv_v81.py`
- `configs/loveda/rpgv_v81.py`、`configs/uavid/rpgv_v81.py`、`configs/deepglobe/rpgv_v81.py`

模型基类 `configs/_base_/models/rpgv_net_v81.py` 继承 v8 基类并显式写出四项改动。**外部协议与 v8 完全相同**：batch8 × 累积1、bf16、AdamW lr6e-4、decay0.01/归一化参数0、梯度裁剪10、7500 次优化器更新、相同 warmup/PolyLR、增强与选权重口径。未新增 `protocol_optimizer_override` / `protocol_amp_override`，因为 v8.1 沿用 v8 已声明的同一档执行步幅。

**尚未独立实测**：v8.1 比 v8 多出结构栅格上的内容相似度张量（`B×N×H_g·W_g`）与 F8/F4 上一条三路融合通路。按 `tools/rpgv/profile_rpgv_v81.py` 在训练机上实测前，不应把 v8 的 batch8/bf16 当作 v8.1 已验证可承载的结论；若实测装不下，应按公平协议在配置中声明 `protocol_optimizer_override` / `protocol_amp_override` 并保持每更新图像预算与学习率轨迹不变。

### 命令

```bash
# 无依赖 dry-run
python scripts/run_experiments.py train-test --dataset wwtp --suite v81-all --dry-run

# 正式训练由使用者按需启动
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset wwtp --suite v81
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset wwtp --suite v81-ablations
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset loveda --suite v81
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset uavid --suite v81
docker compose run --rm wwtp python scripts/run_experiments.py train-test --dataset deepglobe --suite v81
```

公共集仍要求符合现有 manifest/来源/划分验证；`v81` 全系列只输入 RGB，批量入口不会为其生成伪几何。

## 5. 架构导向消融

默认队列围绕完整 v8.1 检验整体分支、支持机制、关系推理、尺度与辅助监督，而不是比较版本。`v81-ablations` 仅适用于 WWTP，包含以下七项；加完整模型共八组。

### 5.1 默认核心

| 配置 | 相对完整 v8.1 的变化 | 检验的问题与归因边界 |
|---|---|---|
| `rgb_only` | `use_gaussians=False` | 删除整个高斯/关系/融合分支和相应辅助头；保留 RGB 解码、轮廓及最终配对目标。整体分支对照，不是等参数容量的普通 RGB 基线 |
| `fixed_gaussians` | `gaussian_cfg.fixed_gaussians=True` | 固定中心与协方差，使用每个尺度各自的格距；幅值、潜高度、内容支持和关系仍学习。检验空间几何可塑性，不能独立归因到各向异性 |
| `no_content_support` | `gaussian_cfg.content_aware=False` | 退回纯高斯支持，同时不创建 query/key；检验内容适应的净作用，参数量有所改变 |
| `content_only_support` | `gaussian_cfg.spatial_support=False` | 支持改为 `A=softmax(content_logits)`；仅删除分配中的高斯空间核，其他几何路径保留 |
| `no_message_passing` | `use_message_passing=False` | 不构造关系网络，原始节点仍池化、回投、融合与监督；检验整个消息传播模块，不是仅删除边值 |
| `single_scale_dense` | `gaussian_cfg.grid_sizes=(20,)` | 单尺度 400 节点对照多尺度 336 节点；节点数更多但不是严格等容量/等计算对照 |
| `no_gaussian_aux` | dense、node、dense Dice 三个权重全部为 0 | 删除整个高斯语义辅助目标组和两类辅助 head；高斯分支仍由最终分割等目标联合训练 |

**三种支持的受控比较：**完整模型为高斯空间先验与内容相似度的联合支持；`no_content_support` 为纯高斯支持；`content_only_support` 为仅内容分配。`spatial_support` 默认 True，不新增参数或 buffer，完整模型默认计算与参数键保持不变。双关闭 `content_aware=False, spatial_support=False` 不构成有效对照，构造时明确报错；直接调用 splat 时没有内容 logits 又关闭空间支持也报错。

仅内容分配仍计算高斯质量覆盖度，节点 shape projection 和图边仍读取几何参数，种子仍来自空间网格。因此它**不是完全去高斯、无位置或无几何的网络**。在相同 primitives 下，三种支持的覆盖度定义不变；分别训练后 primitives 会变化，不能宣称最终覆盖度数值相同。高度、方差、节点池化/回投、支持重叠及接触都会使用修改后的 A，这些是支持变化的下游作用。

336/400 适用于结构栅格没有截断锚点网格的正式设置；小型冒烟会按栅格尺寸缩小节点数。单尺度对照可用于观察多尺度在较少节点下是否有效，但不能严格证明收益只来自尺度组织。实验表应披露参数量、实际节点数、实测显存和吞吐；400 节点的关系矩阵约为 336 节点的 1.42 倍大小，不代表实际运行时间也是该比例。当前队列也不单独证明各向异性、潜高度具有真实几何意义或高斯优于所有非高斯区域表示。

### 5.2 按需细粒度分析

以下原配置、任务名、输出目录和语义保持不变，只通过 `--models` 显式选择，不进入默认 `v81-ablations` / `v81-all`：

| 配置 | 变化 | 检验的问题 |
|---|---|---|
| `no_edge_value` | `graph_cfg.use_edge_value=False` | 关系属性进入消息内容的条件贡献；仍保留边偏置和注意力 |
| `no_local_detail` | `local_detail=False` | F8/F4 局部融合移除新增高分辨率 RGB 通路，保留原高斯局部融合 |
| `no_pairwise` | `region_loss_weight=0`、`final_boundary_loss_weight=0` | 最终配对监督组；仍保留主/粗分割、SDF 与边界带 |
| `no_dense_dice` | `gaussian_dense_dice_weight=0` | 稠密高斯 soft Dice 的条件贡献；dense BCE 和 node 监督仍在 |
| `unbalanced_band` | `balanced_boundary=False` | 前景/背景平衡的边界带目标，仅二分类不同 |

`v81-all` = 七基线 + 完整 v8.1 + 七项核心消融，共 **15 组**。其他版本套件、v7 superiority gate 与自动补齐重训逻辑不变；v8.1 全系列不生成伪几何。旧目录不搬迁、不自动覆盖；不同配置不能作为彼此的 optimizer resume，即使 state_dict 键相同。固定几何、移除分支/消息网络/内容映射/辅助头等配置会改变参数集合，需要报告参数量。

仅关闭 `content_aware`、`use_edge_value`、`local_detail` 三项可恢复旧前向架构，但 **全部损失与 v8 相同还需要**关闭恢复的 pairwise、dense Dice，并令 `balanced_boundary=False`；`spatial_support` 保持 True。已有 `test_all_new_switches_off_reproduce_v8` 同时检查这些条件、严格 state_dict 加载、前向和损失等价，这是实现回归，不是版本精度对比。

```bash
# 默认七项核心，完整模型另由 --suite v81 运行
python scripts/run_experiments.py train-test --dataset wwtp --suite v81-ablations --dry-run

# 原细粒度任务仍可显式选择
python scripts/run_experiments.py train-test --dataset wwtp --models \
  rpgv_v81_no_edge_value rpgv_v81_no_local_detail rpgv_v81_no_pairwise \
  rpgv_v81_no_dense_dice rpgv_v81_unbalanced_band --dry-run
```

沿用同一数据集每更新 8 张图、7500 次更新、相同学习率轨迹与选权重协议。后续训练机实测若需要缩小 micro-batch，应同步调整累积和所有步幅、按协议披露，不增加新的训练预算。本次任务调整不启动正式训练，不声称任何精度提升或显存承载结论。

## 6. Docker 验证与限制

沿用 `wwtp-mmseg:torch2.1.2-cu121`，源码只读绑定，宿主机环境不动：

```bash
docker run --rm --mount type=bind,src="$PWD",dst=/workspace,readonly \
  -w /workspace -e PYTHONDONTWRITEBYTECODE=1 -e CUDA_VISIBLE_DEVICES= \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  wwtp-mmseg:torch2.1.2-cu121 \
  python -m unittest tests.rpgv.test_rpgv_v81 tests.rpgv.test_rpgv_v8 \
    tests.test_rpgv_v81_integration tests.test_run_experiments tests.test_experiment_protocol -v

docker run --rm --mount type=bind,src="$PWD",dst=/workspace,readonly \
  -w /workspace -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  wwtp-mmseg:torch2.1.2-cu121 \
  python tests/smoke_runner.py --config configs/rpgv/v81/rpgv_v81.py --device cpu

docker run --rm --mount type=bind,src="$PWD",dst=/workspace,readonly \
  -w /workspace -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  wwtp-mmseg:torch2.1.2-cu121 \
  python tests/smoke_test.py --dataset wwtp --models rpgv_v81 --backward --device cpu
```

### 初版实现验证（2026-10-13，CPU）

- `tests.rpgv.test_rpgv_v81` **21 项执行、20 项通过、1 项跳过**（CUDA 用例）；覆盖内容支持与解析参考的逐位一致、零内容退化为 v8、置换等变、double gradcheck、预测器开关无残留参数、关系聚合与显式边值实现输出/梯度一致、节点置换、局部融合依赖高分辨率 RGB、平衡边界带与 v7 数值一致且与 v8 不同、配对损失有符号性、忽略标签零损失、soft Dice 解析值、全开关关闭退化为 v8、全忽略零损失、逐开关前后向、多类/滑窗、checkpoint 往返与三通道契约。
- `tests.test_rpgv_v81_integration tests.test_run_experiments` **34 项通过**（含套件规模、四数据集默认值、单因素差异、无伪几何生成、无 v7 gate）。
- 生产入口冒烟：`smoke_runner` 对 `rpgv_v81` 与 `rpgv_v8` 均完成 训练→保存→测试→恢复；`smoke_test` 对 `rpgv_v81` 与初版六个消融配置、以及 LoveDA/UAVid/DeepGlobe 三套公共配置完成前向与反向。
- 四数据集 `--suite v81` 与 `v81-all`/`v81-ablations` dry-run 通过，**0 次 `generate_pseudo_geometry`**。
- CPU 合成 profiler（64×64、batch1）完整模型 **25,239,400** 参数，三项开关全关后 **25,175,144** 参数（与 v8 相同）；`content_query`/`content_key`/两层 `edge_value` 均获得有限非零梯度（量级 5e-3 ~ 6e-2），`center_log_scale` 梯度 2.8e-3。
- 已知与本次改动无关的失败（均在**未改动的 HEAD** 副本上复现）：CPU 环境下 `AmpOptimizerWrapper` 不可用，导致 `tests.rpgv.test_rpgv_v7`/`test_rpgv_v8` 各一项优化器分组测试、`tests.test_deepglobe_training` 四项配置比对与一项 runner 训练用例失败；`tests.rpgv.test_rpgv_v6.V6ConfigTests` 仍断言 `train_dataloader.batch_size == 2`，与当前 batch8 × 累积1 协议不符。v8.1 相关测试不受这些影响。

### 架构导向任务调整验证（2026-10-13，CPU）

- Docker 中执行 `tests.rpgv.test_rpgv_v81 tests.test_rpgv_v81_integration tests.test_run_experiments tests.test_experiment_protocol`：**75 项执行、74 项通过、1 项跳过**（CUDA 用例）。覆盖仅内容支持解析参考与 double gradcheck、置换等变、同 primitives 的 coverage 不变、几何扰动不改变 content-only A、支持来源双关闭报错、默认参数键与显式默认模式一致、最终任务仍训练内容/几何、无高斯辅助时无残留 head/loss，以及全部 v8.1 配置的训练协议和核心任务优化器参数组。
- dry-run 覆盖四数据集完整 `v81`、WWTP 七项核心 / 15 项总队列及五项按需显式选择；命令数量、WWTP 限定、独立目录和无伪几何/gate/自动重训检查通过。
- `smoke_test` 对完整模型、七项核心及五项按需配置，共 **13 个实际配置**完成 CPU 前向、反向和预测。
- `smoke_runner` 对完整模型与 `content_only_support` 均完成合成短步训练→保存→测试→同配置优化器恢复，中心参数继续更新。数据和 checkpoint 仅写入容器临时目录，未启动正式数据集训练。
- 附带 v8 回归的首轮共 101 项：98 项通过、2 项跳过、1 项失败。失败为未改动的 `tests.rpgv.test_rpgv_v8.V8ModelTests.test_optimizer_covers_all_parameters_once` 在 CPU 构造 `AmpOptimizerWrapper`，报 `AmpOptimizerWrapper is only available training on gpu, npu, mlu or musa`。该既有 CPU 限制未在本轮修改；v8.1 核心参数组通过已有协议测试的 CPU 包装器独立校验。

### 本次附带的测试脚手架修复

- `tests/smoke_runner.py`：CPU 分支切到 `OptimWrapper` 时同时移除 AMP 专有的 `dtype`（此前只移除 `loss_scale`，导致所有模型在 CPU 上无法运行该冒烟）。这是测试脚手架修复，不改变训练协议。
- `tools/rpgv/profile_rpgv_v8.py`：CPU 分支同样回退为普通包装器，并允许 v8.1 模型类型与开关；`tools/rpgv/profile_rpgv_v81.py` 复用该实现并暴露 `--no-edge-value/--no-content-support/--no-local-detail`。

### 限制

- **没有任何精度证据。** 上述验证只说明实现可运行、可学习、开关可控、关闭后与 v8 等价，不证明 v8.1 优于 v8 或 v7。
- v8.1 的显存与吞吐必须在训练机上实测（`tools/rpgv/profile_rpgv_v81.py --size 1024 --batch-size 8 --stages`），CPU 合成数字不代表正式训练。
- 恢复的配对监督与 v7 的实现同源，但 v8.1 没有 v7 的伪几何、层次对比与组合扰动目标；**不能把 v8↔v8.1 的差值整体归因于任一项**，应按 §5 逐项消融。
- 内容感知支持改变了“纯高斯溅射”的语义：修改后应表述为**高斯先验引导的内容自适应支持**，不再是单纯的高斯核归一化。
