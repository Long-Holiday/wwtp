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

## 5. 核心消融

`v81-ablations` 仅适用于 WWTP，六项均为单因素：

| 配置 | 变化 | 检验的问题 |
|---|---|---|
| `no_edge_value` | 关系层去掉边值分支（`use_edge_value=False`） | 关系属性是否必须进入消息内容 |
| `no_content_support` | 支持退回纯高斯核（`content_aware=False`） | 内容相似度是否改善支持贴合 |
| `no_local_detail` | F8/F4 局部融合退回 v8 形式（`local_detail=False`） | 高分辨率 RGB 细节通道是否有效 |
| `no_pairwise` | `region_loss_weight=0`、`final_boundary_loss_weight=0` | 恢复的配对监督贡献 |
| `no_dense_dice` | `gaussian_dense_dice_weight=0` | 稠密高斯 soft Dice 贡献 |
| `unbalanced_band` | `balanced_boundary=False`（仅二分类有差异） | 前景/背景平衡的边界带贡献 |

`no_content_support`、`no_edge_value`、`no_local_detail` 关闭的是与本次机制一一对应的开关。**三项同时关闭时，v8.1 在数值上退化为 v8**：同一 seed 下 `state_dict` 可 `strict=True` 互相加载，`encode_decode` 输出与全部损失项逐位相同（`test_all_new_switches_off_reproduce_v8`）。`no_pairwise` 仍保留最终/粗分割、SDF 与边界带，不能把它的差值解释为“取消全部输出监督”。`v81-all` = 七基线 + 完整 v8.1 + 六项消融，共 14 组；`all`/`v7-all`/`v8-all`、v7 superiority gate 与自动补齐重训逻辑均不改变。

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

### 本次验证（2026-10-13，CPU）

- `tests.rpgv.test_rpgv_v81` **21 项执行、20 项通过、1 项跳过**（CUDA 用例）；覆盖内容支持与解析参考的逐位一致、零内容退化为 v8、置换等变、double gradcheck、预测器开关无残留参数、关系聚合与显式边值实现输出/梯度一致、节点置换、局部融合依赖高分辨率 RGB、平衡边界带与 v7 数值一致且与 v8 不同、配对损失有符号性、忽略标签零损失、soft Dice 解析值、全开关关闭退化为 v8、全忽略零损失、逐开关前后向、多类/滑窗、checkpoint 往返与三通道契约。
- `tests.test_rpgv_v81_integration tests.test_run_experiments` **34 项通过**（含套件规模、四数据集默认值、单因素差异、无伪几何生成、无 v7 gate）。
- 生产入口冒烟：`smoke_runner` 对 `rpgv_v81` 与 `rpgv_v8` 均完成 训练→保存→测试→恢复；`smoke_test` 对 `rpgv_v81` 与六个消融配置、以及 LoveDA/UAVid/DeepGlobe 三套公共配置完成前向与反向。
- 四数据集 `--suite v81` 与 `v81-all`/`v81-ablations` dry-run 通过，**0 次 `generate_pseudo_geometry`**。
- CPU 合成 profiler（64×64、batch1）完整模型 **25,239,400** 参数，三项开关全关后 **25,175,144** 参数（与 v8 相同）；`content_query`/`content_key`/两层 `edge_value` 均获得有限非零梯度（量级 5e-3 ~ 6e-2），`center_log_scale` 梯度 2.8e-3。
- 已知与本次改动无关的失败（均在**未改动的 HEAD** 副本上复现）：CPU 环境下 `AmpOptimizerWrapper` 不可用，导致 `tests.rpgv.test_rpgv_v7`/`test_rpgv_v8` 各一项优化器分组测试、`tests.test_deepglobe_training` 四项配置比对与一项 runner 训练用例失败；`tests.rpgv.test_rpgv_v6.V6ConfigTests` 仍断言 `train_dataloader.batch_size == 2`，与当前 batch8 × 累积1 协议不符。v8.1 相关测试不受这些影响。

### 本次附带的测试脚手架修复

- `tests/smoke_runner.py`：CPU 分支切到 `OptimWrapper` 时同时移除 AMP 专有的 `dtype`（此前只移除 `loss_scale`，导致所有模型在 CPU 上无法运行该冒烟）。这是测试脚手架修复，不改变训练协议。
- `tools/rpgv/profile_rpgv_v8.py`：CPU 分支同样回退为普通包装器，并允许 v8.1 模型类型与开关；`tools/rpgv/profile_rpgv_v81.py` 复用该实现并暴露 `--no-edge-value/--no-content-support/--no-local-detail`。

### 限制

- **没有任何精度证据。** 上述验证只说明实现可运行、可学习、开关可控、关闭后与 v8 等价，不证明 v8.1 优于 v8 或 v7。
- v8.1 的显存与吞吐必须在训练机上实测（`tools/rpgv/profile_rpgv_v81.py --size 1024 --batch-size 8 --stages`），CPU 合成数字不代表正式训练。
- 恢复的配对监督与 v7 的实现同源，但 v8.1 没有 v7 的伪几何、层次对比与组合扰动目标；**不能把 v8↔v8.1 的差值整体归因于任一项**，应按 §5 逐项消融。
- 内容感知支持改变了“纯高斯溅射”的语义：修改后应表述为**高斯先验引导的内容自适应支持**，不再是单纯的高斯核归一化。
