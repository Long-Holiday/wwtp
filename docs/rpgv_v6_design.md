# RPGV v6：解码器几何融合与独立轮廓残差

## 1. 当前实现与兼容边界

本版本在原 v6 基础上优化架构与训练路径，**只改 v6 相关实现**，v1–v5.1 的模型和配置保持原样。

主要变化：

- MiT-B2 使用 SDPA，不再计算随后丢弃的注意力权重。
- 几何从编码器 C1/C2 写回改为解码器 **F8/F4** 残差融合。
- 高频分割路径不再被对比样本有效性阈值关闭。
- 使用可靠度归一化的正核 Gaussian 频带分解，减少 Q 孔洞伪边缘。
- 轮廓头分别预测辅助 SDF 和**无固定幅度限制的 logit 残差**。
- 高低频融合增益可学习，默认不随机关闭几何样本。
- SDF 目标在训练数据 worker 中生成，正常固定尺寸训练不再每步同步到 CPU 做 EDT。

这是 WWTP 二分类模型，不是 UAVid/LoveDA 的原生多分类实现；不加载 CLIP 权重或文本编码器。

**旧 v6、v5/v5.1 的完整 checkpoint 不能直接作为本版 resume。** 投影、轮廓头和部分参数定义已改变；默认从 ImageNet MiT-B2 初始化新的联合训练。`--resume` 仅用于本版相同配置的 checkpoint，不做隐藏权重转换。SDPA 骨干保留原 MiT 参数键和形状，ImageNet 初始化路径不变。

关键文件：

- `wwtpseg/models/segmentors/rpgv_net_v6.py`
- `wwtpseg/models/utils/rpgv_v6_modules.py`
- `wwtpseg/models/utils/rpgv_v6_decoder.py`
- `wwtpseg/models/backbones/mit_sdpa.py`
- `wwtpseg/rpgv_v6_targets.py`
- `wwtpseg/datasets/transforms/rpgv_v6_transforms.py`
- `configs/rpgv/v6/rpgv_v6.py`

## 2. 架构与模态独立性

输入沿用 `[BGR, depth, reliability]` 五通道约定，数据管线为 0–255。模型内部归一化 RGB，将深度/Q 恢复到 `[0,1]`。三通道输入视为没有几何。

```text
RGB ── MiT-B2/SDPA ── C1/C2/C3/C4 ── RGB lateral P4/P8/P16/P32
                                             │
                               F16 → F8 + 几何残差 → 空间细化
                                             │
                                      F4 + 几何残差 → 空间细化
                                             │
                                         decoder特征
                                             │
RGB细节 ─────────────────────────────── shared contour trunk
                                             ├─ SDF辅助输出
                                             └─ Δlogit → base + Δlogit

D/Q → 可变形特征校正 → Dc/Q → Q归一化低通 → low/high
                                              ├─ high: 局部卷积 + 软区域上下文
                                              └─ low: RGB query / 几何 key,value
```

RGB 骨干始终只读取 RGB。几何描述子、区域分配和几何原型不读取 RGB。区域对比的 RGB 原型来自**尚未注入几何的 P4/P8 lateral 特征**，不是已包含 F8 几何的 F4；对比 RGB 投影与融合 query 投影独立。

几何残差在几何通道中计算，直接投影到 decoder 通道。显式传递 `delta`，不通过 `fused-rgb` 恢复小残差。残差先以 FP32 相加，再恢复输入特征 dtype，避免改变 Q=0 回退路径的数值类型。

## 3. 几何分支

### 3.1 可变形深度校正

校正器在首个 RGB 特征的空间尺寸上工作；1024 输入通常为 256×256。

- 在任何池化/空间混合前屏蔽 Q=0 的原始深度。
- 可靠度加权池化生成基础深度。
- 导数采用有支持的相邻差分，缺失邻居不形成零填充深度跳变。
- 编码基础深度、Q、dx/dy，预测 18 个有界偏移和 9 个调制权重。
- 复用 `BoundedModulatedDeformConv2d` 的可微 `grid_sample` 实现，无特定 DCN 扩展要求。
- 偏移默认限制为 ±2 个特征像素，深度加性校正默认限制为 ±0.1。

输出区分：

- `corrected_depth = Dc`：未乘 Q 的校正深度；
- `trusted_depth = Q × Dc`：可信深度；
- `reliability`、`base_depth`、`correction`、`offsets`。

校正量不再先乘 Q、然后又随 corrected depth 乘 Q。必要的物理边界仍保留，没有额外收紧。校正器读取几何特征后预测加性残差，**不是直接 warp 深度坐标的 RGB 配准器**。

### 3.2 可靠度归一化的互补频带

在 FP32 中用同一个正权重、归一化可分离 Gaussian 算子 G：

```text
T = Q × Dc
M = G(T) / G(Q)        # 无支持处为0
low  = Q × M
high = T - low
```

约定：

- `low + high = trusted_depth`，浮点误差范围内成立。
- 常量观测深度不会仅因 Q 孔洞/软 Q 边界产生明显高频。
- Q=0 的任意原始深度值不参与滤波；全 Q=0 时两频带为零。
- 非零低 Q 不通过额外阈值截断，避免大分母下限破坏低 Q 的归一化。
- 允许省略 Q，此时 Q=1，两频带重建输入深度。

`frequency_cutoff=0.15` 对应特征网格上的 Gaussian 带宽，`sigma=1/(2π×cutoff)`；`frequency_padding=8` 是有限核半径，边界反射填充，小尺寸自动缩短核并重新归一化。

本版不再使用原 super-Gaussian FFT 掩码：其空间核可能有负旁瓣，不适合直接作为置信度归一化分母。这里是 **Q 条件下的频带分解**，不是全局线性 FFT 分区；不宣称它实现了学习式深度补全。

### 3.3 高频局部细节与区域上下文

各尺度先编码高频，再以归一化几何描述子相似度和二维空间距离执行两轮软聚类。

- 聚类 token 网格最多 **64×64**；1024 图像上约每 token 对应 16×16 原图像素。
- 初始原型为 4×4，小输入自动缩小。
- 用 `assignment × Q` 聚合原型。
- 分割细节保留局部特征，并融合所有有 Q 支持的区域上下文。
- **不使用 InfoNCE 的有效性 mask 关闭分割 detail。** Q=.2/.34 即使没有有效对比区域，分割损失仍能训练高频编码器。

对比有效性判定仍用于选择有意义的配对，不控制分割支路：

| 判定 | 默认值 |
|---|---:|
| 区域平均可靠度 | ≥0.35 |
| 可靠区域质量/token质量和 | ≥2.0 |
| 区域高频绝对能量 | ≥1e-4 |

最终对比池化权重 `detach`，两个模态使用同一空间分配。分割软聚类仍可微；对比目标不通过挪动分配来降低匹配难度。

### 3.4 低频注意力与融合

RGB/decoder 特征是 query，低频几何是 key/value。query 最多32×32、key/value 最多8×8，4个头，加入二维位置和 Q 对数先验；完全无效的 key 不提供上下文。矩阵运算/softmax 保持 FP32。

融合：

```text
delta = Q × (gain_high × gate_high × high_detail
           + gain_low  × gate_low  × low_context)
```

两个 gain 是独立可学习参数，初始0.1，不加第二重 sigmoid/tanh/硬截断。gate bias=0、末层小幅非零初始化，初始 gate 约0.5，首步可训练门控各层。Q=0 时 delta 严格为零；可靠度仍用于池化与最终融合，并非删除所有质量信息。

## 4. 轮廓细化与损失

`ResidualContourHeadV6` 在 ceil-half 输入分辨率上融合 RGB 细节、decoder 语义和 coarse 概率。共享 trunk 后分别输出 SDF 与 delta-logit：

```text
final_logits = upsample(coarse_logits) + delta_logits
```

delta 不受熵门控、SDF tanh 或固定 `max_logit_correction` 限制，因此能独立修正高置信度粗预测错误。SDF 仍是辅助目标，其 loss 内的 tanh 不限制分割残差。`max_logit_correction` 仅对 `contour_mode='legacy'` 消融生效。

默认损失：

| 名称 | 权重/定义 |
|---|---|
| `loss_final` | masked BCE + soft Dice |
| `loss_coarse` | 0.3 × coarse BCE/Dice |
| `loss_sdf` | 0.1 × contour-weighted Smooth L1 |
| `loss_boundary_band` | 0.05 × GT边界带平衡BCE |
| `loss_region_contrastive` | 0.05 × 两尺度对称InfoNCE均值 |
| `loss_depth_rectification` | 0.01 × Q加权归一化校正/偏移平方 |
| `region_valid_count` | 日志指标，不加入总损失 |

区域 InfoNCE 的两个投影头是 `Linear → LayerNorm → tanh-GELU → Linear`，64维、L2归一化、初始温度0.07。可学习相似度尺度保留既有 `[1,100]` 数值范围。使用固定候选数的掩码对称CE，不再按 GPU count 走 Python 分支；不足2对时返回连接两侧头与参数的零损失。

负样本只来自当前 rank 的当前 microbatch。batch2、每图16区域时每尺度最多32对；累积4次不会扩大为128对。语义假负样本问题仍需真实数据验证，没有以增加负样本池作为默认“优化”。

旧 region/final-boundary 损失默认权重为零时，v6 不再计算后丢弃；其他版本的 loss 实现未改。

## 5. 训练数据目标与配方

训练 pipeline 最后使用 `PackRPGVContourInputs`，在所有 flip 之后于 CPU worker 生成 SDF/valid 并存入 `PixelData`。保持原目标定义：adaptive-average occupancy、阈值≥0.5、完整观测cell、到已观测相反类别的EDT、均匀图±1、截断和ignore-safe区域。

缓存必须匹配原始栅格尺寸、head尺寸、truncation。batch padding 或其它不匹配时从实际标签精确重算；不缩放/补齐已经计算的 SDF。直接构造样本的测试/调用仍支持原计算路径。新 SDF 头通常为1/2输入尺寸，截断5个 SDF 像素对应约10个原图像素。

默认保持：

- 1024整图、1m/px、既有正负样本比例；不新增crop/resize。
- MiT-B2 ImageNet初始化，所有分支联合训练，不冻结RGB。
- RGB骨干LR为0.1倍；batch2 × 累积4。
- 20,000 microsteps = 5,000 optimizer updates，原warmup/poly调度不变。
- `geometry_dropout_prob=0`；dropout仍可显式启用实验，不默认压制几何样本。
- `branch_gains` 和 `logit_scale` 不做 weight decay。
- `rgb_encoder.with_cp=True`，通过激活重计算适配低显存；大显存GPU可关闭，不改变训练目标。

## 6. 启动、消融与评估

```bash
# WWTP_DATA_ROOT / WWTP_PSEUDO_ROOT 沿用原数据约定
# 必须有逐像素匹配的images/annotations/pseudo_geometry

docker compose run --rm wwtp bash scripts/rpgv/train_rpgv_v6.sh

# 大显存GPU可关闭激活重计算
# 用不同RPGV_V6_WORK_ROOT避免混用实验目录
docker compose run --rm -e RPGV_V6_WORK_ROOT=work_dirs/rpgv_v6_no_cp_1m \
  wwtp bash scripts/rpgv/train_rpgv_v6.sh --cfg-options model.rgb_encoder.with_cp=False

# 只恢复同版同配置checkpoint
docker compose run --rm wwtp bash scripts/rpgv/train_rpgv_v6.sh --resume

# 使用训练时的同一份配置
docker compose run --rm wwtp python tools/test.py configs/rpgv/v6/rpgv_v6.py \
  work_dirs/rpgv_v6_optimized_1m/iter_20000.pth --split val
```

启动脚本支持 `RPGV_V6_CONFIG`、`RPGV_V6_WORK_ROOT`，仍不覆盖已有非resume目录。默认目录改为 `work_dirs/rpgv_v6_optimized_1m`。

| 配置 | 单因素含义 |
|---|---|
| `rpgv_v6.py` | 完整优化版 |
| `rgb_only.py` | 同decoder/轮廓头，无几何模块 |
| `no_contrastive.py` | 几何保留，无对比头/专用外观投影 |
| `encoder_fusion.py` | 几何注入C1/C2，其余优化固定；不是历史v6复现 |
| `legacy_contour.py` | 使用原熵门控/有界SDF修正头 |
| `cluster32.py` | 聚类网格上限32，对照默认64 |

例如：

```bash
docker compose run --rm -e RPGV_V6_CONFIG=configs/rpgv/v6/no_contrastive.py \
  wwtp bash scripts/rpgv/train_rpgv_v6.sh
```

## 7. 验证与合成性能工具

```bash
# CPU专项：包含模态独立性、低Q梯度、频带孔洞、缓存、SDPA严格权重重载
python -m unittest tests.rpgv.test_rpgv_v6 tests.rpgv.test_mit_sdpa \
  tests.rpgv.test_contour_targets -v

# CUDA专项：FP16缺失几何/all-ignore/dropout与训练后的精确RGB回退
python -m unittest tests.rpgv.test_mit_sdpa.V6CUDAAMPTests -v

# 真实生产入口：合成数据→训练→保存→评估→恢复
python tests/smoke_runner.py --config configs/rpgv/v6/rpgv_v6.py --device cpu

# 1024/batch2前后向+AdamW；无数据集、预训练下载或训练权重保存
python tools/rpgv/profile_rpgv_v6.py --output work_dirs/profiles/v6_train.json

# 推理、关闭checkpointing、关闭缓存或attention实现对照
python tools/rpgv/profile_rpgv_v6.py --mode infer --batch-size 1 \
  --output work_dirs/profiles/v6_infer.json
python tools/rpgv/profile_rpgv_v6.py --size 512 --batch-size 1 --with-cp off
python tools/rpgv/profile_rpgv_v6.py --size 512 --batch-size 1 --no-cache-targets
python tools/rpgv/profile_rpgv_v6.py --size 512 --batch-size 1 --attention weights
```

profile保留配置的优化器参数组与梯度裁剪，检查裁剪前范数、AMP scale、所有参数的梯度和真实骨干更新。它不模拟梯度累积、LR调度、数据IO、验证或真实收敛。

PyTorch 2.1 CPU在混合内存布局下的精确GELU backward存在已知问题；v6专用新模块使用tanh近似和明确连续布局，历史模块没有被全局替换。

### WSL容器GPU兼容

如果 `docker compose` 的 `gpus: all` 报 `no known GPU vendor found`，可在WSL环境使用已验证的设备/驱动只读映射，不需要修改系统GPU配置。例如运行合成GPU测试：

```bash
docker run --rm --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -v "$PWD":/workspace -w /workspace -e PYTHONPATH=/workspace \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  wwtp-mmseg:torch2.1.2-cu121 python tools/rpgv/profile_rpgv_v6.py
```

## 8. 本次验证结果与局限

在RTX3050 Laptop 4GB / WSL / PyTorch2.1.2上，随机初始化、合成数据、FP16、2步预热后3步测量：

| 项目 | 平均耗时 | CUDA allocated峰值 |
|---|---:|---:|
| 1024/batch1推理 | 121ms | 531MiB |
| 1024/batch2完整前后向+AdamW | 1.25s | 3549MiB |
| 512/batch1完整前后向+AdamW，关闭CP | 234ms | 1133MiB |

1024训练完成5步，AMP scale保持128，所有梯度有限，RGB/几何实际更新；生产入口保存/评估/恢复通过。原始v6的1024/batch2在本环境曾OOM。512短测并非所有改动都更快，不能宣称普遍吞吐提升。

WSL allocator可能包含共享系统内存，不等于物理显存占用。少量合成计时受功耗/预热影响，不是长期吞吐保证。SDPA具有浮点差异，不承诺逐位恒等。

没有真实数据/训练权重时，这些测试只证明路径可训练、目标定义一致和资源可用；不能证明前景IoU、边界或小目标指标提升。建议以RGB-only、无对比、完整v6建立同预算对照，再使用上述结构消融。
