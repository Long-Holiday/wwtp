# RPGV v6：可变形深度校正、频域几何与区域级对比学习

## 1. 设计与实现边界

v6 实现以下路径：伪深度经过有界可变形卷积校正后分解为高低频；高频通过空间约束软聚类得到几何原型，并与**同一区域的 RGB 外观原型**进行双向 InfoNCE；低频作为 Key/Value 与 RGB Query 进行交叉注意力；两个分支经过可靠度门控，以残差方式融入 RGB 特征。

这里的“CLIP 思想”指两个独立投影头、共同嵌入空间、L2 归一化、温度缩放和对称对比目标，**不加载 CLIP 权重，不需要文本编码器，也不要求 RGB 原型必须来自高通特征**。

- 模型：`wwtpseg/models/segmentors/rpgv_net_v6.py`，注册名 `RPGVNetV6`。
- 模块：`wwtpseg/models/utils/rpgv_v6_modules.py`。
- 主配置：`configs/rpgv/v6/rpgv_v6.py`。
- 沿用 v5 的 RGB-only MiT-B2 编码器接口，以及 v5.1 的空间细化解码器、轮廓头、边界带/SDF 监督和整图/滑窗推理。
- **这是新的联合训练架构，不是 v5.1 的预测恒等续训版本。** 默认仅使用 MiT-B2 的 ImageNet 初始化，新几何模块随机初始化；不要直接把 v5/v5.1 checkpoint 当作 v6 完整权重。`--resume` 只用于同配置的 v6 checkpoint。
- 当前是 WWTP 二分类模型，没有把它宣称为 UAVid/LoveDA 的原生多分类实现。

## 2. 计算路径

输入沿用项目五通道约定：`[BGR, depth, reliability]`，数据管线均为 0–255；RGB 在模型内归一化，深度和可靠度恢复到 `[0, 1]`。三通道输入会使用全零几何可靠度。

### 2.1 几何独立的可变形校正

`DeformableDepthRectifierV6` 在首个 RGB 特征的空间分辨率上工作，1024 输入通常对应 256×256，避免在整图尺寸运行九次可变形采样。

1. 先屏蔽 `Q=0` 的深度，再用可靠度加权池化得到基础深度。
2. 用基础深度、Q、水平/垂直导数编码几何；预测 18 个偏移和 9 个调制权重。
3. 偏移由 `tanh` 限制在 ±2 个特征像素，调制为 sigmoid。
4. 复用项目已有的 `BoundedModulatedDeformConv2d`，基于可微 `grid_sample`，不依赖特定设备上的 DCN 扩展。
5. 深度残差限幅到 0.1，再由 Q 缩放；校正后的信号在进入 FFT 前再次乘 Q，避免加权平均的归一化把极低可靠度深度恢复为完整幅值。

偏移、调制和深度残差**不读取 RGB**，以免后续“几何—外观对齐”退化为先复制 RGB、再与自身比较。深度残差输出使用很小的非零初始化，保证首步分割损失就能传到校正器；偏移初始为零。

### 2.2 互补频带

`SmoothFrequencySplitV6` 在 FP32 下执行 `rfft2/irfft2`，以平滑径向低通掩码分解信号：

- `M(f) = exp(-0.5 * (||f||² / cutoff²)²)`；默认 `cutoff=0.15`，单位是当前深度特征网格的 cycles/pixel。
- `low = IFFT(FFT(depth) * M)`。
- `high = depth - low`，保证两频带相加重建输入（浮点误差范围内）。

FFT 前最多反射填充 8 个特征像素，变换后裁回原尺寸，减轻周期边界的首尾串扰；也支持奇数尺寸和非 2 次幂尺寸。FP16/BF16 下频域计算仍为 FP32。

这并不等于可靠度感知的深度补全：缺测孔洞的边界仍可能进入频谱。输入置信度加权、区域质量过滤和最终门控降低其影响，但没有声称彻底消除孔洞伪边缘。

### 2.3 高频空间聚类

在 RGB 的 1/4、1/8 两个尺度分别进行：

- 高频深度编码为 32/64 通道的几何特征。
- 聚类网格最多 32×32 token，初始区域中心为 4×4 空间网格；实际小输入会自动缩小网格。
- 两轮软聚类综合归一化几何描述子的余弦相似度和二维空间距离。空间约束避免把相距很远的相似边缘无条件合成同一区域。
- 使用 `soft_assignment × Q` 池化几何原型和 RGB 原型，二者使用**完全相同的区域权重**。RGB 原型来自未注入几何的 MiT 特征投影。
- 分割分支保留可微聚类，将区域原型广播回空间，并与局部高频特征融合。
- 对比学习的最终池化权重执行 `detach`；对比损失训练两个模态的描述子，不通过直接移动区域分配来降低匹配难度。

有效区域同时满足：

| 条件 | 默认值 | 含义 |
|---|---:|---|
| 区域平均可靠度 | ≥ 0.35 | 分配加权后的 Q |
| 可靠区域质量 | ≥ 2.0 | 聚类 token 网格上的权重总和，不是原图像素数 |
| 区域高频能量 | ≥ 1e-4 | 高频绝对值的区域均值 |

三项判定均不反传。训练时 ignore 标签（255，包括预处理 padding）的 Q 被置零，不能成为有效对比区域。空白/无纹理区域、低可靠度区域和被 geometry dropout 禁用的样本不会被强行设为有效正负样本。

### 2.4 区域级双向 InfoNCE

每个尺度有独立的几何/RGB 投影头：`Linear → LayerNorm → GELU → Linear`，嵌入维度 64，输出 L2 归一化。

将当前进程的有效区域展平，得到成对嵌入 `z_g[i]`、`z_r[i]`：

- `S[i,j] = exp(logit_scale) * z_g[i]ᵀ z_r[j]`。
- 对角线同图同区域是正样本，其他有效区域是负样本。
- `L_region = 0.5 * (CE(S, arange(N)) + CE(Sᵀ, arange(N)))`。
- 初始温度 0.07；可学习相似度尺度限制在 `[1, 100]`，且不做 weight decay。
- 两个尺度的对比损失取均值，默认总权重 0.05。
- 少于两个有效区域时返回与两侧投影头、特征和温度参数均连接的零损失，保持反向传播/DDP 参数图完整。

**负样本范围是当前 rank 的当前 microbatch。** 不进行跨 GPU gather，梯度累积也不会扩大负样本池。默认 batch=2、每图最多16区域，因此每尺度最多32对，而不是把有效 batch=8 当作128对。语义相似的不同区域仍可能成为假负样本；这是一项需要通过验证集消融观察的局限，不是已经解决的问题。

推理不运行对比投影头或 N×N 对比 logits；高频聚类仍保留，用于构造分割特征。

### 2.5 低频与 RGB 交叉注意力

`LowFrequencyCrossAttentionV6` 明确定义方向：RGB 为 Query，低频几何为 Key/Value。

- Query 网格最多32×32，Key/Value 最多8×8，默认4个头。
- 两侧加入二维位置投影；低频 token 在池化前先按 Q 加权。
- 注意力加入 Q 的对数先验，完全不可靠的 Key 权重为零；所有 Key 均无效时返回有限的零上下文。
- 注意力矩阵乘法/softmax 使用 FP32，之后恢复调用方 dtype 并上采样。

每尺度最多 `4 × 1024 × 64` 个注意力分数，不构建 1024 原图像素之间的全局平方注意力。

### 2.6 门控残差与 RGB 回退

将投影后的 RGB、高频聚类细节、低频上下文和 Q 拼接，预测两个 sigmoid 门控；初始门控 bias=-2，残差系数0.1。几何残差投影回 C1/C2 的原通道数，再交给 v5.1 解码器。旧版几何编码器和旧几何融合层不实例化，避免重复融合和无用参数。

最终更新形式为：

`RGB_i + W_out(0.1 × Q_i × (gate_high × high_detail + gate_low × low_context))`。

因此 Q=0 时几何残差严格为零；没有深度的三通道输入与 `_run_rgb_branch` 精确一致。Q=0 区域的任意深度值在空间混合前已经被屏蔽，不会泄漏到相邻有效区域。

## 3. 损失与训练配置

保留 `loss_final`、`loss_coarse`、`loss_sdf` 和 `loss_boundary_band`，新增：

- `loss_region_contrastive`：上述双向区域对比，权重0.05。
- `loss_depth_rectification`：按 Q 加权的归一化深度残差与偏移平方惩罚，权重0.01，限制不必要的校正。
- `region_valid_count`：每尺度平均有效区域数，只作日志，不计入总损失。

默认联合训练，不冻结 RGB 分支，保证两侧表征都受到对比目标训练。沿用1024整图、1 m/px、batch2×累积4、20,000 microsteps＝5,000次优化器更新；RGB骨干使用0.1倍学习率。

```bash
# 数据目录沿用 WWTP_DATA_ROOT / WWTP_PSEUDO_ROOT；兼容已有 WWTP_V5_DATA_ROOT
# 如需自定义路径，通过 compose 环境配置传入容器。
docker compose run --rm wwtp bash scripts/rpgv/train_rpgv_v6.sh

# 关闭对比目标，保留频域、聚类、交叉注意力和门控路径
docker compose run --rm -e RPGV_V6_CONFIG=configs/rpgv/v6/no_contrastive.py \
  wwtp bash scripts/rpgv/train_rpgv_v6.sh

# 同解码器的 RGB-only 对照，不构建几何/对比模块
docker compose run --rm -e RPGV_V6_CONFIG=configs/rpgv/v6/rgb_only.py \
  wwtp bash scripts/rpgv/train_rpgv_v6.sh

# 恢复同一个 v6 实验
docker compose run --rm wwtp bash scripts/rpgv/train_rpgv_v6.sh --resume

# 评估：使用训练时的同一份配置
docker compose run --rm wwtp python tools/test.py configs/rpgv/v6/rpgv_v6.py \
  work_dirs/rpgv_v6_1m/iter_20000.pth --split val
```

启动脚本支持 `RPGV_V6_CONFIG`、`RPGV_V6_WORK_ROOT`，已有输出目录且未指定 resume 时拒绝覆盖。主配置目录是 `work_dirs/rpgv_v6_1m`，两个消融分别为 `work_dirs/rpgv_v6_no_contrastive_1m` 与 `work_dirs/rpgv_v6_rgb_only_1m`。

## 4. 验证入口

```bash
# 不需要数据集或预训练权重
python -m unittest tests.rpgv.test_rpgv_v6 -v

# 真实 train.py → 保存 → test.py → 恢复，使用临时合成五通道数据
python tests/smoke_runner.py --config configs/rpgv/v6/rpgv_v6.py --device cpu

# 通用模型合成验证已纳入 v6
python tests/smoke_test.py --include-rpgv --backward --device cpu
```

专项测试覆盖频带重建、FP16 FFT输入、校正幅值与可靠度隔离、区域对应关系、对比公式/双侧梯度、空负样本池、全部参数有限梯度、两步优化与严格权重重载、all-ignore/dropout、同批部分样本缺失几何、RGB回退、奇数尺寸、滑窗、BF16 autocast以及生产/消融配置解析。

PyTorch 2.1.2 的 CPU 精确 GELU 内核在输入与反向梯度内存布局不同时可能给出错误梯度。新增模块统一采用 tanh 近似 GELU，注意力输出显式恢复连续 NCHW，并测试不同梯度布局的数值一致性。Runner 冒烟测试同时检查梯度范数和保存权重的有限性，避免梯度裁剪把 `inf` 变成零更新而产生“测试通过”的假象。

合成测试用于证明代码路径可训练，不能证明真实数据上的精度提升，也不能替代目标 GPU 上的1024训练显存与吞吐测试。

## 5. 参考与区别

- [Deformable ConvNets v2, CVPR 2019](https://arxiv.org/abs/1811.11168)：参考有界采样偏移/调制思想；使用本项目的可移植算子，不声称完整复现 DCNv2 检测系统。
- [CLIP, ICML 2021](https://proceedings.mlr.press/v139/radford21a.html)：参考双模态共同嵌入与对称对比目标；这里的两个模态是区域几何和 RGB，不是图像与文本。
- [DenseCL, CVPR 2021](https://arxiv.org/abs/2011.09157)：参考密集视觉任务需要局部对应关系的思路；本实现使用深度几何定义的软区域，不声称复现 DenseCL 的增强匹配机制。

频率掩码、空间约束聚类与两尺度门控是面向当前任务的工程组合，尚未以实验确认优于 v5.1；建议首先比较完整 v6、无对比目标及 RGB-only，并跟踪有效区域数和验证集前景/边界指标。
