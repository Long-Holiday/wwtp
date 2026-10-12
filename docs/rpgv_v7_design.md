# RPGV v7：几何基元、关系对比与结构感知分割

## 1. 定位与版本边界

v7 在 v6 的 MiT-B2/SDPA、伪深度校正、局部几何融合和轮廓残差底座上新增独立结构路径：**几何基元 → 区域关系 → 局部组合 → 分割反馈**。它显式监督组成关系，而不是仅对齐同一区域的 RGB/几何描述子。

- `RPGVNetV7`：WWTP 二分类。
- `RPGVRemoteSensingV7`：LoveDA 7 类、UAVid 8 类，保留后者的滑窗推理。
- v1–v6 的模型实现和原配置不改；v7 使用独立模块与目录。
- 默认 ImageNet 骨干初始化、全分支联合训练；不加载 v6 完整权重，不冻结 RGB，不增加阶段预算。
- **v6 checkpoint 不能作为 v7 resume。** 仅在相同 v7 配置中恢复；节点数、结构开关或损失头变化也不可直接恢复优化器。
- 几何来自 Depth Anything 的相对伪深度，不是实测高度；软接触/邻接不是严格拓扑不变量。没有独立部件标注，区域节点不能直接被解释成有名称的工艺单元。

核心文件：

| 文件 | 职责 |
|---|---|
| `wwtpseg/models/utils/rpgv_v7_modules.py` | 几何软分区、测量描述、关系网络、组合采样与扰动 |
| `wwtpseg/models/utils/rpgv_v7_losses.py` | 区域/关系/组合对比及关系预测 |
| `wwtpseg/models/utils/rpgv_v7_decoder.py` | F16/F8 结构注入，保留 F8/F4 局部几何 |
| `wwtpseg/models/segmentors/rpgv_net_v7.py` | 完整二分类模型及结构辅助分割 |
| `wwtpseg/models/segmentors/multiclass/model_v7.py` | 复用现有多分类损失和滑窗接口 |

## 2. 基元构建

输入沿用 `[BGR, depth, reliability]`、0–255 范围，内部 D/Q 恢复到 [0,1]。主模型三通道输入视为没有几何。

### 2.1 结构网格

复用 `DeformableDepthRectifierV6` 和 `SmoothFrequencySplitV6`。结构输入包含：

- 校正相对深度；
- 低频与高频互补分量；
- 有效邻域的 x/y 深度差分。

先将几何观测可靠度归一化池化到最大 **64×64** 网格，再运行两个 128 通道卷积块。无需为结构路径构建全分辨率 128 通道 F4 张量；高分辨率定位继续由 v6 细节路径负责。

默认 8×8 空间种子、两次软聚类得到最多 64 个区域，小输入自动缩小种子网格。采用几何相似度与平面空间距离，不用 GT 构图，不按高频能量筛掉平坦节点。

对每个像素的分配 A 在节点维度归一化。几何节点按 A×Q 归一化池化。任意非零支持区域可参与学习，不增加可靠度硬阈值；全 Q=0 的节点保持无效。

### 2.2 两种梯度路径

- 分割图节点：通过软分配和几何编码共同反向。
- 对比图节点：池化权重 detach，几何编码仍可训练。对比损失不能通过移动其区域支持降低匹配难度。
- RGB 节点：由**未注入几何的 C2/C3/C4 特征**分别投影、缩放、平均，再用同一 detach 池化支持聚合。

RGB 对比头不接收 D/Q 数值、测量关系、几何节点或已融合特征。但是共享区域支持本身仍可能携带结构信息；固定网格消融用于检验该因素，不宣称完全消除了跨表示捷径。

### 2.3 固定测量描述

每个节点的 6 维内容描述为：

1. 有效区域面积占比的对数；
2. x 方向空间方差；
3. y 方向空间方差；
4. xy 空间协方差；
5. 相对深度均值；
6. 相对深度标准差。

平面质心单独存储，不混入可置换的内容描述。面积/空间矩使用 A×有效支持，不重复乘软 Q，防止低可靠度被解释成小面积。深度统计使用 A×Q。

深度测量目标来自**校正前的有效池化观测 `base_depth`**；测量描述、质心、接触、邻接和采样选择停止梯度。区域分区会随分割训练变化，因此这些不是跨训练固定的外部真值，也不是部件标签。

## 3. 空间关系与图推理

6 维有向关系依次为：平面距离、log(area_j/area_i)、depth_j−depth_i、软接触、x_j−x_i、y_j−y_i。

- 接触由四邻接方向上的软区域支持交叉量计算，并按面积几何均值归一化。
- 按距离与弱接触优先级选择每节点最多 8 个非自身邻居，再添加自身边；没有把所有非零软接触都作为边。
- 图最多 64 节点。使用有界 N×N 张量和邻接 mask，而不是依赖额外稀疏图扩展。
- 两层、128 维、4 头关系注意力。边描述同时影响注意力偏置和消息值；节点内容测量经线性投影加入节点特征。
- 不向 RGB 对比路径喂入此图的关系属性或消息。
- 空行先经有限值 softmax，再按有效边屏蔽和归一化；全无效节点输出严格为零。

默认图具有方向信息，**不承诺旋转/翻转不变**：翻转输入后重新计算对应关系，距离保持、方向同步改变。普通节点编号重排不改变图表示的数学含义；近邻截断遇到完全相同距离时，排序可能选择不同但同距的邻居。

## 4. 层次化监督

### 4.1 区域对比

同一区域的 RGB/几何节点是正对；非对应节点只有在固定测量描述的平均绝对差大于 `similar_threshold=0.05` 时作为负对。相似的非对应节点保持未知，不强行拉远，也不直接改成正对。

采用带可学习尺度/偏置的 sigmoid 配对损失：

- 正项对有效节点平均；
- 负项先对每个 anchor 的有效负例平均，再对有负例的 anchor 平均；
- 正负两项相加；没有负例时仍保留正项。

这是**平衡 sigmoid 目标**，不是 v6 的“全部配对求和/N”归一化。初始相似度尺度 10、偏置 0，尺度的数值范围沿用 [1,100]。仅当前 rank、当前 microbatch；梯度累积不扩充候选池。

测量描述的各维单位不同，该阈值是公开的初始工程值，不是已校准的语义距离。日志输出实际负对数量，避免把零负例下的正对学习误称为充分对比。

### 4.2 关系对比与测量预测

每个 microbatch 从有效非自身边中随机选最多 **128 个有向关系**，避免完整边集合的平方矩阵。

- RGB 关系头只读取 `(r_i, r_j)`。
- 几何关系头读取 `(g_i, g_j, e_ij)`。
- 同一关系对应匹配；关系描述相似的非对应负对被屏蔽。
- RGB 关系头额外通过 Smooth L1 预测 6 维测量关系；目标 detach。

RGB 的上下文来自原骨干，不是靠直接复制几何边解决任务。监督仍是伪几何测量，不是人工确认的地物功能关系。

### 4.3 两跳组合与结构扰动

每个节点选择两跳内最多 **16 个节点**，优先保留 anchor 和空间近邻，构成局部组合。

- RGB 组合：上下文 RGB 节点的 masked mean 和独立投影头。
- 几何组合：在局部组合上复用**同一个关系网络**，然后 masked mean 与投影。因此组合监督直接训练分割所用的关系网络。
- 每组合默认两个随机扰动候选。
- 在有效槽位之间置换节点内容特征及内容测量；质心和接触保持为原空间槽位，重新计算内容相关的边关系与图输出。
- 不是同步重排节点、位置和邻接的“重编号”，也不引入 GT 类别或新分割标签。
- 使用扰动前后有效测量边的平均差（阈值 1e-4）判定是否有可辨别变化。恒等置换、同质区域交换、单节点和无观测组合不产生强负例。

记正确匹配分数为 s+、扰动分数为 s−，分数为归一化表示的余弦：

`L_comp = mean_compositions(mean_valid_candidates(softplus(0.2 + s− − s+)))`

只对有有效候选的组合平均；没有有效候选时返回仍连接计算图的零损失。当前扰动判定主要由相对面积/深度关系改变触发，不把仅 embedding 改变当作有几何意义的负例。

这是图表示层面的结构匹配任务，不等于生成了具有相同纹理的真实影像，更不构成严格的因果理解证明。

## 5. 结构感知解码与损失

关系节点以 A 回投到稠密结构网格，在 F16/F8 与 RGB decoder 特征拼接后，通过 1×1 投影产生可学习残差。不叠加 sigmoid/tanh 幅度门控或反复乘软 Q；只在完全没有观测支持时关闭注入。

保留原 F8/F4 高频局部区域与低频几何融合，以及独立 SDF/轮廓 logit 残差。额外结构语义头直接从结构图预测低分辨率类别，并以有效标签/几何支持监督。

| 新增目标 | 默认权重 |
|---|---:|
| 区域对比 `loss_node_v7` | 0.05 |
| 关系对比 + 0.25×关系测量预测 `loss_relation_v7` | 0.05 |
| 组合扰动排序 `loss_composition_v7` | 0.05 |
| 结构辅助分割 `loss_structure_aux_v7` | 0.1 |

保留 v6 final/coarse、SDF、boundary band、depth rectification 损失。**v6 原区域对比被替换为新区域目标，不重复叠加**；继承配置中的 `contrastive_loss_weight` 在 v7 构造时固定为 0。

这些权重是可运行初值，不是通过真实实验选择的最优值。分别记录 `raw_*_v7`、有效节点/边/组合/负例数量和 `composition_gap_v7`；日志指标不进入总损失，并使用 MMEngine 可汇总的浮点 tensor。

结构辅助分割二分类使用 BCE/Dice，多分类使用 CE；继承现有类别与 ignore 语义。GT 仅参与语义损失与未知标签支持屏蔽，不生成区域、关系或扰动。

## 6. 消融与公平协议

所有配置继承统一外部协议：1024 整图、AdamW lr=6e-4、统一参数组规则、FP32、max_norm=10、batch2×累积4、30,000 microsteps/7,500 updates、相同 warmup/PolyLR、增强与选权重规则。消融只针对完整 v7 的机制和目标，不以版本差异作为消融结论。默认队列聚焦核心研究问题，更细的监督拆分和图机制配对按需运行。

### 6.1 默认核心消融

`v7` 提供完整模型基准，`v7-ablations` 提供以下六个对照，均直接与完整 v7 比较；加完整模型共七组。

| WWTP 配置 | 改变与对照解释 |
|---|---|
| `rpgv_v7.py` | 完整模型；不属于六项消融 |
| `rgb_only.py` | 删除全部几何/结构支路，保留 RGB decoder/轮廓；整体几何增强对照，不是局部模块单因素 |
| `fixed_grid.py` | 软几何分区替换为确定性网格，其余相同；验证自适应基元构建 |
| `no_feedback.py` | 只删除 F16/F8 结构注入，保留所有结构训练目标与局部几何融合 |
| `no_relation_supervision.py` | 仅关闭关系对比和测量预测，**保留组合排序**与区域/辅助监督；不是关闭关系图 |
| `no_composition.py` | 仅关闭整个组合扰动排序目标，其余监督保留；不是仅删除扰动采样而保留另一种组合目标 |
| `no_structure_aux.py` | 仅移除结构辅助语义 head/损失，保留所有层次目标、图与反馈；区分普通辅助监督与层次学习 |

优先验证整体几何、分区、反馈及关系/组合/辅助监督，不默认拆解每个层级。`no_relation_supervision` 检查的是关系对比与回归这一目标组，不能独立归因其中每个子项；默认六项也不单独证明区域目标或图消息传递的贡献。

### 6.2 按需分析

| 配置 | 保留的问题与对照范围 |
|---|---|
| `no_hierarchical_supervision.py` | 关闭区域、关系（含测量预测）、组合目标，保留结构网络、反馈及辅助分割；整个监督包分析，不等于关闭全部结构监督 |
| `no_node_supervision.py` | 仅关闭区域对比，保留关系、组合与辅助监督；更细的区域监督条件贡献 |
| `no_message_passing.py` | **相对 `no_composition`** 将关系注意力改为逐节点残差 MLP；保留区域、关系及辅助监督 |

以上三项通过显式 `--models` 选择，不进入 `v7-ablations` 或 `v7-all`。保留它们是为了可复现与后续机制分析，不代表必须全部补跑。整体层次与单层消融测量整体/条件贡献，单项收益不能简单相加。

**图推理必须做受控配对：`no_composition ↔ no_message_passing`。** 逐节点 MLP 后做 masked mean 无法区分有效节点内容的联合置换，保留组合排序会产生不可辨别的正负表示（有有效候选时排序项理论上为固定 margin 基准）。因此两侧共同关闭组合目标，但关系监督读取图推理前节点及测量关系，仍然有效，无需一并关闭。节点 MLP 仍通过结构反馈和辅助语义目标学习。

该配对替换了跨节点消息传递及边属性条件化，参数量也改变，应分别披露参数量；不是“仅将边属性归零”。`full ↔ no_message_passing` 同时改变组合目标与图，不能将该差值全部归因于消息传递。它仍按原区域支持回投节点，未删除每个节点的空间定位。

### 6.3 保留但不默认运行的历史配置

| 配置 | 保留语义与使用范围 |
|---|---|
| `no_contrastive.py` | 与新 `no_hierarchical_supervision` 模型设置相同；旧名字未准确体现关系回归也被关闭 |
| `no_perturbation.py` | 与新 `no_composition` 模型设置相同；实际关闭整个组合目标 |
| `no_relational_supervision.py` | 同时关闭关系和组合目标，保留区域对比；仅适用于历史嵌套监督分析 |
| `no_edges.py` | 继承旧 `no_relational_supervision` 再改为逐节点 MLP；仅与该旧配置配对，不与 full 作单因素归因 |
| `rgb_capacity.py` | 额外分支使用 RGB 灰度及单位支持，忽略外部 D/Q；是替代信息源控制，不是严格纯参数容量控制 |

这些配置仍能通过显式 `--models` 选择，路径、输出名字与模型语义不变；默认队列不再重复训练同义配置。新名字使用独立输出目录，不自动搬迁旧结果或恢复旧优化器。`rgb_only` 与历史 `rgb_capacity` 的 pipeline 不加载伪几何；灰度差不是独立几何，单位 Q 也不同于真实可靠度分布，不能仅以参数量解释替代信息源对照。

## 7. 入口

```bash
# 依赖无关的计划检查：原七基线 + v6 + 完整 v7 + 六个 v7 消融（15 项）
python scripts/run_experiments.py train-test --dataset wwtp --suite v7-all --dry-run

# 完整 v7 / 消融训练与验证选权重后的测试
python scripts/run_experiments.py train-test --dataset wwtp --suite v7
python scripts/run_experiments.py train-test --dataset wwtp --suite v7-ablations

# 按需图消息传递配对，不在默认队列；直接与完整 v7 的差值不是单因素
python scripts/run_experiments.py train-test --dataset wwtp \
  --models rpgv_v7_no_composition rpgv_v7_no_message_passing

# 历史配置仍可显式复现，但不自动加入主消融队列
python scripts/run_experiments.py train-test --dataset wwtp \
  --models rpgv_v7_no_relational_supervision rpgv_v7_no_edges

# 公开集完整 v7（公开 test 不产生伪本地测试分数）
python scripts/run_experiments.py train-test --dataset loveda --suite v7
python scripts/run_experiments.py train-test --dataset uavid --suite v7

# WWTP 单模型 shell 入口，自动检查 train/val 缺失伪几何
bash scripts/rpgv/train_rpgv_v7.sh
RPGV_V7_CONFIG=configs/rpgv/v7/fixed_grid.py bash scripts/rpgv/train_rpgv_v7.sh
```

上述训练命令需在已安装项目依赖的环境运行，或加 `docker compose run --rm wwtp` 前缀。`WWTP_DATA_ROOT`、`WWTP_PSEUDO_ROOT`、公开集 root 环境变量沿用现有约定。shell 入口尊重显式伪几何路径及 `--cfg-options` 的有效 pipeline。

旧 `all` 仍是原 13 项集合，旧 `v6`/`v6-ablations` 不变，避免旧命令意外扩大训练。v7 消融和 `v7-all` 仅用于 WWTP；公开集支持七基线和完整 v7。默认输出新目录，不覆盖已有非 resume 实验。

## 8. 验证与资源诊断

```bash
python -m unittest tests.rpgv.test_rpgv_v7 tests.rpgv.test_rpgv_v6 \
  tests.test_run_experiments tests.test_experiment_protocol -v
python tests/smoke_runner.py --config configs/rpgv/v7/rpgv_v7.py --device cpu

# 无真实数据、无预训练下载；保留正式 FP32 和优化器参数组
python tools/rpgv/profile_rpgv_v7.py --device cpu --size 64 --warmup 0 --steps 2
python tools/rpgv/profile_rpgv_v7.py --size 1024 --batch-size 2 \
  --output work_dirs/profiles/v7_train.json
python tools/rpgv/profile_rpgv_v7.py --mode infer --size 1024 --batch-size 1 \
  --output work_dirs/profiles/v7_infer.json
```

诊断文件已存在时拒绝覆盖。合成 profiler 每步执行一次完整优化器更新，**不模拟梯度累积、LR 调度、数据 IO 或验证**。输出逐组梯度、首参数更新、原始损失、结构负例计数/分差和 CUDA allocated/reserved 峰值。

WSL GPU 可沿用 v6 的 `/dev/dxg`、`/usr/lib/wsl` 只读驱动挂载。CPU 完整模块测试覆盖低 Q/平坦区域、缺失观测隔离、目标/池化梯度边界、图重编号、内容置换、同质负例屏蔽、公式对照、分割对结构梯度、所有消融反向、ignore/RGB 回退、二/多分类、滑窗、重载与统一协议。另有 CUDA FP16 支持/全无效样本梯度测试；这不改变正式训练的 FP32 配置。

### 六项默认方案验证（2026-10-12）

默认从九项收敛为六项，另三项保留为按需分析，配置及模型语义不变。

- 在既有 `wwtp-mmseg:torch2.1.2-cu121` 镜像中沿用 WSL GPU 透传，重新运行上述四组测试：**118 项全部通过，无跳过**，约 41 秒。
- 配置、损失开关、梯度和前后向测试继续覆盖六项默认、三项按需与五项历史配置，未因默认队列缩减降低测试覆盖。
- Docker 内 dry-run 核对：默认消融为 6 次训练/6 次测试，`v7-all` 为 15/15，三项按需显式选择为 3/3，五项历史显式选择为 5/5。
- 未改模型、共同训练策略或 gate，未启动真实数据训练。

### 九项方案历史验证（2026-10-12）

环境：既有 `wwtp-mmseg:torch2.1.2-cu121` 镜像，当前工作区挂载到 `/workspace`；WSL `/dev/dxg` 与只读 `/usr/lib/wsl` 驱动挂载透传 RTX3050 Laptop GPU。

- `tests.test_run_experiments`、`tests.test_experiment_protocol`、`tests.rpgv.test_rpgv_v7`、`tests.rpgv.test_rpgv_v6` 共 **118 项测试全部通过，无跳过**，约 43 秒；后者仅作既有功能回归，不是版本消融。
- 九项主消融及五项历史配置完成配置加载与 CPU 小尺寸前后向；新增断言覆盖 loss/head 启停、辅助头移除不改变公共权重下推理、无消息传递的节点独立性，以及图配对的真实非零梯度与参数更新。既有 CUDA FP16 有/无支持梯度测试通过；正式训练仍为 FP32。
- Docker 内依赖无关 dry-run 核对：`v7-ablations` 为 9 次训练/9 次测试，`v7-all` 为 18/18，五个历史名字显式选择为 5/5。
- 未启动真实数据训练、未改完整 v7 或统一训练策略；这些验证不证明任何消融的精度差值。

### 本次实际验证（2026-10-11）

环境：`wwtp-mmseg:torch2.1.2-cu121`，RTX3050 Laptop 4GB，WSL。

- 生产 CPU 合成流程：训练 → 保存 → 测试 → 恢复至第 4 步通过。
- CUDA FP16 有/无几何的前后向及全参数有限梯度检查通过。
- 默认完整模型参数：25,413,802（包含训练期对比/辅助头）。
- 1024、batch1、FP32 合成推理：1 步预热、2 步测量，平均约 **0.200 s**，CUDA allocated 峰值约 **441 MiB**。
- 1024、batch2、FP32 合成训练：1 步预热、2 步测量，平均约 **25.10 s/更新**，allocated 峰值约 **4005 MiB**、reserved 约 **5548 MiB**；RGB、基元、关系网络及结构融合均有有限梯度与实际更新。

小显存 WSL 的 allocator/共享内存行为不能等同物理显存容量；本次训练虽完成，但耗时明显，不作为吞吐达标或优于 v6 的证据。少量合成测量也不是长期性能保证。

没有运行真实数据集长训练，没有验证 v7 精度优于 v6，也没有完成独立组成关系标注集评估。正式研究还应补充外观扰动（固定伪几何与重新生成伪几何分开）、结构扰动、容量对照以及冻结表征关系探测。仅 IoU 提升、错误几何导致退化或可视化注意力，均不足以单独证明“真正理解地物组成”。
