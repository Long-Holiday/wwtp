# RPGV v5.1 实现审查与候选机制（2026-09-28）

本审查以两次训练实际保存的配置和标量日志为准：[v5 配置](../../work_dirs/rpgv_v5_1m/rpgv_v5.py)、[SegNeXt 配置](../../work_dirs/segnext_1m/segnext.py)、[v5 曲线](../../work_dirs/rpgv_v5_1m/20260928_004513/vis_data/scalars.json)、[SegNeXt 曲线](../../work_dirs/segnext_1m/20260928_042114/vis_data/scalars.json)。指标只涉及 304 张 1 m 验证图；没有借用测试集结果。两次最佳 checkpoint 均由 `binary/Foreground_IoU` 在 12k micro-iteration 选出。

## 实际协议与结构差异

两者均在相同 2435/304/305 train/val/test 划分上使用整张 1024×1024、1 m 图，RGB 光度扰动和相同的三方向翻转；batch=2、累积=4、20k micro-iteration，即约 40k 图像曝光（约 16.4 次训练集遍历）。验证是 whole inference，边界容差 1.5 px、`pixel_size_m=1.0`、拓扑小岛阈值 64 px。没有旧 2048 基准缩放或前景裁剪。v5 额外读取 depth/Q 五通道，SegNeXt 只读取 RGB，因而不是单一模块受控消融。

| 方面 | v5 | SegNeXt 1 m |
| --- | --- | --- |
| RGB encoder | ImageNet MiT-B2，4 级、层数 3/4/6/3，RGB 单次前向 | ImageNet MSCAN-S，4 级、层数 2/2/4/2，多尺度卷积空间注意力 |
| Decoder | 四级特征均投影 64 通道，C4 池化 1/2/4 网格，自顶向下至 1/4；depth/Q 在 1/8、1/4 注入有界残差 | LightHamHead 256 通道，仅用 1/8、1/16、1/32 的 C2/C3/C4，融合后在约 1/8 预测 |
| 最终输出 | 1/4 coarse head；原 RGB 与特征融合后于 1/2 输出带 SDF 的统一轮廓头，插值到整图 | 两类 softmax logits，从 decode head 插值到整图；无独立 SDF/轮廓头 |
| 目标 | 末端 masked BCE/Dice 各 0.5；coarse 0.3、同类相邻一致 0.05、unit-transition 边界差分 0.1、SDF 0.1 | 两类 CE 1.0 + Dice 1.0，decode head supervision |
| 优化 | AMP，AdamW 6e-4，MiT ×0.1；grad clip 10 | FP32，AdamW 6e-5，head ×10；grad clip 1 |
| 调度 | 均为 1000 micro-iteration warmup，20k PolyLR，seed 42，按 val IoU 存最佳权重 | 同左 |

MiT 主干与 MSCAN 主干的基础学习率都为 6e-5；v5 新 decoder/geometry 与 SegNeXt head 的学习率都为 6e-4。不同精度、裁剪阈值、预训练权重、解码器、损失和模态仍同时变化。训练 loss 数值定义不同，不能直接比较大小。当前对照无法证明 MSCAN 比 MiT 更适合本数据。

## 验证曲线说明的事实

| 按 val IoU 选出的 12k checkpoint | IoU | BF1 | Precision | Recall | HD95 m | 小预测组件 | 游离 FP 组件 | 预测孔洞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v5 | **82.2121** | **17.7149** | **90.3138** | **90.1620** | **51.5859** | 262 | 249 | 253 |
| SegNeXt | 79.9888 | 17.5349 | 88.7881 | 88.9761 | 64.5130 | **35** | **69** | **48** |

v5 的区域 IoU 高 2.2233 个百分点，Precision/Recall 均更高、HD95 低约 12.93 m；BF1 只高 0.1800 点，同时小预测组件约为 SegNeXt 的 7.5 倍、游离 FP 组件 3.6 倍、孔洞 5.3 倍。这是区域收益与局部碎片并存的可靠观察。指标针对一个 seed 和同一验证集，不能将差异单独归因于 backbone、1/2 分辨率轮廓头或几何分支。

两者在 12k 后 IoU 都未继续上升。v5 在 20k 的 IoU/BF1 是 81.7107/19.0479，SegNeXt 是 78.5142/18.7445：边界分数继续上升而 IoU 从最佳点回落。v5 8k 验证 IoU 短降至 75.3407，Precision 94.7644、Recall 78.6129，9k 恢复 80.2782；这更像固定阈值下的前景覆盖和校准波动，不能仅凭日志指认梯度爆炸。两次训练在后期训练 loss 下降而 val IoU 不再提升，提示应继续按验证最佳权重选模，并把区域与拓扑作为并列诊断。

v5 的 `loss_final_boundary` 在 1k–20k 大多为 0.085–0.099，20k 总 loss 0.1374 中该项为 0.0941；`loss_region_consistency` 仅约 2.5×10⁻⁵。该边界项对 GT 跨类相邻像素要求预测概率差接近 ±1，而最终 logits 先在 1/2 输出再插值为整图，训练后该项仍接近初期水平。数值占比**不等于梯度占比**，但它是值得单独替换并测梯度的具体目标失配候选。SegNeXt 没有这项，仍获得类似 BF1 且碎片更少；这支持测试目标变化，不能证明旧项单独造成碎片。

从结构看，v5 融合了高分辨率 C1 与 RGB 细节，SegNeXt 则从 1/8 解码，后者可能自然抑制小斑点。这只是机制假设：SegNeXt 的卷积空间归纳偏置、head、loss、FP32 优化也同时不同。直接替换 v5 的 MiT 主干将破坏现有最强区域 IoU 的可归因基础。

## v5 几何贡献与 v5.1 候选的边界

Docker CPU 只读检查 v5 最佳 12k checkpoint：`fuse8` 可学习幅度为 0.10754、输出投影权重 L2 为 5.09565；`fuse4` 幅度为 0.10598、L2 为 4.81218。幅度从初值 0.1 略升，权重非零，但这**不能证明**几何改善验证集，也不能表示实际残差或像素贡献。需要同一 checkpoint 上 full、Q0=0、depth 打乱、RGB capacity control 的完整验证诊断，再决定 v5.1 默认是否启用几何。

目前候选 v5.1 保留 MiT、几何门控和 unified contour 接口，只在 decoder 的 1/8、1/4 增加少量 5×5 depthwise 与 7/11 strip depthwise 空间残差；最后 1×1 投影零初始化且无后置归一化，初始严格等价于加载的 v5 权重。限制残差为 `0.5*tanh(·)`，防止局部更新无限放大。此设计测试“轻量卷积空间先验能否减少碎片”，不代表移植 MSCAN 或 LightHamHead。零初始化使第一个更新只训练输出投影，内部空间核从随后更新开始收到梯度；是否足以产生效果需记录梯度和验证曲线。1/4 特征仍有 256×256 空间大小，strip 分支可能增加显存和时间，正式预算前需在 Docker 中按 batch=2 做 AMP profile。

损失候选将旧 unit-transition 差分与近乎零的 region 项替换为 GT 边界半径 3 个输入像素内的前景/背景均衡 BCE；全图 final BCE/Dice、coarse 与 SDF 保留。边界由**图内且两侧有效**的相邻标签差生成，ignore 邻域排除；全背景、全前景、all-ignore 图返回有限零。边界带只作用在 GT 边缘附近，不能直接惩罚远离目标的游离假阳性；全图 BCE/Dice 仍承担此任务。若 BF1 改善而小岛不降，应另查高频 decoder 响应与概率阈值，而不能把该目标当作拓扑正则化。

实现候选保留 `RPGVNetV5` 与旧 decoder 的 state_dict 键，新增的 `decoder.spatial8/spatial4` 是唯一缺失键。`spatial_refinement=False` 不构建新块；`boundary_mode='legacy'` 恢复 v5 原有 region/final-boundary 权重，支持结构单因素对照。若用 v5 12k warmstart，则应同时比较继续训练的原 v5、仅空间块、仅边界带和两者叠加，保持相同 checkpoint、优化器重启方式、数据顺序、预算与验证选择规则。若只做叠加实验，无法区分效果来源。

Docker CPU 小模型检查已确认：加载相同 v5 参数后，v5.1 的初始最终 logits 与 v5 最大差为 0；旧键无意外缺失，新增缺失键仅空间块；空图、全前景、all-ignore 的边界带损失为 0，正常边界损失有限。真实 `model.loss` 的 band 模式只返回 final/coarse/SDF/boundary-band，legacy 模式恢复 final/coarse/SDF/region/final-boundary；两种模式损失有限且零初始化空间块的输出投影首步梯度非零。该检查没有验证真实 1024 AMP 显存、更新后的收敛或 1 m 验证集收益。
