# RPGV v5 实现与 1 m 数据审计（2026-09-28）

本文只核查实现、数据接口和验证边界；历史模型效果见 [实验审计](rpgv_v5_experiment_audit.md)，完整 v5 设计与训练协议另见主设计文档。v5 尚无收敛或验证集结果，不能从前向检查推断精度。

## 转换数据和旧配置

`wwtp_semantic_dataset_1m/dataset_stats.json` 记录 1024×1024、1 m/px，train/val/test 分别为 2435/304/305 张。逐划分比对了影像 PNG、标注 PNG、伪几何 NPZ 的**全部文件名 stem**，三者完全匹配。Docker 只读抽查各划分两个样本：影像为 1024×1024 RGB，NPZ 的 `depth` 和 `reliability` 均为 1024×1024，样本世界文件像元间距为横向 1、纵向 −1 m。此抽查没有逐一解压验证全部 3044 个 NPZ 的内容。

既有 `configs/_base_/datasets/wwtp_pseudo_1024x1024.py` 和 RGB 数据配置默认读取原 `wwtp_semantic_dataset`。`docker-compose.yml` 也将 `WWTP_DATA_ROOT` 固定为原数据目录。若仅切换环境变量到 1 m 数据却继承旧训练流水线，`RandomResize(scale=(2048,2048), ratio_range=(0.5,1.5))` 会让 1024 图像通常先放大后裁成 1024，改变物理尺度与有效视野；`BinaryBoundaryMetric(pixel_size_m=0.5)` 会使 HD/HD95 的米制数值折半。旧 3 px 边界容差也不能直接视为新数据的同一物理容差。

`configs/v5/rpgv_v5.py` 独立使用 `WWTP_V5_DATA_ROOT`（默认 `wwtp_semantic_dataset_1m`），全幅 1024 输入，空间增强为同步翻转；不再继承旧缩放、旋转和前景裁剪。验证与测试使用 `pixel_size_m=1.0`、`boundary_tolerance=1.5`，后者尽量保持旧 3×0.5=1.5 m 的容差。栅格化后离散边界不同，因此新旧 BF1 仍不可直接逐数比较。v5 使用整图推理，避免 1024 图再走旧 1024 滑窗路径。

## 架构接口核查

`RPGVNetV5` 只构建一次 MiT RGB encoder 并调用其原生前向；几何不进入编码器。`ContextTopDownDecoder` 将四级 RGB 特征投影为 64 通道，C4 的 1/2/4 网格池化上下文汇入 C4，再自顶向下融合到 1/4。`CompactGeometryEncoder` 在任何空间卷积前计算 `depth * Q`，并与 Q 一起编码为 1/4、1/8 特征。`BoundedGeometryFusion` 只在 decoder 的 1/8 与 1/4 注入残差；最后投影没有归一化或零初始化，因此初始最终分割梯度能够进入几何分支。

融合项为 `0.5 * sigmoid(scale_logit) * Q_down * tanh(project([RGB, geometry, RGB*geometry]))`。scale 初值 0.1，逐元素注入幅度的绝对值严格小于 0.5（有限输入、Q∈[0,1]）；此界限作用于 decoder 特征，不能直接解释为最终 logit 上限。Q 整图为零时，depth 模式的 decoder 输出和同权重 RGB 路径一致；任意 Q=0 像素的深度值在空间混合前被遮蔽，因此该像素的任意有限深度变化不会传播到邻域。3 通道 depth 模式自动补零 D/Q，提供同权重回退。RGB capacity control 则用原 RGB 灰度和常数一代替 D/Q，3/5 通道一致；其 Q0 输入不触发回退，这是容量对照的定义。

两种有额外分支的训练都使用相同概率的样本级分支 dropout。depth 模式沿用输入 Q 整样本清零；RGB control 对自身构造的常数一门控整样本清零，始终不读取原 D/Q。`use_geometry=False` 不构建几何 encoder 和融合模块；`use_context=False` 不构建上下文投影；`use_contour=False` 禁用轮廓修正，但默认仍监督 SDF。轮廓头在输入 1/2 分辨率预测 SDF，`contour_truncation=5` 对应约 10 m 的输入尺度距离。

## 继承路径及依赖风险

v5 继承 v4 以复用 `loss` 和 3/5 通道 `_split_inputs`，但直接调用 `BaseSegmentor.__init__`，不会构建 v1/v2/v4 已删除的 rectifier、RGR、DFGV、独立几何头或编码器内融合。已逐条核对当前实际调用链：

| 入口 | 继承实现所需状态 | v5 提供或覆写 |
| --- | --- | --- |
| `loss`（v4） | `training_stage`, `geometry_dropout_prob`, `use_global_context`, `_run_network`, `_targets`, `_shape_losses` | 均提供；`use_global_context=False` 避免旧 thumbnail 分支 |
| `_shape_losses`（v2） | `coarse_loss_weight`, `region_loss_weight`, `final_boundary_loss_weight`, `loss_weights['boundary'/'sdf']`, `contour_truncation`, 输出中的 `coarse_logits/final_logits/sdf` | 均提供；`boundary=0` 避免读取旧 RGB boundary head |
| 训练调度（v1） | `train()` 的特殊冻结分支仅在 `training_stage='geometry'` 运行；`_apply_geometry_dropout` 依赖 joint 与 Q 通道 | 固定 `joint`；RGB control 覆写 dropout，depth 沿用旧方法 |
| 推理（v1） | `encode_decode` 调 `_run_network` 并读取 `seg_logits`；whole/slide 读取 `test_cfg`，global token 仅在 `use_global_context=True` 构造 | `_run_network` 和 `seg_logits` 已覆写；默认 `test_cfg.mode='whole'` |
| `parse_losses`（v1） | 仅检查 loss 有限性，并在错误中读取 `training_stage` | 已提供 `joint` |

`RPGVNetV2.loss` 的 `learnable_loss_weights` 属性在方法解析中被 v4 的 `loss` 覆盖，当前不会访问。旧 `_geometry_losses`、`_decode_and_refine`、`_encode_fused` 未被 v5 的 loss/推理路径调用。继承层次较深，若未来修改这些父类入口，应重新运行 v5 回归并检查依赖，不能只据类定义推断安全。

## 已验证范围与后续边界

最初用现有 Docker 镜像完成 CPU 小尺寸前向与梯度检查：depth 模式全 Q0、3 通道对显式 RGB 回退的最终 logit 最大差均为 0；最终 logit 对 1/8 融合、1/4 融合和几何 stem 均有非零梯度。主代理随后完成 9 项 Docker CPU 回归，以及真实 1024/1 m 图像 batch 2 的 6 次 AMP 前向、loss、反向和 AdamW 更新；后者记录在 [profile JSON](rpgv_v5_profile_v5_b2.json)，峰值已分配显存约 10.5 GiB，warmup 后计算步均值约 0.495 秒。profile 使用固定 batch、随机初始化，不包含数据读取、验证、梯度累积或收敛结论；也不支持直接将 batch 4 视为已验证。

上述 9 项回归和 profile 均在 RGB control dropout 公平性修复之前完成。修复只改变该对照的训练态门控；需要重新跑受影响的 control 回归，并验证 dropout 概率为 1 时训练态输出与显式 RGB 分支一致。正式训练还需监测最佳 val checkpoint、同权重 RGB 回退与几何增益，最后才在锁定模型后使用 test 集。
