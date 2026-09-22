# RPGV-Net 工作流程可视化

脚本：[tools/visualize_rpgv_workflow.py](../tools/visualize_rpgv_workflow.py)。它读取一张验证集或测试集图像、对应的离线伪几何 NPZ、GT，以及一个**已写完**的联合阶段 checkpoint。Depth Anything 不会再次运行。所有源文件仅被读取；结果只写到指定的全新目录。脚本拒绝把输出目录放在数据目录、伪几何目录或 checkpoint 所在目录中，也拒绝覆盖已有结果。

模型训练结束后，可选一张图和一块感兴趣区域执行：

```bash
docker compose run --rm wwtp python tools/visualize_rpgv_workflow.py \
  configs/experiments/rpgv_stage3_joint.py \
  work_dirs/rpgv_staged/stage3_joint/finished-checkpoint.pth \
  --split val --image-id wwtp_000149 \
  --roi 0 0 1024 1024 \
  --device cuda:0 \
  --output-dir analysis/rpgv_wwtp_000149_case1 \
  --save-maps
```

把 `finished-checkpoint.pth` 换成训练结束后保存的 checkpoint 文件名。`--roi` 是原图坐标中的 `X Y W H`，宽高需为 32 的倍数；省略时取配置中的 1024×1024 居中窗口。若要在论文中突出窄结构或边界，可再加 `--display-zoom X Y W H`，坐标相对于 ROI；它只裁切论文图的展示范围，不改变模型推理窗口或诊断图。`--device cpu` 可以避免占用训练 GPU。如果数据目录与配置不同，可分别指定 `--data-root` 和 `--pseudo-root`。不同案例使用不同的新输出目录。

输出包括诊断图 `workflow.png`、论文主图 `paper_workflow.pdf` 与 600 dpi 的 `paper_workflow.png`，以及 `metadata.json`；加 `--save-maps` 时还会保存 `maps.npz`。论文主图是 7.2 英寸宽的双栏排版，PDF 中保留矢量文字。它用固定 4×4 面板对应四个计算阶段，并提供统一色标。图中包含：

1. BGR 输入转成可视 RGB、离线深度 D0 与离线可靠度 Q0；
2. RGR 的修正深度、真实校正量 `Dhat - D0@1/4`、学习到的可靠度衰减 Qlearn 和任务可靠度 Qd；
3. RGB 与几何辅助分割概率；
4. 高频边界与低频区域验证权重的**通道均值**，以及验证后的候选残差强度；
5. 两处融合后的实际特征变化 `mean(abs(F_fused - F_RGB))`，以及启用的边界/细节精修门控；
6. 同一 joint checkpoint 的 RGB-only 解码输出、最终概率、二值掩膜、GT 和误差图。

此模型没有单个“Gate/Fusion weight”。DFGV 在 Haar 高频边界和 1/16 区域两个位置各产生逐通道权重；随后按 Qd 连续缩放残差。图中的验证权重是逐通道均值，特征响应是绝对值均值，不能把它们解释为最终像素的类别概率。关闭对应消融组件后，诊断图会省略该图，论文图会标记 `Disabled in config`。

论文图使用 0–1 的固定色标显示相对深度、可靠度/权重和前景概率；RGR 校正使用以 0 为中心的双向色标。边界与区域特征响应**共用同一个色标**，默认上界为两者合并后的 99 分位数；在跨样本或跨消融图并排比较时，建议对所有命令使用相同的 `--response-limit` 和 `--correction-limit`，并在图注注明数值。最终预测上的白色轮廓是 GT 边界。绿色、橙色和蓝色分别表示 TP、FP 和 FN。

可直接修改以下英文图注中的样本信息后用于论文：

若使用 `--display-zoom`，需在图注中补充展示区域是推理窗口内的局部放大图；`metadata.json` 记录了推理 ROI 和放大范围。

> Qualitative visualization of RPGV-Net on a WWTP validation crop. The offline pseudo-depth $D_0$ and reliability $Q_0$ are refined by RGR to produce $\hat D$ and task reliability $Q_d$. The RGB and geometry auxiliary predictions provide branch-level evidence. The channel-averaged Haar boundary and region validation weights ($\bar W_h$, $\bar W_l$) show the spatial pattern of geometry validation; the contribution maps show the mean absolute change to the fused RGB features at strides 4 and 16. The final foreground probability is overlaid with the ground-truth boundary. Error colors denote true positives (green), false positives (orange), and false negatives (blue). Continuous maps share the labeled color scales shown below the panels.

图中的最终预测是**一个局部窗口**的前向结果，并使用整张原图的全局 token。它用于解释该窗口内的计算链路；全图正式评估使用带 Hann 重叠融合的滑窗推理，因此局部结果可能与拼接后的全图结果略有不同。`metadata.json` 里的 IoU 也仅针对这个窗口，不能作为测试集指标。GT 中值为 255 的像素被忽略。
