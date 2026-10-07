# RPGV v1 结构参考

MiT-B2 RGB 编码器、可靠度引导几何校正 RGR、双频几何验证 DFGV、多尺度可靠度残差融合、边界和细节精修。RGB／几何／联合三个阶段仍由 training_stage 控制。

入口：configs/experiments/rpgv_stage1_rgb.py、rpgv_stage2_geometry.py、rpgv_stage3_joint.py；直接联合训练用 rpgv_net.py。

所有版本现在默认使用 1024×1024、1m WWTP 整图与一致的精确翻转增强。512 thumbnail 是部分旧版本的全局上下文输入，不是训练图像裁剪尺度。模型结构、损失开关和权重迁移保留；阶段预算仍按各版本配置。当前使用方式以 [README](../README.md) 和实际配置为准。

详细原设计、历史实验日志与审计保存在 `experiment-logs/2026-10-07` 分支，可用 `git show experiment-logs/2026-10-07:docs/model_design.md` 阅读。主分支不保留历史结果报告。
