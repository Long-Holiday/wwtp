# RPGV v2 结构参考

复用 RGB 与几何分支，采用加法多尺度解码、SDF 轮廓修正和结构监督。单阶段联合训练与原阶段配置均保留。固定损失、递进式和单因素删除消融位于 configs/rpgv/v2/ablations/。

入口：configs/rpgv/v2/rpgv_v2_joint.py；统一 v5 预算用 configs/rpgv/v5/controls/v2_1m.py。

所有版本现在默认使用 1024×1024、1m WWTP 整图与一致的精确翻转增强。512 thumbnail 是部分旧版本的全局上下文输入，不是训练图像裁剪尺度。模型结构、损失开关和权重迁移保留；阶段预算仍按各版本配置。当前使用方式以 [README](../README.md) 和实际配置为准。

详细原设计、历史实验日志与审计保存在 `experiment-logs/2026-10-07` 分支，可用 `git show experiment-logs/2026-10-07:docs/rpgv_v2_design.md` 阅读。主分支不保留历史结果报告。
