# RPGV v3 结构参考

保留经过初始化验证的冻结 RGB 锚点，用几何条件适配器学习误差修正。提供 RGB 修正对照、误差聚焦与 RGB 保护消融。冻结锚点、初始化迁移和回退测试保持原实现。

入口：configs/v3/rpgv_v3_adapter.py；tools/initialize_rpgv_v3.py。

所有版本现在默认使用 1024×1024、1m WWTP 整图与一致的精确翻转增强。512 thumbnail 是部分旧版本的全局上下文输入，不是训练图像裁剪尺度。模型结构、损失开关和权重迁移保留；阶段预算仍按各版本配置。当前使用方式以 [README](../README.md) 和实际配置为准。

详细原设计、历史实验日志与审计保存在 `experiment-logs/2026-10-07` 分支，可用 `git show experiment-logs/2026-10-07:docs/rpgv_v3_design.md` 阅读。主分支不保留历史结果报告。
