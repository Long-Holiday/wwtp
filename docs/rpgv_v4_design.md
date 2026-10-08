# RPGV v4 结构参考

在四级编码器内融合 RGB 与轻量几何特征，浅层 Haar 频域验证；轻量解码与轮廓精修。全部参数单阶段训练；几何、频率和全局上下文开关完整保留。

入口：configs/rpgv/v4/；统一 v5 预算用 configs/rpgv/v5/controls/v4_1m.py。

所有版本现在默认使用 1024×1024、1m WWTP 整图与一致的精确翻转增强。512 thumbnail 是部分旧版本的全局上下文输入，不是训练图像裁剪尺度。模型结构、损失开关和权重迁移保留；阶段预算仍按各版本配置。当前使用方式以 [README](../README.md) 和实际配置为准。

详细原设计、历史实验日志与审计保存在 `experiment-logs/2026-10-07` 分支，可用 `git show experiment-logs/2026-10-07:docs/rpgv_v4_design.md` 阅读。主分支不保留历史结果报告。
