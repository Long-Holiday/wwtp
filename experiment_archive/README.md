# 实验日志归档

此分支保存重构前的完整代码、实验配置、训练日志、验证/测试 JSON、审计与实验报告。master 不再追踪历史运行产物。

- `EXPERIMENT_REPORT.md`：已有实验总报告，原样保留。
- `work_dirs/`：906 个已追踪训练/测试配置、日志和结果文件。
- `docs/analysis/`：历史模型审计和诊断。
- `analysis/`：历史可视化及中间数据。
- `experiment_archive/index.json`：精确文件索引与来源 commit。

归档不包含原本被忽略的 checkpoint、数据集和外部文件。可以用 `git show experiment-logs/2026-10-07:路径` 阅读单个文件，无需切换训练工作区。
