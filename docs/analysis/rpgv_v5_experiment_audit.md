# RPGV v5 设计前的实验审计（2026-09-28）

此表从本地训练 `vis_data/scalars.json`、阶段 `completed.json`、独立评估 JSON 与 checkpoint 加载日志重建。可复跑的完整数值和每条来源在 [审计 JSON](rpgv_v5_experiment_audit.json)，提取程序在 [`tools/audit_rpgv_v5_results.py`](../../tools/audit_rpgv_v5_results.py)。表中 IoU、BF1、Precision、Recall 的单位是百分比；差值是百分点。所有既有 WWTP 正式实验使用原始 2048 像素场景和对应 0.5 m 指标设置；下面的 1024×1024 是训练/滑窗裁剪尺寸，**不是**用户新指定的 1024 像素、1 m 转换数据。快速筛选另用 512 裁剪。本审计没有训练、推理或修改既有结果。

## 有效的主实验结果

| 运行 | 训练预算/已见验证 | 按 val IoU 选择的 checkpoint | Val IoU / BF1 | 独立 Test IoU / BF1 | 证据与解释 |
| --- | --- | --- | ---: | ---: | --- |
| v1 RGB 阶段一 | 20k / 20k | 16k | 71.7332 / 13.5720 | 64.9525 / 9.7711 | [曲线](../../work_dirs/rpgv_staged/stage1_rgb/20260919_103639/vis_data/scalars.json)、[test](../../work_dirs/rpgv_staged/stage1_rgb/test_eval/test_results.json)；该 test JSON 缺少身份元数据，需谨慎引用 |
| v1 几何阶段二 | 12k / 12k | 10k | 35.9012 / 2.1248 | — | [曲线](../../work_dirs/rpgv_staged/stage2_geometry/20260919_135612/vis_data/scalars.json)；是几何辅助头独立分割精度，不是最终输出 |
| v1 联合阶段三 | 40k / 40k | 40k | 76.8698 / 16.5916 | 77.1717 / 16.6801 | [曲线](../../work_dirs/rpgv_staged/stage3_joint/20260919_153804/vis_data/scalars.json)、[test](../../work_dirs/rpgv_staged/stage3_joint/test_eval/20260920_004330/20260920_004330.json)；测试日志确认载入最佳 40k checkpoint |
| v1 阶段三重启 | 另训 8k / 8k | 重启第 6k | 77.3610 / 16.8055 | 77.7362 / 16.9941 | [曲线](../../work_dirs/rpgv_stage3_restart_30k/20260921_110310/vis_data/scalars.json)、[test](../../work_dirs/rpgv_stage3_restart_30k/test_eval/20260921_130631.json)；[测试日志](../../work_dirs/rpgv_stage3_restart_30k/test_eval/20260921_125546/20260921_125546.log)确认载入最佳 6k checkpoint |
| v2 RGB 阶段一 | 40k / 40k | 36k | 75.1428 / 14.4175 | — | [completed](../../work_dirs/rpgv_v2_paper/full/seed_42/full/stage1_rgb/completed.json) |
| v2 几何阶段二 | 20k / 20k | 20k | 40.1800 / 2.4491 | — | [completed](../../work_dirs/rpgv_v2_paper/full/seed_42/full/stage2_geometry/completed.json) |
| v2 联合阶段三 Full | 40k / 40k | 38k | **78.4041 / 16.9588** | **78.5600 / 17.1286** | [completed](../../work_dirs/rpgv_v2_paper/full/seed_42/full/stage3_joint/completed.json)、[test 指标](../../work_dirs/rpgv_v2_paper/full/seed_42/full/evaluation_test/metrics.json)、[test 身份](../../work_dirs/rpgv_v2_paper/full/seed_42/full/evaluation_test/identity.json)；无后处理 |
| v2 单阶段 joint_fixed | 100k 预期 / 74k 已见 | 64k | 78.4425 / 16.9958 | — | [曲线](../../work_dirs/rpgv_v2_joint_fixed/20260925_083005/vis_data/scalars.json)；64k 后至 74k 降至 76.7765，不能当作已完成 100k |
| v4 单阶段 | 40k / 40k | 26k | 74.4435 / 14.8618 | 72.7716 / 14.5317 | [曲线](../../work_dirs/rpgv_v4/20260924_032119/vis_data/scalars.json)、[test](../../work_dirs/rpgv_v4/test_eval/20260924_091541.json)；[日志](../../work_dirs/rpgv_v4/test_eval/20260924_090635/20260924_090635.log)确认 26k checkpoint |

v3 在工作区只有 [初始化记录](../../work_dirs/rpgv_v3_initialization/init.json)和 init checkpoint。用户确认未训练 v3，而是直接训练 v4，因此没有 v3 性能数值。旧 [v4 设计文档](../rpgv_v4_design.md)称“尚未完整训练”，现已过时；以上原始训练和测试文件优先。

v2 Full 与 v4 都有 40k 的单次阶段三/单阶段训练上限，但初始化、结构、目标、数据增强与优化路径不同。v4 低 **3.9606 pp val**、**5.7884 pp test** 是已观察到的整体结果，不能单独归因于其 RGB–几何交互结构。v1 重启 test 与 v2 Full test 的差为 **0.8238 pp**，但 v1 RGB 仅训 20k、联合阶段经历重启，亦非等预算受控结构比较。

## v2 正式和接近正式的消融

| 变体 | 预算 / 最后验证 | 最佳 val 迭代 | Val IoU / BF1 | 对 Full val IoU | 结论等级 |
| --- | ---: | ---: | ---: | ---: | --- |
| Full | 40k / 40k | 38k | 78.4041 / 16.9588 | 0 | 正式，seed 42 |
| 去 RGR | 40k / 40k | 38k | 78.3314 / 16.6102 | −0.0728 | 相同三阶段预算，单种子；[阶段二](../../work_dirs/rpgv_v2_paper_reuse/paper_reuse/seed_42/no_rgr/stage2_geometry/completed.json) 从 40.1800 降到 21.8563，而最终差仅 0.0728 pp；[阶段三](../../work_dirs/rpgv_v2_paper_reuse/paper_reuse/seed_42/no_rgr/stage3_joint/completed.json) |
| 去频率验证 | 40k / 16k 早停 | 6k | 76.2019 / 15.4196 | −2.2023 | [completed](../../work_dirs/rpgv_v2_paper_reuse/paper_reuse/seed_42/no_frequency_validation/stage3_joint/completed.json)；复用 Full 阶段二初始化，仅阶段三消融且早停，表明此训练设置有风险，不证明该模块稳定贡献 2.20 pp |
| 无权融合 | 40k / 日志至约 5.8k | 2k（已有验证） | 72.8206 / 未作为终局 | 不可比 | [早期曲线](../../work_dirs/rpgv_v2_paper_reuse/paper_reuse/seed_42/unweighted_fusion/stage3_joint/20260924_013105/vis_data/scalars.json)；后有重启日志，无 completed 和最终评估 |
| 从头单阶段 progressive base | 100k / 30k 早停 | 20k | 73.8619 / 13.8398 | 不可比 | [completed](../../work_dirs/rpgv_v2_single_stage_fixed/single_stage/seed_42/progressive_base/joint/completed.json) |
| 从头单阶段 progressive RGR | 100k / 60k 早停 | 50k | 78.2713 / 17.0547 | 不可比 | [completed](../../work_dirs/rpgv_v2_single_stage_fixed/single_stage/seed_42/progressive_rgr/joint/completed.json)；距三阶段 Full 的 val IoU 仅 0.1328 pp，但训练路径/预算不同 |
| 从头单阶段 progressive DFGV | 100k / 曲线至 28k | 24k（暂时） | 74.5924 / 14.5559 | 不可比 | [曲线](../../work_dirs/rpgv_v2_single_stage_fixed/single_stage/seed_42/progressive_dfgv/joint/20260927_041955/vis_data/scalars.json)；没有完成标记 |

`work_dirs/rpgv_v2_paper/full/seed_42/` 中其余变体仅有计划配置或极早阶段记录；`work_dirs/rpgv_v2_ablations/` 没有完成的训练曲线。汇总文件中 `planned` 不是性能为零。所有正式消融仅一个随机种子，0.0728 pp 这样的差值不能作显著性判断。去 RGR 的阶段二独立任务大降，而最终基本不变，是支持**削减几何独立分割任务和 RGR 成本**的最直接受控证据；这仍不等于几何无效。

## v1 短阶段消融与测试

下列变体有独立测试评估，但各自阶段三只验证至 4k–8k，均与 Full 40k 不等预算；不能把差值写成模块的完整训练贡献。测试日志载入各自按验证选择的 checkpoint。来源均在 `work_dirs/rpgv_ablations/<变体>/stage3_joint/` 的时间戳 `vis_data/scalars.json` 与 `test_eval/*.json`，审计 JSON 列出精确路径。

| 变体 | 最后验证 / 最佳迭代 | 最佳 Val IoU | Test IoU / BF1 |
| --- | ---: | ---: | ---: |
| 去 RGR | 7k / 7k | 74.4780 | 73.6336 / 15.3358 |
| 去深度矫正 | 5k / 5k | 73.1608 | 72.4532 / 14.1253 |
| 去全局上下文 | 8k / 5k | 71.2943 | 71.8299 / 14.1961 |
| 去细节精修 | 4k / 1k | 71.6320 | 70.7290 / 13.1962 |
| 去频率验证 | 4k / 1k | 71.5922 | 70.7289 / 13.1470 |
| 无权融合 | 4k / 1k | 71.4790 | 70.6702 / 13.1936 |
| 去几何融合 | 4k / 1k | 71.6356 | 70.4728 / 13.2163 |

## v2 快速筛选：只用于排优先级

两轮均从相同 v2 joint_fixed **64k checkpoint** 开始。零训练开关评估用了固定 mini-val（64 / 256 张原 2048 场景），短微调分别只训 **1000 / 1500 iter**、512 裁剪、256 thumbnail、batch 8×累积 2，随后在各自 mini-val 上评估。参数来源是两轮 [finetune identity](../../work_dirs/rpgv_v2_quick_screen/finetune/full/identity.json)、[expanded identity](../../work_dirs/rpgv_v2_quick_screen_expanded/finetune/full/identity.json)；各行原始 `metrics.json` 路径在审计 JSON 中。既有 [快速筛选说明](../rpgv_v2_quick_screen.md)写的默认 3000 iter 与实际本轮记录不同。

| 变体 | 64 张零训练 ΔIoU | 64 张短微调 ΔIoU | 256 张零训练 ΔIoU | 256 张短微调 ΔIoU |
| --- | ---: | ---: | ---: | ---: |
| Full IoU | 80.5607 | 71.1093 | 78.2324 | 70.8212 |
| progressive base | −0.3399 | −0.0793 | −0.3724 | −0.1923 |
| progressive RGR | −0.2005 | **+0.2793** | −0.3053 | **−0.5865** |
| progressive DFGV | −0.3102 | −0.2947 | −0.3112 | −0.7750 |
| progressive weighted | −0.1909 | −0.7626 | −0.1739 | −0.8655 |
| progressive contour | 0.0000 | +0.0888 | 0.0000 | +0.0527 |

RGR 的短微调名次在扩大样本后翻转，说明 64 张排名不稳。`progressive_contour` 在零训练阶段与 Full **逐数相同**，因为两者前向配置相同；Full 相比它另有 `region_loss_weight=0.1` 和 `final_boundary_loss_weight=0.1`。两者短微调差 +0.0888 / +0.0527 pp 是这两项**训练监督**的差，不能称为轮廓模块收益。真正的轮廓开关是 [`progressive_contour.py`](../../configs/ablations_v2/progressive_contour.py) 相对 [`progressive_weighted.py`](../../configs/ablations_v2/progressive_weighted.py) 的 `use_contour=True`：64 张短微调 **+0.8514 pp**，256 张 **+0.9182 pp**。这给轮廓修正保留优先级，但仍需完整训练与多种子确认。DFGV 与 weighted 两轮均低于 Full，足以降低其后续优先级，却不足以宣称完整训练时必然劣于 Full。

## 外部架构基线与边界

本地 2048 场景 Test 中，最好的一组通用模型是扩训 DeepLabV3+ **69.1951 IoU / 11.4165 BF1**、扩训 SegNeXt **65.9790 / 11.8045**；原 DeepLabV3+ 为 **65.2712 / 9.5685**。其余 13 个原版/扩训基线全部列在审计 JSON 的 `baselines`，来源是各自 `work_dirs/*/test_results/<时间戳>/<时间戳>.json`。这些不是与 v2 同训练阶段、迭代数、裁剪面积、模态输入和参数量的受控消融；它们说明任务难度与性能量级，不能直接为某个 RPGV 模块归因。CBR-Net 两次测试分别为 22.6429、27.0496 IoU，HD-Net 为 34.9388；测试文件亦列在 JSON。它们的实现协议更不一致，故不用于 v5 结构选择。

## 对 v5 决策的证据强弱

1. **高可信的运行事实：** v2 Full 是已完成模型中的最强已核验 WWTP 结果；v4 全程训练明显回退。保留 v2 的强 RGB 表征与稳定的最终分割路径作为 v5 参照，并用相同 1 m 转换数据重新训练/评估。
2. **中等可信的机制线索：** v2 去 RGR 的几何辅助头降 18.3237 pp，而最终只降 0.0728 pp；给 RGR、几何独立重建和昂贵级联减少复杂度是合理假设。需要同 checkpoint 的几何开关、RGB-only 容量对照、跨种子确认真实几何增益。
3. **中等可信的优化风险：** v2 joint_fixed 从最佳 64k 的 78.4425 降至 74k 的 76.7765；v4 最佳 26k 后至 40k 降到 73.6021。架构评估必须保存 val 最佳权重、独立最终 test，并监测 RGB-only anchor 与融合增量，避免晚期退化掩盖收益。
4. **弱证据：** 短阶段 v1 消融、v2 早停去频率模块、两轮快速筛选不能独立证明模块贡献。新数据从 2048 转为 1024、0.5 m 转为 1 m 后，边界宽度、最小岛面积、几何细节和滑窗协议均改变；旧 HD95 米制结果和这些小增益不应直接外推。

首次 v5 实验建议在转换后的 1024 像素、1 m 数据上锁定 train/val/test 划分和评估尺度，同时运行同预算 RGB anchor、RGB+几何修正、无几何输入容量对照及无可靠度门控对照；先用验证集选择 checkpoint，只对选定模型运行一次独立 test。这样才能判断几何是否提供超出 RGB 再训练的真实增益。
