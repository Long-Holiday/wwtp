# Potsdam 与 LoveDA 语义分割实验

这套实验使用仓库现有的 MMSegmentation 1.2.2 容器、SegFormer MiT-B2 模型和独立的 `data/remote_sensing/` 数据目录。原有 `wwtp_semantic_dataset/`、配置、权重及训练输出不会被读取或覆盖。原始 ZIP 始终保留；预处理只写到新建的 `prepared/`。所有输出目录已加入 `.gitignore`。

## 数据与协议

| 数据集 | 官方来源 | 训练 / 模型选择 / 最终评估 | 类别和指标 |
| --- | --- | --- | --- |
| LoveDA | [Zenodo 5706578](https://zenodo.org/records/5706578) | 官方 Train 2522 / Val 1669 / Test 1796；本地评估使用 Val，Test 无公开标签 | 七类；原始 0 为 no-data，训练与评估忽略；mIoU、mFscore |
| ISPRS Potsdam | [ISPRS 官方下载](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx) | 历史 24 张训练瓦片中取 20 张训练、4 张内部验证；最终训练用全部 24 张，独立评估历史 14 张测试瓦片 | RGB、六类训练；报告六类 mIoU，以及排除 clutter 的五类 mIoU/mF1/OA |

Potsdam 原始瓦片为 6000×6000。预处理生成无重叠的 512×512 PNG；边缘补齐区域标签设为 0，MMSeg 加载后转为 ignore index 255。按完整瓦片分组划分，没有来自同一瓦片的训练与验证补丁交叉。内部验证用 `2_10, 4_12, 6_7, 7_9` 四张。最终训练配置包含这四张，但不使用历史 14 张测试标签选权重。`manifest.json` 记录划分。

官方 `5_Labels_all.zip` 的 `4_12` 文件是标注叠加影像，不能当作纯色分类图；这一问题也见 [TorchGeo 记录](https://github.com/torchgeo/torchgeo/issues/1727)。此外，`6_7` 在两套标注包中的校验值不同。预处理对 24 张历史训练瓦片统一使用 `5_Labels_for_participants.zip`，对 14 张独立测试瓦片使用 2018 年公开的完整参考，并严格拒绝其他异常标签颜色。若预处理中断，可在检查 `.building` 目录后加 `--resume` 继续已完整生成的瓦片。

Potsdam `full/*` 使用完整参考；若官方 ZIP 内含 `5_Labels_all_noBoundary.zip`，预处理还生成 `test_eroded` 标注，并可用独立配置评价。`eroded3/*` 则是从每张 512 补丁近似计算的 3 像素腐蚀参考；补丁外缘也会被腐蚀，不能当作官方 eroded-reference 数值。与使用完整参考的论文比较时采用 `full/mIoU5` 和 `full/mF15`。[ISPRS 官方说明](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx)区分完整参考和边界腐蚀参考。

## 下载与准备

官方原始压缩包合计约 22.9 GB，预处理和临时解包还需额外磁盘空间。脚本校验 LoveDA 的官方 MD5 和 Potsdam ZIP 的官方大小；已存在且不匹配的文件会报错，不会覆盖。下载中断后再次执行可从 `.part` 续传。ISPRS 的下载密码来自其公开网页上的图片。

```bash
python3 tools/remote_sensing/download.py --dataset all
docker compose run --rm --user "$(id -u):$(id -g)" wwtp python tools/remote_sensing/prepare.py loveda
docker compose run --rm --user "$(id -u):$(id -g)" wwtp python tools/remote_sensing/prepare.py potsdam
```

如只下载一个数据集，把 `all` 改为 `loveda` 或 `potsdam`。准备脚本拒绝写入已经存在的目标目录；若中途失败，保留 `.building` 目录供检查。

## 训练、评估与论文对照

LoveDA 使用官方 Train 训练和 Val 选权重，测试脚本再次在 Val 上计算指标：

```bash
docker compose run --rm wwtp python tools/remote_sensing/run.py train configs/remote_sensing/loveda_segformer.py
docker compose run --rm wwtp python tools/remote_sensing/run.py eval configs/remote_sensing/loveda_segformer.py --checkpoint work_dirs/remote_sensing/loveda_segformer/best_mIoU_iter_XXXX.pth
python3 tools/remote_sensing/compare.py loveda work_dirs/remote_sensing/loveda_segformer_eval/metrics.json
```

Potsdam 先用内部验证集调试，再固定设置，用全部 24 张训练瓦片重新训练并只在 14 张历史测试瓦片上评估一次：

```bash
docker compose run --rm wwtp python tools/remote_sensing/run.py train configs/remote_sensing/potsdam_segformer.py
docker compose run --rm wwtp python tools/remote_sensing/run.py train configs/remote_sensing/potsdam_segformer_full.py
docker compose run --rm wwtp python tools/remote_sensing/run.py eval configs/remote_sensing/potsdam_segformer_full.py --checkpoint work_dirs/remote_sensing/potsdam_segformer_full/iter_32000.pth
python3 tools/remote_sensing/compare.py potsdam work_dirs/remote_sensing/potsdam_segformer_full_eval/metrics.json
```

如 `manifest.json` 的 `official_eroded_reference` 为 `true`，可用同一最终权重单独评价官方去边界标注：

```bash
docker compose run --rm wwtp python tools/remote_sensing/run.py eval configs/remote_sensing/potsdam_segformer_official_eroded.py --checkpoint work_dirs/remote_sensing/potsdam_segformer_full/iter_32000.pth
```

`XXXX` 换成实际最佳迭代数。Potsdam baseline（含内部验证配置）训练使用 32k iterations、batch size 4；RPGV 保持 40k iterations，二者训练预算不同。其他默认设置为随机种子 42、ImageNet 预训练权重；LoveDA 验证在 1024×1024 原图上做 512 窗口、384 步长滑窗推理，Potsdam 对无重叠 512 补丁推理。若单卡显存不足，用 `--cfg-options train_dataloader.batch_size=2` 并记录更改。训练和评估输出位于 `work_dirs/remote_sensing/`，不会写入已有 WWTP 实验目录。

`compare.py` 从 [Samba 论文表 2、表 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC11466675/)读取 SegFormer、DeepLabV3+、Samba 的公开参考数值，并计算本地结果差值。论文采用的模型规模、训练设置和推理方式不完全相同；该差值是文献对照，不等同于受控复现。尤其论文表中的 SegFormer 参数量为 3.7M，当前配置是 MiT-B2，不能视为同一模型。

| 论文方法 | LoveDA Val mIoU / mF1 | Potsdam 14 张、五类 mIoU / mF1 |
| --- | ---: | ---: |
| DeepLabV3+ / ResNet50 | 34.60 / 50.72 | 75.23 / 85.70 |
| SegFormer / Mix ViT | 43.16 / 59.75 | 81.13 / 89.42 |
| Samba / UperNet | 47.11 / 63.17 | 82.29 / 90.15 |

LoveDA 官方 Test 没有公开标签；需要盲测成绩时，将预测按 [LoveDA 当前 Codabench 竞赛](https://www.codabench.org/competitions/13030)要求提交。不要用 Val 分数冒充 Test 分数。
LoveDA 作者限定影像与标注仅用于学术用途，禁止商业使用，详见[官方仓库许可说明](https://github.com/Junjue-Wang/LoveDA#license)。

## 扩展方法

`run.py` 接受任意符合 MMSeg 格式的配置文件。新方法只需继承其中一个数据集配置，修改 `model` 并为其类别数设置解码头。保留同一划分、输入通道、裁剪大小、训练步数、预训练来源和评价指标，才能做受控对照。
