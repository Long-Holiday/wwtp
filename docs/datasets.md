# 数据准备与划分协议

准备程序不下载数据。LoveDA/UAVid保留官方划分；DeepGlobe从官方有标签train生成确定性的本地划分，见下节。使用已解压包，输入目录必须为：

```text
raw/loveda/
  Train/{Urban,Rural}/{images_png,masks_png}/*.png
  Val/{Urban,Rural}/{images_png,masks_png}/*.png
  Test/{Urban,Rural}/images_png/*.png
raw/uavid/
  uavid_train/seq*/{Images,Labels}/*.png
  uavid_val/seq*/{Images,Labels}/*.png
  uavid_test/seq*/Images/*.png
```

| 数据集 | 官方 train | 官方 val | 官方 test | 划分单位 |
| --- | ---: | ---: | ---: | --- |
| LoveDA | 2522 | 1669 | 1796 | 官方图像 |
| UAVid2020 | 200 | 70 | 150 | 官方 sequence，20/7/15 个 sequence，每个十帧 |

UAVid2020 必须使用官方 `uavid_train/uavid_val/uavid_test` 目录，不混用 UAVid 早期版本、UAVid++ 或第三方重新划分版本。程序检查 sequence 不跨集合、图像/标签完整配对；保持原文件来源清单。不能仅凭总数量证明数据来自官方包，输入包的版本和来源由使用者保证。

输出：LoveDA/UAVid统一为 `img_dir/{train,val,test}`、`ann_dir/{train,val}`、`manifest.json`。文件名前缀保留 sequence 或 Urban/Rural，防止重名。训练、评估、导出入口强制校验清单中的官方源数量、sequence 划分和实际图片/标签清单，缺失或混入其他 sequence 时立即失败。UAVid 切片不改变源帧的划分归属。

LoveDA 使用 MMSeg `LoveDADataset`：原掩膜 0 忽略，1–7 变成训练 ID 0–6，忽略像素为 255。提交使用 0–6 ID、原图片编号和平铺目录，与[官方预测脚本](https://github.com/Junjue-Wang/LoveDA/blob/master/Semantic_Segmentation/predict.py)一致。类别为 background、building、road、water、barren、forest、agricultural。官方标签语义见 [MMSeg LoveDA 实现](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/datasets/loveda.py)。

UAVid 的八类训练 ID 与 [GeoSeg 颜色转换约定](https://github.com/WangLibo1995/GeoSeg/blob/main/tools/uavid_patch_split.py) 一致：

| ID | 类别 | RGB |
| ---: | --- | --- |
| 0 | building | 128,0,0 |
| 1 | road | 128,64,128 |
| 2 | tree | 0,128,0 |
| 3 | low vegetation | 128,128,0 |
| 4 | moving car | 64,0,128 |
| 5 | static car | 192,0,192 |
| 6 | human | 64,64,0 |
| 7 | clutter | 0,0,0 |

白色边界与切片补齐区为 255。未知颜色拒绝转换，避免静默误标为背景。全部 val/test 帧保留原尺寸与边缘，推理结果回到原图尺寸。官方测试标签为私有，测试导出不计算本地指标。UAVid 数据集说明见 [官方站点](https://uavid.nl/)，LoveDA 来源见 [官方仓库](https://github.com/Junjue-Wang/LoveDA)。

默认 `WWTP_DATA_ROOT=wwtp_semantic_dataset_1m`、`UAVID_DATA_ROOT=data/remote_sensing/prepared/uavid`、`LOVEDA_DATA_ROOT=data/remote_sensing/prepared/loveda`、`DEEPGLOBE_DATA_ROOT=data/remote_sensing/prepared/deepglobe`；Compose 中传入 `/workspace/...` 路径。WWTP 数据保留 `images/<split>`、`annotations/<split>`。WWTP 的旧版尺度转换用 `tools/data/resample_wwtp_dataset.py`，不在训练管线中偷偷改变地面分辨率。

## DeepGlobe Road Extraction

来源：[Kaggle 数据集](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)，[DeepGlobe 2018 论文](https://arxiv.org/abs/1805.06561)。下载和许可证确认由使用者完成；解压到独立的新目录，项目不执行原始数据包中的脚本。

```text
raw/deepglobe/
  train/<id>_sat.jpg       # 必需，6226张RGB图像
  train/<id>_mask.png      # 必需，6226张配对黑白道路mask
  valid/<id>_sat.jpg       # 可选，1243张官方无标签验证图像
  test/<id>_sat.jpg        # 可选，1101张官方无标签测试图像
```

图像和mask必须为1024×1024，mask支持灰度或三通道一致的RGB黑白编码；原始0映射背景0，原始255映射道路1。拒绝其他像素值和错配尺寸，不通过阈值静默修正。训练类别为 `('background', 'road')`，`reduce_zero_label=False`，255仅保留为训练ignore约定，不是道路标签。数据版本/来源仍由用户保证，数量检查不能认证下载包。

### 本地可评估划分

官方valid/test没有可用真值，因此**仅从官方有标签train划分**本地训练/验证/测试集合：

1. 按原始ID字符串排序，再用独立 `random.Random(split_seed)` 打乱。
2. 本地val/test数量分别为 `floor(N×val_ratio)`、`floor(N×test_ratio)`；其余归train。
3. 默认seed42、比例0.8/0.1/0.1，6226张得到 **4982 / 622 / 622**。
4. 不切块、不resize、不合并官方无标签图像。源图ID无重复，各集合互斥且覆盖全部有标签train。

这是随机原图级划分，不是官方划分，也不是已知地理空间隔离划分。相同源图不跨split并不能排除不同原图的空间邻近；报告结果时必须披露本地协议，不与官方排行榜或不同划分结果直接比较。

```bash
python tools/data/prepare_public.py deepglobe \
  --source data/remote_sensing/raw/deepglobe --output data/remote_sensing/prepared/deepglobe
# 可选自定义；同一次正式模型比较必须复用同一prepared目录
python tools/data/prepare_public.py deepglobe \
  --source data/remote_sensing/raw/deepglobe --output data/remote_sensing/prepared/deepglobe_seed7 \
  --split-seed 7 --split-ratios 0.8 0.1 0.1
```

准备后的原图统一转为PNG、以原始ID命名：

```text
prepared/deepglobe/
  img_dir/{train,val,test}/<id>.png
  ann_dir/{train,val,test}/<id>.png    # 包括本地test标签
  img_dir/official_val/<id>.png       # 仅原始valid已提供时存在
  img_dir/official_test/<id>.png      # 仅原始test已提供时存在
  manifest.json
```

仅train的完整有标签包可完成准备。可选valid/test目录一旦提供，必须满足对应官方数量、无源ID重复且无mask；程序不把它们作为本地val/test。无标签集合只生成 `img_dir`，没有 `ann_dir`。准备使用 `.building` 暂存并拒绝覆盖，失败目录需检查原因后由用户清理或使用新的输出路径。

manifest区分 `official_source_counts`（标准官方数量）、`source_files`（三组本地用途，全部来自原始train）、`prediction_source_files`（实际提供的官方无标签来源），并记录协议版本、seed、比例、取整规则、各集合计数、标签语义与选权重/评估集合。运行时重算划分并验证prepared图像/标签成员，拒绝身份、划分和清单篡改。

### 训练与评估

七基线与公开RPGV v6/v7配置位于 `configs/deepglobe/`，采用1024整图训练和whole推理，同组统一RGB增强、AdamW与30k共享训练协议。`RoadIoUMetric`复用IoUMetric的累计像素交并统计，额外输出百分数 `roadIoU`/`roadF1`；没有道路且没有道路预测时道路分数为0，不按图平均、不附加边界距离或物理尺度指标。

所有模型按本地**val的roadIoU**保存最佳权重，批量 `train-test` 默认加载 `best_roadIoU_iter_*.pth` 评估本地**test**；缺最佳权重则报错，不回退到最后一次权重。test不能用于权重选择。RPGV伪几何采用已有离线生成工具和 `PUBLIC_PSEUDO_ROOT`，只从RGB计算，不读取标签。

官方无标签导出使用 `tools/evaluation/predict_public.py --split official_test`（或`official_val`），输出原始ID对应的 `<id>_mask.png`，像素0/255。RPGV需要提前为所选官方split准备伪几何缓存；自定义 `PUBLIC_PSEUDO_ROOT` 时，生成工具也要使用一致的 `--output-root`。导出不计算指标，也不自动提交外部评测服务。具体Docker训练、导出及测试命令见 [README](../README.md#deepglobe-road-extraction)。
