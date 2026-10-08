# 数据准备与官方划分

准备程序不包含下载和随机划分逻辑。使用官方解压包，输入目录必须为：

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

输出统一为 `img_dir/{train,val,test}`、`ann_dir/{train,val}`、`manifest.json`。文件名前缀保留 sequence 或 Urban/Rural，防止重名。训练、评估、导出入口强制校验清单中的官方源数量、sequence 划分和实际图片/标签清单，缺失或混入其他 sequence 时立即失败。UAVid 切片不改变源帧的划分归属。

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

默认 `WWTP_DATA_ROOT=wwtp_semantic_dataset_1m`、`UAVID_DATA_ROOT=data/remote_sensing/prepared/uavid`、`LOVEDA_DATA_ROOT=data/remote_sensing/prepared/loveda`；Compose 中传入 `/workspace/...` 路径。WWTP 数据保留 `images/<split>`、`annotations/<split>`。WWTP 的旧版尺度转换用 `tools/data/resample_wwtp_dataset.py`，不在训练管线中偷偷改变地面分辨率。
