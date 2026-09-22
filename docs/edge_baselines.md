# CBR-Net 与 HD-Net 边界模型对比

本目录新增两个独立的遥感边界分割实验。网络代码位于
`wwtpseg/edge_baselines/`，实验配置位于 `configs/edge_baselines/`。
两者复用项目原有的 `WWTPDataset`、512 像素训练裁剪、2048 像素原图滑窗测试和
`BinaryBoundaryMetric`。训练时从增强后的 0/1 掩膜即时计算边界与方向监督；
不生成或修改 `wwtp_semantic_dataset/` 中的文件。模型仅在对应配置的
`custom_imports` 中注册，原有模型配置不需要加载它们。

| 模型 | 实现依据 | 主要机制 | 辅助监督 |
| --- | --- | --- | --- |
| CBR-Net | [官方仓库](https://github.com/HaonanGuo/CBRNet)，commit `5d521ce1` | VGG16-BN 编码器，四级粗到细预测，边界门控的八方向像素修正 | 四级分割、四级边界、方向分类、自监督修正预测 |
| HD-Net | [官方仓库](https://github.com/danfenghong/ISPRS_HD-Net)，commit `186d6562` | 两个双分支与四个三分支高分辨率阶段，流场驱动的主体/边界解耦 | 六级分割、六级边界及最终融合输出 |

代码按现有 MMSeg 接口重新组织，与上游脚本的数据格式、权重键名和训练循环不完全相同。
上游发布的 checkpoint 不能直接当作本实现的权重加载。HD-Net 上游按 GPL-3.0 发布；
如果将本目录代码分发到其他项目，应保留原作者及 [GPL-3.0 许可证](../wwtpseg/edge_baselines/GPL-3.0.txt)。CBR-Net 上游仓库未附显式许可证，
本目录没有复制其原始文件。

## 验证

在项目现有镜像中运行：

```bash
docker compose run --rm wwtp python tools/smoke_edge_baselines.py --dataset
```

该脚本用 64×64 合成样本检查前向、反向传播与 96×96 滑窗输出；
`--dataset` 另读取每个实验的一份真实训练样本。构建测试时 CBR-Net 不下载预训练权重，
但正式配置默认使用 ImageNet VGG16-BN 权重。

## 训练与评估

```bash
docker compose run --rm wwtp \
  python tools/train.py configs/edge_baselines/cbr_net.py \
  --work-dir work_dirs/edge_baselines/cbr_net

docker compose run --rm wwtp \
  python tools/train.py configs/edge_baselines/hd_net.py \
  --work-dir work_dirs/edge_baselines/hd_net
```

测试时将 `<checkpoint>` 替换为对应实验目录中的最佳 checkpoint：

```bash
docker compose run --rm wwtp \
  python tools/test.py configs/edge_baselines/cbr_net.py <checkpoint> \
  --work-dir work_dirs/edge_baselines/cbr_net

docker compose run --rm wwtp \
  python tools/test.py configs/edge_baselines/hd_net.py <checkpoint> \
  --work-dir work_dirs/edge_baselines/hd_net
```

离线环境若无 VGG16-BN 预训练权重缓存，可给 CBR-Net 训练命令加
`--cfg-options model.pretrained=False`，此时会从随机初始化训练。
两份配置使用与现有基线相同的划分、优化器与调度文件，并将输出固定到各自的
`work_dirs/edge_baselines/` 子目录。对比时可直接汇总 `Foreground_IoU`、
`Boundary_F1` 和 `HD95_px`，并明确记录 CBR-Net 是否使用预训练权重。
