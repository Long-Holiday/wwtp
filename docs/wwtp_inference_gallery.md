# WWTP val/test 推理对比图

使用 `tools/visualize_wwtp_predictions.py` 在 **WWTP** 数据集上生成 val/test 推理结果。它从每个模型的原始训练和续训目录中，按训练日志记录的验证集 `binary/Foreground_IoU` 选取最高分的 `best_*.pth`。测试集指标不参与选权重。纳入 RPGV-Net、DeepLabV3+、HRNet、Mask2Former、RS-Mamba、SegFormer、SegNeXt、U-Net、UNetFormer、CBR-Net 和 HD-Net。

先检查选中的权重：

```bash
docker compose run --rm wwtp python tools/visualize_wwtp_predictions.py plan
```

训练占用 GPU 时，请等训练结束后再运行：

```bash
docker compose run --rm wwtp python tools/visualize_wwtp_predictions.py run
```

默认输出 `work_dirs/wwtp_inference_gallery/`。`manifest.json` 记录每个模型对应的配置、权重路径和验证分数。`masks/{val,test}/<model>/<image>.png` 是原分辨率的 0/255 黑白掩码；`comparisons/{val,test}/<image>.png` 是原图、原图叠加红色 GT、各模型黑白掩码的带标题小图。运行中断后重复执行会跳过已有掩码和对比图。只读取数据集，所有文件均写入独立的输出目录。

单张烟测可使用独立目录：

```bash
docker compose run --rm wwtp python tools/visualize_wwtp_predictions.py run \
  --models SegFormer --splits val --limit 1 \
  --output /workspace/work_dirs/wwtp_gallery_smoke
```
