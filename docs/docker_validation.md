# Docker 镜像验证记录

验证日期：2026-10-08。镜像：`wwtp-mmseg:torch2.1.2-cu121`。

## 环境与结果

宿主机为 Ubuntu 26.04 / WSL2，GPU 为 NVIDIA GeForce RTX 3050 Laptop GPU（4 GB），驱动 616.92。镜像使用 Python 3.10、PyTorch 2.1.2、torchvision 0.16.2、CUDA 12.1、NumPy 1.26.4、MMCV 2.1.0、MMEngine 0.10.7、MMSegmentation 1.2.2、MMDetection 3.3.0。

| 验证 | 结果 |
| --- | --- |
| 镜像构建、`pip check`、CUDA 扩展与项目模块导入 | 通过 |
| 全部现有单元与回归测试 | 65 项中 CPU 通过 61 项，跳过的 4 项在 GPU 补跑通过 |
| GPU 补充数值与 AMP 测试 | 7 项全部通过，包含上述 4 项 |
| WWTP、UAVid、LoveDA 各七个主流模型 | 21 组 CUDA 前向、反向和预测检查通过，包含 UAVid 滑窗预测 |
| RPGV v1、v2、v3、v4、v5、v5.1 | 6 组 CUDA 前向、反向和预测检查通过 |
| 生产训练入口、保存、评估、恢复 | 合成数据训练至第 2 次迭代，保存并评估，再恢复至第 4 次迭代，通过 |
| torchvision / MMCV CUDA 算子实测 | NMS、可变形卷积及可变形注意力通过，后两者包含反向检查 |

测试无需真实数据集和预训练权重。模型冒烟测试使用 64×64 输入、batch 2；数值测试另覆盖部分 1024 分辨率损失。上述结果不代表完整 1024 分辨率训练的显存需求、吞吐或精度。

## 复现

```bash
docker compose build
docker run --rm -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  wwtp-mmseg:torch2.1.2-cu121 \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

本机尚未安装 NVIDIA Container Toolkit，`docker compose` 的 `gpus: all` 会报告 `no known GPU vendor found`。本次 GPU 测试使用 WSL 的 `/dev/dxg` 和只读驱动库临时映射，不修改宿主机配置：

```bash
docker run --rm --device /dev/dxg \
  --mount type=bind,src=/usr/lib/wsl,dst=/usr/lib/wsl,readonly \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/conda/lib \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e RPGV_V4_TEST_CUDA=1 \
  --shm-size 8g wwtp-mmseg:torch2.1.2-cu121 bash -c '
    set -e
    python tests/smoke_test.py --dataset wwtp --include-rpgv --backward --device cuda
    python tests/smoke_test.py --dataset uavid --backward --device cuda
    python tests/smoke_test.py --dataset loveda --backward --device cuda
    python tests/smoke_runner.py --device cuda
    python -m unittest tests.rpgv.test_rpgv_numerics \
      tests.rpgv.test_rpgv_v2.V2Test.test_cuda_amp_updates \
      tests.rpgv.test_rpgv_v4.V4Test.test_cuda_amp_update -v
  '
```

这条 GPU 启动命令仅适用于 WSL2。常规部署应按 [NVIDIA 官方安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)配置 NVIDIA Container Toolkit，再使用 README 中的 Compose 命令。

## 测试期间修复

- APT 增加下载重试，处理 Ubuntu 索引下载中断。
- UAVid 滑窗冒烟测试使用 `ConfigDict`，满足 MMSeg 的配置访问方式。
- 推理阶段移除训练早停钩子，避免独立评估尝试加载训练集。
- RPGV v2 独立评估替换已有拓扑指标，避免重复指标名，保持其 256 像素阈值。
- 合成评估用例的尺寸校验同步为 64×64，保留生产配置的 1024×1024 校验。
