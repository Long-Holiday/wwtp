# Newer PyTorch/CUDA profile. MMSeg 1.2.2 requires MMCV < 2.2;
# build the real CUDA ops against this PyTorch instead of reusing an ABI-mismatched wheel.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"
ARG MAX_JOBS=2
ARG MMCV_FROM_SOURCE=1
ARG MMCV_WHEEL_INDEX=https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 MMCV_WITH_OPS=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PYTHONPATH=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements /tmp/requirements
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python -m pip install "pip<25" "setuptools<80" wheel ninja \
    && python -m pip install numpy==1.26.4 mmengine==0.10.7 \
    && if [ "$MMCV_FROM_SOURCE" = "1" ]; then \
         python -m pip install --no-build-isolation --no-binary=mmcv mmcv==2.1.0; \
       else \
         python -m pip install --only-binary=mmcv mmcv==2.1.0 -f "$MMCV_WHEEL_INDEX"; \
       fi \
    && python -m pip install -r /tmp/requirements/runtime.txt \
    && python -c "import torch, mmcv, mmseg, mmdet; from mmcv.ops import MultiScaleDeformableAttention; print(torch.__version__, torch.version.cuda, mmcv.__version__, mmseg.__version__)"
COPY configs configs
COPY tools tools
COPY scripts scripts
COPY wwtpseg wwtpseg
COPY tests tests
COPY README.md README.md
CMD ["bash"]
