# syntax=docker/dockerfile:1
# Newer PyTorch/CUDA profile. MMSeg 1.2.2 requires MMCV < 2.2;
# build the real CUDA ops against this PyTorch instead of reusing an ABI-mismatched wheel.
ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace \
    PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL="" \
    PIP_CONFIG_FILE=/dev/null PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=5 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Download caches live in BuildKit, not in the resulting training image.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgl1 libglib2.0-0 ninja-build
WORKDIR /workspace

COPY requirements/core.txt /tmp/requirements/core.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install "pip<25" "setuptools<80" wheel ninja \
    && python -m pip install -r /tmp/requirements/core.txt

# Changing application requirements/code must not rebuild the CUDA extension.
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"
ARG MAX_JOBS=2
ARG MMCV_FROM_SOURCE=1
ARG MMCV_WHEEL_INDEX=https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ "$MMCV_FROM_SOURCE" = "1" ]; then \
         FORCE_CUDA=1 MMCV_WITH_OPS=1 CUDA_HOME=/usr/local/cuda \
         TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" MAX_JOBS="$MAX_JOBS" \
         PIP_CONSTRAINT=/tmp/requirements/core.txt \
         python -m pip install -v --no-build-isolation --no-binary=mmcv mmcv==2.1.0; \
       else \
         python -m pip install -c /tmp/requirements/core.txt --only-binary=mmcv mmcv==2.1.0 -f "$MMCV_WHEEL_INDEX"; \
       fi

COPY requirements/runtime.txt /tmp/requirements/runtime.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install -r /tmp/requirements/runtime.txt \
    && python -m pip install --force-reinstall --no-deps --only-binary=:all: ninja==1.11.1.1 \
    && python -m pip check \
    && python -c "import torch, mmcv, mmseg, mmdet; from mmcv.ops import MultiScaleDeformableAttention; print(torch.__version__, torch.version.cuda, mmcv.__version__, mmseg.__version__)"
COPY configs configs
COPY tools tools
COPY scripts scripts
COPY wwtpseg wwtpseg
COPY tests tests
RUN python -c "import wwtpseg, wwtpseg.models.segmentors.hdnet, wwtpseg.models.segmentors.multiclass; print('Project imports OK')"
COPY README.md README.md
CMD ["bash"]
