# syntax=docker/dockerfile:1
# Match OpenMMLab's prebuilt MMCV 2.1 wheels; training needs no CUDA compiler.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace \
    PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL="" \
    PIP_CONFIG_FILE=/dev/null PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=5 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Download caches live in BuildKit, not in the resulting training image.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends git libgl1 libglib2.0-0
WORKDIR /workspace

COPY requirements/core.txt /tmp/requirements/core.txt
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install "pip<25" "setuptools<80" wheel \
    && python -m pip install -r /tmp/requirements/core.txt

# Require the official CUDA wheel: fail instead of silently compiling from source.
# Changing application requirements/code reuses this dependency layer.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install -c /tmp/requirements/core.txt --only-binary=mmcv mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

COPY requirements/runtime.txt /tmp/requirements/runtime.txt
# Replace Conda's Ninja metadata with a compatible PyPI wheel for pip check.
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
