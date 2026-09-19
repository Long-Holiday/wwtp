ARG PYTORCH_IMAGE=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential git libgl1 libglib2.0-0 ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements /tmp/requirements

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install openmim==0.3.9 \
    && mim install "mmengine==0.10.7" "mmcv==2.1.0" \
    && python -m pip install -r /tmp/requirements/runtime.txt \
    && python -m pip install --no-build-isolation \
       -r /tmp/requirements/mamba.txt

COPY configs configs
COPY tools tools
COPY scripts scripts
COPY wwtpseg wwtpseg
COPY README.md README.md

CMD ["bash"]
