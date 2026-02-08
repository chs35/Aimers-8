# EXAONE-4.0-1.2B hackathon evaluation-like environment
# Target: Ubuntu 22.04.x + CUDA 12.8 + Python 3.11.14 + vLLM 0.14.1 + torch 2.9.0+cu128

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG PYTHON_VERSION=3.11.14

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# README 기준 시스템 패키지 + Python 3.11.14 소스 빌드 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    openssh-server \
    ninja-build \
    libomp-dev \
    libblas3 \
    liblapack3 \
    gfortran \
    libatlas-base-dev \
    curl \
    wget \
    ca-certificates \
    unzip \
    procps \
    cmake \
    tzdata \
    libxcb1 \
    xz-utils \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# SSH 설정 복원
RUN mkdir -p /var/run/sshd /root/.ssh \
    && printf '%s\n' \
        'PermitRootLogin yes' \
        'PasswordAuthentication no' \
        'PubkeyAuthentication yes' >> /etc/ssh/sshd_config \
    && printf '%s\n' \
        'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFEqD9Pih5Cae023XyG+wgRV6WIrSybg0Y545TijJr8P junseok0913@DESKTOP-VJH6PBH' \
        'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHfpMq5LJbRweQEFLAOPtX9JZ4SvgWiYM+YniYKL+vvT junseok0913@kakao.com' \
        'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB/yvwiEQyuR2mW5mBo/GluXW9MqZEhwIlQS/I9BnDPr hs040305@naver.com' \
        > /root/.ssh/authorized_keys \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/authorized_keys

# Python 3.11.14 정확 고정 설치
RUN cd /tmp \
    && wget -q https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --prefix=/usr/local --with-ensurepip=install \
    && make -j"$(nproc)" \
    && make altinstall \
    && ln -sf /usr/local/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip \
    && python --version \
    && pip --version \
    && cd / \
    && rm -rf /tmp/Python-${PYTHON_VERSION} /tmp/Python-${PYTHON_VERSION}.tgz

RUN pip install --upgrade pip setuptools wheel

# vLLM 0.14.1은 torch==2.9.1을 강제하므로 설치 후 cu128 스택으로 재고정
RUN pip install vllm==0.14.1 \
    && pip install --force-reinstall \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        torch==2.9.0+cu128 \
        torchvision==0.24.0+cu128 \
        torchaudio==2.9.0+cu128

# README 명시 Python 패키지 고정
RUN pip install \
    numpy==2.2.6 \
    pandas==2.3.3 \
    transformers==4.57.3 \
    tokenizers==0.22.1 \
    accelerate==1.10.1 \
    datasets==4.4.1 \
    huggingface-hub==0.36.0 \
    safetensors==0.7.0 \
    sentencepiece==0.2.1 \
    protobuf==6.33.2 \
    evaluate==0.4.6 \
    rouge_score==0.1.2 \
    sacrebleu==2.5.1 \
    tqdm==4.67.1 \
    regex==2025.11.3 \
    nltk==3.9.2 \
    compressed-tensors==0.13.0 \
    math-verify==0.8.0 \
    antlr4-python3-runtime==4.11.0 \
    sympy==1.14.0 \
    langdetect==1.0.9 \
    immutabledict==4.2.2 \
    torch-c-dlpack-ext==0.1.4

# 핵심 버전 검증
RUN python - <<'PY'
import platform
import torch
import vllm

assert platform.python_version() == "3.11.14", platform.python_version()
assert torch.__version__.startswith("2.9.0+cu128"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert vllm.__version__ == "0.14.1", vllm.__version__

print("python", platform.python_version())
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
PY

WORKDIR /workspace

RUN git lfs install

CMD ["/bin/bash"]
