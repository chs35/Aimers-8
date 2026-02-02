# EXAONE-4.0-1.2B 경량화 해커톤 평가 서버 환경
# OS: Ubuntu 22.04.5 LTS, CUDA 12.8, Python 3.11.14

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    python3.11-dev \
    build-essential \
    git \
    git-lfs \
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
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# SSH 설정
RUN mkdir -p /var/run/sshd ~/.ssh && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config && \
    echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config && \
    echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFEqD9Pih5Cae023XyG+wgRV6WIrSybg0Y545TijJr8P junseok0913@DESKTOP-VJH6PBH' > ~/.ssh/authorized_keys && \
    echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHfpMq5LJbRweQEFLAOPtX9JZ4SvgWiYM+YniYKL+vvT junseok0913@kakao.com' >> ~/.ssh/authorized_keys && \
    chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys

# Python 3.11을 기본으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# pip 업그레이드
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    python3.11 -m pip install --upgrade pip

# PyTorch 설치 (CUDA 12.8)
RUN pip install torch==2.9.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Python 패키지 설치
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

# vLLM 설치
RUN pip install vllm==0.14.1

# 작업 디렉토리 설정
WORKDIR /workspace

# Git LFS 초기화
RUN git lfs install

# 기본 명령어
CMD ["/bin/bash"]
