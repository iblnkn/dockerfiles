# syntax=docker/dockerfile:1.4

################################################################################
# 1) BASE  –  Minimal CUDA build environment on Ubuntu 24.04
################################################################################
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS base  

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG USERNAME=dev
ARG USER_UID=2000
ARG USER_GID=$USER_UID

FROM base AS system

# ---------- Locale, tz, core utils ------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        locales tzdata curl ca-certificates gnupg2 sudo git git-lfs openssh-client \
        build-essential cmake ninja-build pkg-config \
        wget unzip nano vim less htop tree util-linux \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv python3-pip \
        libglib2.0-0 libgl1 libglx-mesa0 ffmpeg \
        portaudio19-dev libgeos-dev \
    && locale-gen en_US.UTF-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PATH=/root/.local/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# ---------- Poetry (system-wide, no venv nesting) ---------------------------------
RUN curl -sSL https://install.python-poetry.org | python${PYTHON_VERSION} - && \
    poetry config virtualenvs.create false

# ---------- Non-root user ---------------------------------------------------------
RUN set -eux; \
    # 1) create or re-use the group
    getent group "${USER_GID}" > /dev/null || groupadd --gid "${USER_GID}" "${USERNAME}" && \
    # 2) create the user only if it does not exist yet
    id -u "${USERNAME}" > /dev/null 2>&1 || adduser --disabled-password --gecos "" --uid "${USER_UID}" --gid "${USER_GID}" "${USERNAME}" && \
    # 3) sudo without password
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

################################################################################
# 2) RUNTIME  –  Everything needed to *run* LeRobot + PyTorch CU128
################################################################################
FROM system AS runtime

# ---------- PyPI packages ---------------------------------------------------------
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Create and activate virtual environment
RUN python3.12 -m venv $VIRTUAL_ENV && \
    . $VIRTUAL_ENV/bin/activate && \
    python -m pip install --upgrade pip && \
    python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.2.* torchvision torchaudio && \
    python -m pip install \
        cmake>=3.29 datasets>=2.19 deepdiff>=7.0.1 diffusers>=0.27.2 draccus==0.10.0 \
        einops>=0.8 flask gdown gymnasium==0.29.1 h5py huggingface-hub[cli,hf-transfer]>=0.27 \
        imageio[ffmpeg] jsonlines numba omegaconf opencv-python-headless av pymunk pynput pyzmq \
        rerun-sdk termcolor wandb zarr

# ---------- (optional) clone LeRobot into /opt if you want baked-in code ----------
# RUN git clone https://github.com/huggingface/lerobot.git /opt/lerobot && \
#     cd /opt/lerobot && poetry install --without dev --no-interaction

WORKDIR /workspace
USER $USERNAME
CMD [ "bash" ]

################################################################################
# 3) DEV  –  Adds test/lint/debug tools, produces the image VS Code will open
################################################################################
FROM runtime AS dev

USER root
RUN python -m pip install \
        ipython ipdb jupyterlab \
        black flake8 ruff pytest pytest-cov \
        pre-commit \
        rich \
        transformers \
        accelerate==0.30.*

# VS Code expects its user to own the workspace folder
ARG USERNAME=dev
RUN mkdir -p /workspace && chown $USERNAME:$USERNAME /workspace
USER $USERNAME
WORKDIR /workspace
