ARG CUDA_VERSION="12.4.1"
ARG CUDNN_VERSION=""
ARG UBUNTU_VERSION="22.04"
ARG DOCKER_FROM=nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION
ARG GRADIO_PORT=7860
ARG DOTNET_PORT=5000

FROM $DOCKER_FROM AS base

WORKDIR /

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHON_VERSION=3.12
ENV NUM_GPUS=1

# Install dependencies required for Miniconda
RUN apt-get update -y && \
    apt-get install -y wget bzip2 ca-certificates git curl && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    emacs \
    git \
    jq \
    libcurl4-openssl-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libssl-dev \
    libxext6 \
    libxrender-dev \
    software-properties-common \
    openssh-server \
    openssh-client \
    git-lfs \
    vim \
    zip \
    unzip \
    zlib1g-dev \
    libc6-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get update && apt-get install libstdc++6 -y

ENV LD_LIBRARY_PATH=/usr/local/cuda-12.4.1/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y git-lfs && git lfs install && apt-get install -y nginx

RUN add-apt-repository ppa:dotnet/backports

RUN apt install aspnetcore-runtime-9.0 -y

COPY docker/default /etc/nginx/sites-available/default

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY requirements.txt .

RUN uv pip install jupyterlab ipywidgets jupyter-archive jupyter_contrib_nbextensions nodejs --system

RUN uv pip install -U "huggingface_hub[cli]" --system

COPY triton-3.2.0-cp312-cp312-linux_x86_64.whl .
COPY sageattention-2.1.1-cp312-cp312-linux_x86_64.whl .
COPY flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl .
COPY transformers-4.49.0.dev0-py3-none-any.whl .

EXPOSE 8888

# Tensorboard
EXPOSE 6006 

# Debug
# RUN $CONDA_DIR/bin/conda run -n pyenv \
#     pip install debugpy

# EXPOSE 5678


# Copy the entire project
COPY --chmod=755 . /diffusion-pipe

COPY --chmod=755 docker/initialize.sh /initialize.sh
COPY --chmod=755 docker/entrypoint.sh /entrypoint.sh

# Expose the Gradio port
EXPOSE $GRADIO_PORT
EXPOSE $DOTNET_PORT

CMD [ "/initialize.sh" ]
