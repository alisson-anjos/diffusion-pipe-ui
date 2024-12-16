ARG CUDA_VERSION="12.4.1"
ARG CUDNN_VERSION=""
ARG UBUNTU_VERSION="22.04"
ARG DOCKER_FROM=nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION
ARG GRADIO_PORT=7860

FROM $DOCKER_FROM AS base

WORKDIR /workspace

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHON_VERSION=3.12
ENV CONDA_DIR=/opt/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
ENV MODEL_DIR="/models" 
ENV OUTPUT_DIR="/output"
ENV POETRY_HOME="$CONDA_DIR"
ENV PATH="$POETRY_HOME/bin:$PATH"
ENV NUM_GPUS=1

# Install dependencies required for Miniconda
RUN apt-get update -y && \
    apt-get install -y wget bzip2 ca-certificates git curl && \
    apt-get install -y --no-install-recommends openssh-server openssh-client git-lfs vim zip unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda init bash

# Create environment with Python 3.12
RUN $CONDA_DIR/bin/conda create -n pyenv python=3.12 -y

# Install Poetry in the conda environment
RUN $CONDA_DIR/bin/conda run -n pyenv pip install poetry && \
    $CONDA_DIR/bin/conda run -n pyenv poetry config virtualenvs.create false

# Define PyTorch versions via arguments
ARG PYTORCH="2.4.1"
ARG CUDA="124"

# Install PyTorch with specified version and CUDA
RUN $CONDA_DIR/bin/conda run -n pyenv \
    pip install torch==$PYTORCH torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$CUDA

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* /workspace/

# Install Poetry dependencies
RUN $CONDA_DIR/bin/conda run -n pyenv \
    poetry install --only main --no-interaction --no-ansi

# Install git lfs
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Install nginx
RUN apt-get update && \
apt-get install -y nginx

COPY default /etc/nginx/sites-available/default

# Add Jupyter Notebook
RUN pip3 install jupyterlab nodejs
EXPOSE 8888

COPY poetry.lock pyproject.toml /workspace/

COPY --chmod=755 start.sh /start.sh
COPY --chmod=755 entrypoint.sh /entrypoint.sh

# Expose the Gradio port
EXPOSE $GRADIO_PORT

CMD [ "/start.sh" ]
