ARG CUDA_VERSION="12.4.1"
ARG CUDNN_VERSION=""
ARG UBUNTU_VERSION="22.04"
ARG DOCKER_FROM=nvidia/cuda:$CUDA_VERSION-cudnn$CUDNN_VERSION-devel-ubuntu$UBUNTU_VERSION
ARG GRADIO_PORT=7860

FROM $DOCKER_FROM AS base

WORKDIR /app

# Variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHON_VERSION=3.12
ENV CONDA_DIR=/opt/conda
ENV PATH="$CONDA_DIR/bin:$PATH"
ENV MODEL_DIR="/models" 
ENV OUTPUT_DIR="/output"
ENV POETRY_HOME="$CONDA_DIR"
ENV PATH="$POETRY_HOME/bin:$PATH"

# Instalar dependências necessárias para Miniconda
RUN apt-get update -y && \
    apt-get install -y wget bzip2 ca-certificates git curl && \
    apt-get install -y --no-install-recommends openssh-server openssh-client git-lfs vim zip unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Baixar e instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda init bash

# Criar ambiente com Python 3.12
RUN $CONDA_DIR/bin/conda create -n pyenv python=3.12 -y

# Instalar Poetry no ambiente conda
RUN $CONDA_DIR/bin/conda run -n pyenv pip install poetry && \
    $CONDA_DIR/bin/conda run -n pyenv poetry config virtualenvs.create false

# Definir versões do PyTorch via argumentos
ARG PYTORCH="2.4.1"
ARG CUDA="124"

# Instalar PyTorch com a versão e CUDA especificadas
RUN $CONDA_DIR/bin/conda run -n pyenv \
    pip install torch==$PYTORCH torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$CUDA

# Copiar arquivos de configuração do Poetry
COPY pyproject.toml poetry.lock* /app/

# Instalar dependências do Poetry
RUN $CONDA_DIR/bin/conda run -n pyenv \
    poetry install --only main --no-interaction --no-ansi

# Copiar todo o conteúdo do projeto
COPY . /app

# Garantir que o script de entrypoint seja executável
RUN chmod +x /app/entrypoint.sh

# Expor a porta do Gradio
EXPOSE $GRADIO_PORT

# Definir o entrypoint original
ENTRYPOINT ["/app/entrypoint.sh"]