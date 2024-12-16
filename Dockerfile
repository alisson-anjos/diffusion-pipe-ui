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
ENV PATH="/root/.local/bin:/usr/local/bin:$PATH"
ENV NUM_GPUS=1

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y \
    software-properties-common \
    git \
    git-lfs \
    curl \
    wget \
    zip \
    unzip \
    vim \
    nginx \
    openssh-server \
    openssh-client \
    openmpi-bin \
    libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false

# Install git lfs
RUN git lfs install

# Configure nginx
COPY default /etc/nginx/sites-available/default

# Copy project files
COPY pyproject.toml poetry.lock* /workspace/

# Add Jupyter Notebook
RUN pip3 install jupyterlab nodejs
EXPOSE 8888

# Copy scripts
COPY --chmod=755 start.sh /start.sh
COPY --chmod=755 entrypoint.sh /entrypoint.sh

# Create necessary directories
RUN mkdir -p /workspace/models /workspace/output /workspace/config_history /workspace/datasets

# Expose the Gradio port
EXPOSE $GRADIO_PORT

CMD [ "/start.sh" ]
