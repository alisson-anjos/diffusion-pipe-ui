#!/bin/bash

# Marker file path
INIT_MARKER="/var/run/container_initialized"
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-"true"}  # Default if not set

echo "DOWNLOAD_MODELS is: $DOWNLOAD_MODELS"

source /opt/conda/etc/profile.d/conda.sh
conda activate pyenv

REPO_URL_UI=${REPO_URL_UI:-"https://github.com/alisson-anjos/diffusion-pipe-ui"}
REPO_BRANCH_UI=${REPO_BRANCH_UI:-"testing"}
REPO_DIR_UI=${REPO_DIR_UI:-"/workspace/diffusion-pipe-ui"}

# Clone repository if not present
if [ ! -d "$REPO_DIR_UI/.git" ]; then
    echo "Cloning repository $REPO_URL_UI with submodules..."
    git clone --recurse-submodules --branch $REPO_BRANCH_UI $REPO_URL_UI $REPO_DIR_UI
fi

# Update submodules
cd $REPO_DIR_UI
git submodule update --init --recursive

if [ ! -f "$INIT_MARKER" ]; then
    echo "First-time initialization..."

    echo "Installing CUDA nvcc..."
    conda install -y -c nvidia cuda-nvcc --override-channels

    echo "Installing dependencies from requirements.txt..."
    pip install --no-cache-dir -r $REPO_DIR_UI/diffusion-pipe/requirements.txt

    export PYTHONPATH="$REPO_DIR_UI:$REPO_DIR_UI/diffusion-pipe:$REPO_DIR_UI/diffusion-pipe/submodules/HunyuanVideo:$PYTHONPATH"
    export PYTHONPATH="$REPO_DIR_UI:$REPO_DIR_UI/diffusion-pipe/configs:$PYTHONPATH"
    export PATH="$REPO_DIR_UI:$REPO_DIR_UI/diffusion-pipe:$PATH"

    if [ "$DOWNLOAD_MODELS" = "true" ]; then
        echo "DOWNLOAD_MODELS is true, downloading models..."
        MODEL_DIR="/workspace/models"
        mkdir -p "$MODEL_DIR"

        # Clone llava-llama-3-8b-text-encoder-tokenizer repository
        if [ ! -d "${MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer" ]; then
            git clone https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer "${MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer"
            cd "${MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer"
            git lfs pull
            cd -
        fi

        # Download hunyuan_video_720_cfgdistill_fp8_e4m3fn model
        if [ ! -f "${MODEL_DIR}/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" ]; then
            curl -L -o "${MODEL_DIR}/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors?download=true"
        fi

        # Download hunyuan_video_vae_fp32 model
        if [ ! -f "${MODEL_DIR}/hunyuan_video_vae_fp32.safetensors" ]; then
            curl -L -o "${MODEL_DIR}/hunyuan_video_vae_fp32.safetensors" "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_vae_fp32.safetensors?download=true"
        fi

        # Download hunyuan_video_vae_fp16 model
        if [ ! -f "${MODEL_DIR}/hunyuan_video_vae_fp16.safetensors" ]; then
            curl -L -o "${MODEL_DIR}/hunyuan_video_vae_fp16.safetensors" "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_vae_fp16.safetensors?download=true"
        fi

        # Clone the entire CLIP repo
        if [ ! -d "${MODEL_DIR}/clip-vit-large-patch14" ]; then
            git clone https://huggingface.co/openai/clip-vit-large-patch14 "${MODEL_DIR}/clip-vit-large-patch14"
            cd "${MODEL_DIR}/clip-vit-large-patch14"
            git lfs pull
            cd -
        fi
    else
        echo "DOWNLOAD_MODELS is false, skipping model downloads."
    fi

    # Create marker file
    touch "$INIT_MARKER"
    echo "Initialization complete."
else
    echo "Container already initialized. Skipping first-time setup."
fi

# Create Triton autotune directory
mkdir -p /root/.triton/autotune

exec python /workspace/diffusion-pipe-ui/main.py
