#!/bin/bash

# Marker file path
INIT_MARKER="/var/run/container_initialized"
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-"true"}  # Default if not set
DOWNLOAD_BF16=${DOWNLOAD_BF16:-"false"}  # Default if not set
REPO_DIR=${REPO_DIR:-"/workspace/diffusion-pipe"}
MODELS_DIR="/workspace/models"

echo "DOWNLOAD_MODELS is: $DOWNLOAD_MODELS and DOWNLOAD_BF16 is: $DOWNLOAD_BF16"

# source /opt/conda/etc/profile.d/conda.sh
# conda activate pyenv


if [ ! -f "$INIT_MARKER" ]; then
    echo "First-time initialization..."

    # export UV_PROJECT_ENVIRONMENT="/workspace"
    export VIRTUAL_ENV="/workspace/venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    # export UV_COMPILE_BYTECODE="true"

    uv venv $VIRTUAL_ENV --python $PYTHON_VERSION
    
    uv pip install ninja wheel setuptools
    uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    uv pip install /triton-3.2.0-cp312-cp312-linux_x86_64.whl
    uv pip install /sageattention-2.1.1-cp312-cp312-linux_x86_64.whl
    uv pip install /flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl
    uv pip install /transformers-4.49.0.dev0-py3-none-any.whl
    uv pip install -r /requirements.txt
     
    # Create marker file
    touch "$INIT_MARKER"
    echo "Initialization complete."
else
    echo "Container already initialized. Skipping first-time setup."
fi

if [ "$DOWNLOAD_MODELS" = "true" ]; then

    echo "DOWNLOAD_MODELS is true, downloading models..."
    mkdir -p "$MODELS_DIR"

    # Clone llava-llama-3-8b-text-encoder-tokenizer repository
    if [ ! -d "${MODELS_DIR}/llava-llama-3-8b-text-encoder-tokenizer" ]; then
        huggingface-cli download Kijai/llava-llama-3-8b-text-encoder-tokenizer --local-dir "${MODELS_DIR}/llava-llama-3-8b-text-encoder-tokenizer"
    else
        echo "Skipping the model llava-llama-3-8b-text-encoder-tokenizer download because it already exists."
    fi
    # Download hunyuan_video_720_cfgdistill_fp8_e4m3fn model
    if [ ! -f "${MODELS_DIR}/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" ]; then
        huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors --local-dir "${MODELS_DIR}"
    else
        echo "Skipping the model hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors download because it already exists."
    fi
    # Download hunyuan_video_720_cfgdistill_bf16 model
    if [ ! -f "${MODELS_DIR}/hunyuan_video_720_cfgdistill_bf16.safetensors" ] && [ "${DOWNLOAD_BF16}" == "true" ]; then
        huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_720_cfgdistill_bf16.safetensors --local-dir "${MODELS_DIR}"
    fi
    # Download hunyuan_video_vae_fp32 model
    if [ ! -f "${MODELS_DIR}/hunyuan_video_vae_fp32.safetensors" ]; then
        huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_vae_fp32.safetensors --local-dir "${MODELS_DIR}"
    else
        echo "Skipping the model hunyuan_video_vae_fp32.safetensors download because it already exists."
    fi
    # Download hunyuan_video_vae_fp16 model
    if [ ! -f "${MODELS_DIR}/hunyuan_video_vae_bf16.safetensors" ]; then
        huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_vae_bf16.safetensors --local-dir "${MODELS_DIR}"
    else
        echo "Skipping the model hunyuan_video_vae_bf16.safetensors download because it already exists."
    fi
    # Clone the entire CLIP repo
    if [ ! -d "${MODELS_DIR}/clip-vit-large-patch14" ]; then
        huggingface-cli download openai/clip-vit-large-patch14 --local-dir "${MODELS_DIR}/clip-vit-large-patch14"
    else
        echo "Skipping the model clip-vit-large-patch14 download because it already exists."
    fi
else
    echo "DOWNLOAD_MODELS is false, skipping model downloads."
fi

echo "Adding environmnent variables"

export PYTHONPATH="$REPO_DIR:$REPO_DIR/submodules/HunyuanVideo:$PYTHONPATH"
export PATH="$REPO_DIR/configs:$PATH"
export PATH="$REPO_DIR:$PATH"

echo $PATH
echo $PYTHONPATH

cd /workspace/diffusion-pipe

# Use conda python instead of system python
echo "Starting Gradio interface..."
uv run gradio_interface.py & 

# Use debugpy for debugging
# exec python -m debugpy --wait-for-client --listen 0.0.0.0:5678 gradio_interface.py

echo "Starting Tensorboard interface..."
uv run tensorboard --logdir_spec=/workspace/outputs --bind_all --port 6006 &

wait