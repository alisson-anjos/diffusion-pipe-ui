#!/bin/bash

# Marker file path
INIT_MARKER="/var/run/container_initialized"
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-"true"}  # Default if not set
DOWNLOAD_BF16=${DOWNLOAD_BF16:-"false"}  # Default if not set
REPO_DIR=${REPO_DIR:-"/workspace/diffusion-pipe"}

echo "DOWNLOAD_MODELS is: $DOWNLOAD_MODELS and DOWNLOAD_BF16 is: $DOWNLOAD_BF16"

source /opt/conda/etc/profile.d/conda.sh
conda activate pyenv



if [ ! -f "$INIT_MARKER" ]; then
    echo "First-time initialization..."

    echo "Installing CUDA nvcc..."
    conda install -y -c nvidia cuda-nvcc --override-channels

    echo "Installing dependencies from requirements.txt..."
    pip install --no-cache-dir -r $REPO_DIR/requirements.txt

    if [ "$DOWNLOAD_MODELS" = "true" ]; then
        echo "DOWNLOAD_MODELS is true, downloading models..."
        MODEL_DIR="/workspace/models"
        mkdir -p "$MODEL_DIR"

        # Clone llava-llama-3-8b-text-encoder-tokenizer repository
        if [ ! -d "${MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer" ]; then
            huggingface-cli download Kijai/llava-llama-3-8b-text-encoder-tokenizer --local-dir "${MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer"
            # git clone https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer "${MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer"
            # cd "${MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer"
            # git lfs pull
            # cd -
        else
            echo "Skipping the model llava-llama-3-8b-text-encoder-tokenizer download because it already exists."
        fi

        # Download hunyuan_video_720_cfgdistill_fp8_e4m3fn model
        if [ ! -f "${MODEL_DIR}/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" ]; then
            huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors --local-dir "${MODEL_DIR}"

            # curl -L -o "${MODEL_DIR}/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors?download=true"
        else
            echo "Skipping the model hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors download because it already exists."
        fi

        # Download hunyuan_video_720_cfgdistill_bf16 model
        if [ ! -f "${MODEL_DIR}/hunyuan_video_720_cfgdistill_bf16.safetensors" ] && [ "${DOWNLOAD_BF16}" == "true" ]; then
            huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_720_cfgdistill_bf16.safetensors --local-dir "${MODEL_DIR}"

            # curl -L -o "${MODEL_DIR}/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors?download=true"
        fi


        # Download hunyuan_video_vae_fp32 model
        if [ ! -f "${MODEL_DIR}/hunyuan_video_vae_fp32.safetensors" ]; then
            huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_vae_fp32.safetensors --local-dir "${MODEL_DIR}"
            # curl -L -o "${MODEL_DIR}/hunyuan_video_vae_fp32.safetensors" "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_vae_fp32.safetensors?download=true"
        else
            echo "Skipping the model hunyuan_video_vae_fp32.safetensors download because it already exists."
        fi

        # Download hunyuan_video_vae_fp16 model
        if [ ! -f "${MODEL_DIR}/hunyuan_video_vae_bf16.safetensors" ]; then
            huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_vae_bf16.safetensors --local-dir "${MODEL_DIR}"
            # curl -L -o "${MODEL_DIR}/hunyuan_video_vae_fp16.safetensors" "https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_vae_fp16.safetensors?download=true"
        else
            echo "Skipping the model hunyuan_video_vae_bf16.safetensors download because it already exists."
        fi

        # Clone the entire CLIP repo
        if [ ! -d "${MODEL_DIR}/clip-vit-large-patch14" ]; then
            huggingface-cli download openai/clip-vit-large-patch14 --local-dir "${MODEL_DIR}/clip-vit-large-patch14"
            # git clone https://huggingface.co/openai/clip-vit-large-patch14 "${MODEL_DIR}/clip-vit-large-patch14"
            # cd "${MODEL_DIR}/clip-vit-large-patch14"
            # git lfs pull
            # cd -
        else
            echo "Skipping the model clip-vit-large-patch14 download because it already exists."
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

echo "Adding environmnent variables"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/submodules/HunyuanVideo:$PYTHONPATH"
export PATH="$REPO_DIR/configs:$PATH"
export PATH="$REPO_DIR:$PATH"

echo $PATH
echo $PYTHONPATH

cd /workspace/diffusion-pipe

# Use conda python instead of system python
echo "Starting Gradio interface..."
python gradio_interface.py &

# Use debugpy for debugging
# exec python -m debugpy --wait-for-client --listen 0.0.0.0:5678 gradio_interface.py

echo "Starting Tensorboard interface..."
$CONDA_DIR/bin/conda run -n pyenv tensorboard --logdir_spec=/workspace/outputs --bind_all --port 6006 &

wait