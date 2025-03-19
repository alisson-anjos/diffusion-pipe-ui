#!/bin/bash

# Marker file path
INIT_MARKER="/var/run/container_initialized"
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-"true"}    # Default if not set
DOWNLOAD_BF16=${DOWNLOAD_BF16:-"false"}         # Default if not set
REPO_DIR=${REPO_DIR:-"/workspace/diffusion-pipe"}
MODELS_DIR="/workspace/models"

echo "DOWNLOAD_MODELS is: $DOWNLOAD_MODELS and DOWNLOAD_BF16 is: $DOWNLOAD_BF16"

# source /opt/conda/etc/profile.d/conda.sh
# conda activate pyenv

if [ ! -f "$INIT_MARKER" ]; then
    echo "First-time initialization..."

    # Setup virtual environment
    export VIRTUAL_ENV="/workspace/venv"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    
    uv venv $VIRTUAL_ENV --python $PYTHON_VERSION
    
    uv pip install ninja wheel setuptools
    # uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
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
    echo "DOWNLOAD_MODELS is true, processing model downloads..."
    mkdir -p "$MODELS_DIR"
    
    if [ -z "$MODEL_GROUPS" ]; then
        echo "MODEL_GROUPS is not set. Skipping model downloads."
    else
        # Process specified groups
        IFS=',' read -ra GROUPS_ARRAY <<< "$MODEL_GROUPS"
        for group in "${GROUPS_ARRAY[@]}"; do
            group=$(echo "$group" | xargs)  # trim spaces
            echo "Processing model group: $group"
            case "$group" in
                "wan")
                    # If no specific variant is defined, download all variants
                    if [ -z "$WAN_VARIANTS" ]; then
                        WAN_VARIANTS="I2V480P,I2V720P,T2V480P-1.3B,T2V480P-14B"
                    fi
                    IFS=',' read -ra VARIANTS <<< "$WAN_VARIANTS"
                    for variant in "${VARIANTS[@]}"; do
                        variant=$(echo "$variant" | xargs)
                        echo "  Downloading WAN variant: $variant"
                        case "$variant" in
                            "I2V480P")
                                if [ ! -d "${MODELS_DIR}/wan/I2V-480P" ]; then
                                    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "${MODELS_DIR}/wan/I2V-480P"
                                else
                                    echo "    Skipping Wan-AI/Wan2.1-I2V-14B-480P; already exists."
                                fi
                                ;;
                            "I2V720P")
                                if [ ! -d "${MODELS_DIR}/wan/I2V-720P" ]; then
                                    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir "${MODELS_DIR}/wan/I2V-720P"
                                else
                                    echo "    Skipping Wan-AI/Wan2.1-I2V-14B-720P; already exists."
                                fi
                                ;;
                            "T2V480P-1.3B")
                                if [ ! -d "${MODELS_DIR}/wan/T2V-480P-1.3B" ]; then
                                    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir "${MODELS_DIR}/wan/T2V-480P-1.3B"
                                else
                                    echo "    Skipping Wan-AI/Wan2.1-T2V-1.3B; already exists."
                                fi
                                ;;
                            "T2V480P-14B")
                                if [ ! -d "${MODELS_DIR}/wan/T2V-480P-14B" ]; then
                                    huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir "${MODELS_DIR}/wan/T2V-480P-14B"
                                else
                                    echo "    Skipping Wan-AI/Wan2.1-T2V-14B; already exists."
                                fi
                                ;;
                            *)
                                echo "    Unknown WAN variant: $variant. Skipping."
                                ;;
                        esac
                    done
                    ;;
                "hunyuan")
                    if [ ! -f "${MODELS_DIR}/hunyuan/t2v/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors" ]; then
                        huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors --local-dir "${MODELS_DIR}/hunyuan/t2v/"
                    else
                        echo "  Skipping hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors; already exists."
                    fi
                    if [ ! -f "${MODELS_DIR}/hunyuan/t2v/hunyuan_video_720_cfgdistill_bf16.safetensors" ] && [ "$DOWNLOAD_BF16" = "true" ]; then
                        huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_720_cfgdistill_bf16.safetensors --local-dir "${MODELS_DIR}/hunyuan/t2v/"
                    else
                        echo "  Skipping hunyuan_video_720_cfgdistill_bf16.safetensors; already exists or DOWNLOAD_BF16 is false."
                    fi
                    if [ ! -f "${MODELS_DIR}/hunyuan/t2v/hunyuan_video_vae_fp32.safetensors" ]; then
                        huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_vae_fp32.safetensors --local-dir "${MODELS_DIR}/hunyuan/t2v/"
                    else
                        echo "  Skipping hunyuan_video_vae_fp32.safetensors; already exists."
                    fi
                    if [ ! -f "${MODELS_DIR}/hunyuan/t2v/hunyuan_video_vae_bf16.safetensors" ]; then
                        huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_vae_bf16.safetensors --local-dir "${MODELS_DIR}/hunyuan/t2v/"
                    else
                        echo "  Skipping hunyuan_video_vae_bf16.safetensors; already exists."
                    fi
                    ;;
                "flux")
                    # Download FLUX diffusers model
                    if [ ! -d "${MODELS_DIR}/flux/diffusers" ]; then
                        huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir "${MODELS_DIR}/flux/diffusers"
                    else
                        echo "  Skipping FLUX.1-dev diffusers; already exists."
                    fi
                    # Download FLUX transformer model override
                    if [ ! -f "${MODELS_DIR}/flux/transformer/flux1-schnell.safetensors" ]; then
                        huggingface-cli download black-forest-labs/FLUX.1-schnell flux1-schnell.safetensors --local-dir "${MODELS_DIR}/flux/transformer"
                    else
                        echo "  Skipping FLUX.1-schnell transformer; already exists."
                    fi
                    ;;
                "ltx")
                    if [ ! -d "${MODELS_DIR}/ltx" ]; then
                        huggingface-cli download Lightricks/LTX-Video --local-dir "${MODELS_DIR}/ltx"
                    else
                        echo "  Skipping LTX-Video; already exists."
                    fi
                    ;;
                "cosmos")
                    # Download cosmos diffusers model
                    if [ ! -d "${MODELS_DIR}/cosmos/diffusers" ]; then
                        huggingface-cli download nvidia/Cosmos-1.0-Diffusion-7B-Text2World --local-dir "${MODELS_DIR}/cosmos/diffusers"
                    else
                        echo "  Skipping cosmos diffusers model; already exists."
                    fi
                    # Download cosmos VAE checkpoint
                    if [ ! -f "${MODELS_DIR}/cosmos/vae/cosmos_cv8x8x8_1.0.safetensors" ]; then
                        huggingface-cli download comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI cosmos_cv8x8x8_1.0.safetensors --local-dir "${MODELS_DIR}/cosmos/vae"
                    else
                        echo "  Skipping cosmos VAE; already exists."
                    fi
                    # Download cosmos flux text encoder checkpoint
                    if [ ! -f "${MODELS_DIR}/cosmos/text_encoder/t5xxl_fp16.safetensors" ]; then
                        huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir "${MODELS_DIR}/cosmos/text_encoder"
                    else
                        echo "  Skipping cosmos text encoder; already exists."
                    fi
                    ;;
                "chroma")
                    if [ ! -d "${MODELS_DIR}/chroma" ]; then
                        huggingface-cli download lodestones/Chroma --local-dir "${MODELS_DIR}/chroma"
                    else
                        echo "  Skipping Chroma; already exists."
                    fi
                    ;;
                "lumina")
                    if [ ! -d "${MODELS_DIR}/lumina" ]; then
                        huggingface-cli download Comfy-Org/Lumina_Image_2.0_Repackaged --local-dir "${MODELS_DIR}/lumina"
                    else
                        echo "  Skipping Lumina; already exists."
                    fi
                    ;;
                "sdxl")
                    if [ ! -f "${MODELS_DIR}/sdxl/sd_xl_base_1.0_0.9vae.safetensors" ]; then
                        huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0_0.9vae.safetensors --local-dir "${MODELS_DIR}/sdxl"
                    else
                        echo "  Skipping SDXL model; already exists."
                    fi
                    ;;
                *)
                    echo "Unknown model group: $group. Skipping."
                    ;;
            esac
        done
    fi
else
    echo "DOWNLOAD_MODELS is false, skipping model downloads."
fi

echo "Adding environment variables"

export PYTHONPATH="$REPO_DIR:$REPO_DIR/submodules/HunyuanVideo:$PYTHONPATH"
export PATH="$REPO_DIR/configs:$PATH"
export PATH="$REPO_DIR:$PATH"

echo "PATH is: $PATH"
echo "PYTHONPATH is: $PYTHONPATH"

cd /workspace/diffusion-pipe

# Decide which UI to launch based on UI_TYPE
UI_TYPE=${UI_TYPE:-"blazor"}  # default to gradio if not set

case "$UI_TYPE" in
    "gradio")
        echo "Starting Gradio interface..."
        uv run gradio_interface.py &
        ;;
    "blazor")
        echo "Starting Blazor UI..."
        cd $REPO_DIR/ui/deploy
        dotnet DiffusionPipeInterface.dll --urls http://0.0.0.0:5000 &
        cd $REPO_DIR
        ;;
    *)
        echo "Unknown UI_TYPE: $UI_TYPE. No UI will be started."
        ;;
esac

echo "Starting Tensorboard interface..."
uv run tensorboard --logdir_spec=/workspace/outputs --bind_all --port 6006 &

wait
