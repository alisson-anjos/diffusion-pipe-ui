This repository is a fork of the original repository ([diffusion-pipe](https://github.com/tdrussell/diffusion-pipe)) but its goal is to provide a gradio interface as well as a docker image to facilitate training in any environment that supports docker (windows, linux) without any difficulty. You can also use the interface without using docker and you need to do the normal installation process that is in the original README and then instead of running train.py directly you can run gradio_interface.py, for now this fork is focused only on the Hunyuan model so a pull request is not made for the original repository (supports ltx, ... flux).


### Updates
- 2025-01-18 (gradio interface)
  - Now when saving the model instead of saving it with the name adapter_model.safetensors it will save the name of the dataset + the epoch.
  - Added 2 buttons to force saving the model and the checkpoint.

- 2025-01-17 (diffusion-pipe original)
  - For the HunyuanVideo VAE when loaded via the vae_path option, incorrect mosaic sample size fixed. Training loss is now moderately lower overall. The quality of trained LoRAs should be improved, but the improvement is likely small.
  - You should update any cached latencies made prior to this change. Either delete the cache directory within the dataset directories or run the training script with the --regenerate_cache command line option (this functionality is not yet added to the interface, but you can create a new dataset which is the same as running this command).

### Note

If you have any very complex issues regarding the interface open a issue or send me a message on my civitiai profile.

[My Profile](https://civitai.com/user/alissonerdx)

## Gradio Interface

![preview gradio](/preview-gradio.png)

### Features
- Docker Image
- Web UI (Gradio) for configuring and executing LoRA training.
- Optional NVIDIA GPU support for accelerated training.
- Ability to map model and output directories from the host system into the container.
- Optional automatic download of required models upon first initialization.
- Tensorboard to visualize training loss/epoch
- Jupyter Lab to manage files
- Wandb

### Improvements for the future
- If the page is updated during training, restore the training data as well as the log
- Generate samples between epochs to be able to visualize the influence of Lora.

### Prerequisites

- **Docker:**  
  - Install Docker for your platform by following the official documentation: [Get Docker](https://docs.docker.com/get-docker/)
  - [How to Install Docker on Windows](https://youtu.be/ZyBBv1JmnWQ)
  - [How to Install Docker on Ubuntu](https://www.youtube.com/watch?v=J4dZ2jcpiP0) 

- **GPU Support (optional):**  
  To utilize GPU acceleration (NVIDIA):
  - **Linux:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and ensure your NVIDIA drivers are set up.
  - **Windows/macOS:** Check respective Docker and NVIDIA documentation for GPU passthrough (e.g., WSL2 on Windows). If GPU support is not available, you can run the container without `--gpus all`.

### 

### How to Run using Docker (Windows, Linux)

#### Basic Run Command

```bash
docker run --gpus all -d -p 7860:7860 -p 8888:8888 -p 6006:6006 alissonpereiraanjos/diffusion-pipe-interface:latest
```

Note: the -d argument makes it run in detached mode, that is, it will run in the background. Now, if you want to see the log in the terminal where you ran the command, you can use -it instead of -d.

This command will download the models needed to perform the training and make the gradio interface available on port 7860, as well as Jupyter Lab UI on 8888 and Tensorboard on 6006. If you want to map the volumes for some specific reason, such as already having the models on your Windows/Linux, you can look at the section below that explains how to map the volumes and disable the download of the models.

- `--gpus all`: Enables GPU support if configured.  
- `-p 7860:7860`: Exposes port 7860 so you can access the Gradio UI at `http://localhost:7860`.
- `-p 8888:8888`: (optional) Exposes port 8888 so you can access the Jupyter Lab UI at `http://localhost:8888`.
- `-p 6006:6006`: (optional) Exposes port 6006 so you can access the Tensorboard and visualize your training loss at `http://localhost:6006`.

#### Mapping Directories for Models and Output

You can mount host directories to store models and training outputs outside the container:

```bash
docker run --gpus all -it \
  -v /path/to/models:/workspace/models \
  -v /path/to/outputs:/workspace/outputs \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/configs:/workspace/configs \
  -p 8888:8888 \
  -p 7860:7860 \
  -p 6006:6006 \
  alissonpereiraanjos/diffusion-pipe-interface:latest
```

- Replace `/path/to/models`, `/path/to/output`, `/path/to/datasets` and `/path/to/configs`  with your desired host directories.
- On Windows, for example:
```bash
docker run --gpus all -it \
  -v D:\AI\hunyuan\models:/workspace/models \
  -v D:\AI\hunyuan\outputs:/workspace/outputs \
  -v D:\AI\hunyuan\datasets:/workspace/datasets \
  -v D:\AI\hunyuan\configs:/workspace/configs \
  -p 8888:8888 \
  -p 7860:7860 \
  -p 6006:6006 \
  alissonpereiraanjos/diffusion-pipe-interface:latest
```

#### Controlling Model Downloads

By default, the container downloads the required models during the first initialization. If you already have the models in `/workspace/models` and want to skip automatic downloads, set the `DOWNLOAD_MODELS` environment variable to `false`:

```bash
docker run --gpus all -it \
  -v /path/to/models:/workspace/models \
  -v /path/to/outputs:/workspace/outputs \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/configs:/workspace/configs \
  -p 8888:8888 \
  -p 7860:7860 \
  -p 6006:6006 \
  -e DOWNLOAD_MODELS=false \
  alissonpereiraanjos/diffusion-pipe-interface:latest
```

#### Running in Detached Mode

If you prefer to run the container in the background without an interactive terminal, use `-d`:

```bash
docker run --gpus all -d \
  -v /path/to/models:/workspace/models \
  -v /path/to/outputs:/workspace/outputs \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/configs:/workspace/configs \
  -p 8888:8888 \
  -p 7860:7860 \
  -p 6006:6006 \
  -e DOWNLOAD_MODELS=false \
  alissonpereiraanjos/diffusion-pipe-interface:latest
```

Access the UI (gradio) at `http://localhost:7860`.
Access the Jupiter lab UI at `http://localhost:8888`.

#### Summary of Options

- `-v /host/path:/container/path`: Mount host directories into the container.
- `-p host_port:container_port`: Map container ports to host ports.
- `-e VARIABLE=value`: Set environment variables.
- `-e DOWNLOAD_MODELS=false`: Skips downloading models inside the container.
- `--gpus all`: Enables GPU support if available.
- `-it`: Start in interactive mode (useful for debugging).
- `-d`: Start in detached mode (runs in the background).

Use these options to tailor the setup to your environment and requirements.

## Update Image Docker (Important)

To update the docker image with the new changes, if you already have the image on your machine, you can run the command: 

`docker pull alissonpereiraanjos/diffusion-pipe-interface:latest`

This is very important so that you have all the updates, it is good to always update the image before running the container, this is for those who run in Docker locally, because when running through the runpod this is done every time you create a new pod.

### Running on RunPod

If you prefer to use [RunPod](https://runpod.io/), you can quickly deploy an instance based on this image by using the following template link:

[Deploy on RunPod](https://runpod.io/console/deploy?template=t46lnd7p4b&ref=8t518hht)

This link takes you to the RunPod console, allowing you to set up a machine directly with the provided image. Just configure your GPU, volume mounts, and environment variables as needed.

Tip: If you train often, I advise you to create a Network Volume in the runpod and use it in your pod, set the Volume to at least 100GB because then you will always have the models and data from your training in it and you will not waste your pod's time downloading the models every time.

![Network Volume Runpod](https://github.com/user-attachments/assets/32f7dc06-b7d1-4974-ac07-dce172c53c64)

### Running on Vast.ai

[Template](https://cloud.vast.ai/?ref_id=142589&creator_id=142589&name=Hunyuan%20Lora%20Train%20Simple%20Interface)


# Original README (diffusion-pipe)

Currently supports SDXL, Flux, LTX-Video, HunyuanVideo (t2v), Cosmos, Lumina Image 2.0, Wan2.1 (t2v)

**Work in progress and highly experimental.** It is unstable and not well tested. Things might not work right.

## Features
- Pipeline parallelism, for training models larger than can fit on a single GPU
- Useful metrics logged to Tensorboard
- Compute metrics on a held-out eval set, for measuring generalization
- Training state checkpointing and resuming from checkpoint
- Efficient multi-process, multi-GPU pre-caching of latents and text embeddings
- Easily add support for new models by implementing a single subclass

## Recent changes
- 2025-02-25
  - Support LoRA training on Wan2.1 t2v variants.
  - SDXL: debiased estimation loss, init from existing lora, and arbitrary caption length.
- 2025-02-16
  - SDXL supports separate learning rates for unet and text encoders. These are specified in the [model] table. See the supported models doc for details.
  - Added full fine tuning support for SDXL.
- 2025-02-10
  - Fixed a bug in video training causing width and height to be flipped when bucketing by aspect ratio. This would cause videos to be over-cropped. Image-only training is unaffected. If you have been training on videos, please pull the latest code, and regenerate the cache using the --regenerate_cache flag, or delete the cache dir inside the dataset directories.
- 2025-02-09
  - Add support for Lumina Image 2.0. Both LoRA and full fine tuning are supported.
- 2025-02-08
  - Support fp8 transformer for Flux LoRAs. You can now train LoRAs with a single 24GB GPU.
  - Add tentative support for Cosmos. Cosmos doesn't fine tune well compared to HunyuanVideo, and will likely not be actively supported going forward.
- 2025-01-20
  - Properly support training Flex.1-alpha.
  - Make sure to set ```bypass_guidance_embedding=true``` in the model config. You can look at the example config file.
- 2025-01-17
  - For HunyuanVideo VAE when loaded via the ```vae_path``` option, fixed incorrect tiling sample size. The training loss is now moderately lower overall. Quality of trained LoRAs should be improved, but the improvement is likely minor.
  - You should update any cached latents made before this change. Delete the cache directory inside the dataset directories, or run the training script with the ```--regenerate_cache``` command line option.

## Windows support
It will be difficult or impossible to make training work on native Windows. This is because Deepspeed only has [partial Windows support](https://github.com/microsoft/DeepSpeed/blob/master/blogs/windows/08-2024/README.md). Deepspeed is a hard requirement because the entire training script is built around Deepspeed pipeline parallelism. However, it will work on Windows Subsystem for Linux, specifically WSL 2. If you must use Windows I recommend trying WSL 2.

## Installing
Clone the repository:
```
git clone --recurse-submodules https://github.com/tdrussell/diffusion-pipe
```

If you alread cloned it and forgot to do --recurse-submodules:
```
git submodule init
git submodule update
```

Install Miniconda: https://docs.anaconda.com/miniconda/

Create the environment:
```
conda create -n diffusion-pipe python=3.12
conda activate diffusion-pipe
```

Install nvcc: https://anaconda.org/nvidia/cuda-nvcc. Probably try to make it match the CUDA version that was installed on your system with PyTorch.

Install the dependencies:
```
pip install -r requirements.txt
```

### Cosmos requirements
NVIDIA Cosmos additionally requires TransformerEngine. This dependency isn't in the requirements file. Installing this was a bit tricky for me. On Ubuntu 24.04, I had to install GCC version 12 (13 is the default in the package manager), and make sure GCC 12 and CUDNN were set during installation like this:
```
CC=/usr/bin/gcc-12 CUDNN_PATH=/home/anon/miniconda3/envs/diffusion-pipe/lib/python3.12/site-packages/nvidia/cudnn pip install transformer_engine[pytorch]
```

## Dataset preparation
A dataset consists of one or more directories containing image or video files, and corresponding captions. You can mix images and videos in the same directory, but it's probably a good idea to separate them in case you need to specify certain settings on a per-directory basis. Caption files should be .txt files with the same base name as the corresponding media file, e.g. image1.png should have caption file image1.txt in the same directory. If a media file doesn't have a matching caption file, a warning is printed, but training will proceed with an empty caption.

For images, any image format that can be loaded by Pillow should work. For videos, any format that can be loaded by ImageIO should work. Note that this means **WebP videos are not supported**, because ImageIO can't load multi-frame WebPs.

## Supported models
See the [supported models doc](./docs/supported_models.md) for more information on how to configure each model, the options it supports, and the format of the saved LoRAs.

## Training
**Start by reading through the config files in the examples directory.** Almost everything is commented, explaining what each setting does.

Once you've familiarized yourself with the config file format, go ahead and make a copy and edit to your liking. At minimum, change all the paths to conform to your setup, including the paths in the dataset config file.

Launch training like this:
```
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/hunyuan_video.toml
```
RTX 4000 series needs those 2 environment variables set. Other GPUs may not need them. You can try without them, Deepspeed will complain if it's wrong.

If you enabled checkpointing, you can resume training from the latest checkpoint by simply re-running the exact same command but with the `--resume_from_checkpoint` flag. You can also specify a specific checkpoint folder name after the flag to resume from that particular checkpoint (e.g. `--resume_from_checkpoint "20250212_07-06-40"`). This option is particularly useful if you have run multiple training sessions with different datasets and want to resume from a specific training folder.

Please note that resuming from checkpoint uses the **config file on the command line**, not the config file saved into the output directory. You are responsible for making sure that the config file you pass in matches what was previously used.

## Output files
A new directory will be created in ```output_dir``` for each training run. This contains the checkpoints, saved models, and Tensorboard metrics. Saved models/LoRAs will be in directories named like epoch1, epoch2, etc. Deepspeed checkpoints are in directories named like global_step1234. These checkpoints contain all training state, including weights, optimizer, and dataloader state, but can't be used directly for inference. The saved model directory will have the safetensors weights, PEFT adapter config JSON, as well as the diffusion-pipe config file for easier tracking of training run settings.

## Parallelism
This code uses hybrid data- and pipeline-parallelism. Set the ```--num_gpus``` flag appropriately for your setup. Set ```pipeline_stages``` in the config file to control the degree of pipeline parallelism. Then the data parallelism degree will automatically be set to use all GPUs (number of GPUs must be divisible by pipeline_stages). For example, with 4 GPUs and pipeline_stages=2, you will run two instances of the model, each divided across two GPUs.

## Pre-caching
Latents and text embeddings are cached to disk before training happens. This way, the VAE and text encoders don't need to be kept loaded during training. The Huggingface Datasets library is used for all the caching. Cache files are reused between training runs if they exist. All cache files are written into a directory named "cache" inside each dataset directory.

This caching also means that training LoRAs for text encoders is not currently supported.

Two flags are relevant for caching. ```--cache_only``` does the caching flow, then exits without training anything. ```--regenerate_cache``` forces cache regeneration. If you edit the dataset in-place (like changing a caption), you need to force regenerate the cache (or delete the cache dir) for the changes to be picked up.

## Extra
You can check out my [qlora-pipe](https://github.com/tdrussell/qlora-pipe) project, which is basically the same thing as this but for LLMs.
