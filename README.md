# Diffusion-Pipe-UI

This repository provides a LoRA Training Interface for the Hunyuan Video model using [Gradio](https://gradio.app/). The UI was initially created with the assistance of ChatGPT and may contain various issues or rough edges. Contributions, bug reports, and improvements are welcome!

The core training logic is based on the [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) project, which provides the underlying functionality and scripts.

## Features

- Web UI (Gradio) for configuring and executing LoRA training.
- Optional NVIDIA GPU support for accelerated training.
- Ability to map model and output directories from the host system into the container.
- Optional automatic download of required models upon first initialization.

## Prerequisites

- **Docker:**  
  Install Docker for your platform by following the official documentation:  
  [Get Docker](https://docs.docker.com/get-docker/)

- **GPU Support (optional):**  
  To utilize GPU acceleration (NVIDIA):
  - **Linux:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and ensure your NVIDIA drivers are set up.
  - **Windows/macOS:** Check respective Docker and NVIDIA documentation for GPU passthrough (e.g., WSL2 on Windows). If GPU support is not available, you can run the container without `--gpus all`.

## How to Run

### Basic Run Command

```bash
docker run --gpus all -it -p 7860:7860 alissonpereiraanjos/diffusion-pipe-ui:latest
```

- `--gpus all`: Enables GPU support if configured.  
- `-p 7860:7860`: Exposes port 7860 so you can access the Gradio UI at `http://localhost:7860`.

If you do not have or do not want GPU support, omit `--gpus all`.

### Mapping Directories for Models and Output

You can mount host directories to store models and training outputs outside the container:

```bash
docker run --gpus all -it \
  -v /path/to/models:/models \
  -v /path/to/output:/output \
  -v /path/to/datasets:/datasets \
  -v /path/to/config_history:/config_history \
  -p 7860:7860 \
  alissonpereiraanjos/diffusion-pipe-ui:latest
```

- Replace `/path/to/models` and `/path/to/output` with your desired host directories.
- On Windows, for example:
  ```bash
  docker run --gpus all -it \
    -v D:\AI\hunyuan\models:/models \
    -v D:\AI\hunyuan\output:/output \
    -p 7860:7860 \
    alissonpereiraanjos/diffusion-pipe-ui:latest
  ```

### Controlling Model Downloads

By default, the container downloads the required models during the first initialization. If you already have the models in `/models` and want to skip automatic downloads, set the `DOWNLOAD_MODELS` environment variable to `false`:

```bash
docker run --gpus all -it \
  -v /path/to/models:/models \
  -v /path/to/output:/output \
  -v /path/to/datasets:/datasets \
  -v /path/to/config_history:/config_history \
  -p 7860:7860 \
  -e DOWNLOAD_MODELS=false \
  alissonpereiraanjos/diffusion-pipe-ui:latest
```

### Running in Detached Mode

If you prefer to run the container in the background without an interactive terminal, use `-d`:

```bash
docker run --gpus all -d \
  -v /path/to/models:/models \
  -v /path/to/output:/output \
  -v /path/to/datasets:/datasets \
  -v /path/to/config_history:/config_history \
  -p 7860:7860 \
  -e DOWNLOAD_MODELS=false \
  alissonpereiraanjos/diffusion-pipe-ui:latest
```

Access the UI at `http://localhost:7860`.

## Running on RunPod

If you prefer to use [RunPod](https://runpod.io/), you can quickly deploy an instance based on this image by using the following template link:

[Deploy on RunPod](https://runpod.io/console/deploy?template=t46lnd7p4b&ref=8t518hht)

This link takes you to the RunPod console, allowing you to set up a machine directly with the provided image. Just configure your GPU, volume mounts, and environment variables as needed.

## Platform Notes

- **Windows:**  
  Ensure that Docker Desktop can access the drive (e.g., `D:\`). For GPU support, use WSL2 and follow NVIDIA/Docker instructions for GPU acceleration in WSL2.

- **Linux:**  
  You can map any directory. For GPU support, install the NVIDIA Container Toolkit.

- **macOS:**  
  Native GPU passthrough is not supported. You may run the container without GPU acceleration.

## Accessing the Web UI

Once the container is running, open your browser at:

```
http://localhost:7860
```

You can upload datasets, configure training parameters, start training, and download results and configurations directly from the interface.

## Summary of Options

- `-v /host/path:/container/path`: Mount host directories into the container.
- `-p host_port:container_port`: Map container ports to host ports.
- `-e VARIABLE=value`: Set environment variables.
  - `DOWNLOAD_MODELS=false`: Skips downloading models inside the container.
- `--gpus all`: Enables GPU support if available.
- `-it`: Start in interactive mode (useful for debugging).
- `-d`: Start in detached mode (runs in the background).

Use these options to tailor the setup to your environment and requirements.

## Contributions

The UI, initially created with ChatGPT's assistance, is not perfect and may contain various errors or rough parts. Contributions and improvements are welcome! If you encounter issues, feel free to open an issue or submit a pull request.

## References

- [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe): The core repository providing the underlying training logic and scripts used by this UI.