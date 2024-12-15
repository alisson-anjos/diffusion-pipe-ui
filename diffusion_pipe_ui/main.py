#!/usr/bin/env python3

import gradio as gr
import os
import shutil
import subprocess
import toml
from datetime import datetime
import threading
import zipfile
import signal
import psutil
import json

# -----------------------------
# Configuration and Constants
# -----------------------------

# Working directories
MODEL_DIR = "/workspace/models"
BASE_DATASET_DIR = "/workspace/datasets"
OUTPUT_DIR = "/workspace/output"
CONFIG_HISTORY_DIR = "/workspace/config_history"

# Create directories if they don't exist
os.makedirs(CONFIG_HISTORY_DIR, exist_ok=True)
os.makedirs(BASE_DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum number of media to display in the gallery
MAX_MEDIA = 50

# Determine if running on Runpod by checking the environment variable
IS_RUNPOD = os.getenv("IS_RUNPOD", "false").lower() == "true"

# Maximum upload size in bytes
MAX_UPLOAD_SIZE = 500 * 1024 * 1024 if IS_RUNPOD else 2 * 1024 * 1024 * 1024  # 500MB or 2GB

# -----------------------------
# Training Process Management
# -----------------------------

training_process = None
training_lock = threading.Lock()

def kill_child_processes(parent_pid):
    """Terminate all child processes of the given parent process."""
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        gone, alive = psutil.wait_procs(children, timeout=5)
        parent.wait(5)
    except psutil.NoSuchProcess:
        pass

def generate_unique_filename(base_name):
    """Generate a unique filename based on the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.toml"

def create_dataset_config(dataset_path, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets):
    """Create and save the dataset configuration in TOML format."""
    dataset_config = {
        "resolutions": resolutions,  # Uses the provided list of resolutions
        "enable_ar_bucket": enable_ar_bucket,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "frame_buckets": frame_buckets,  # Uses the provided frame_buckets
        "directory": [
            {
                "path": dataset_path,
                "num_repeats": num_repeats
            }
        ]
    }
    dataset_file = generate_unique_filename("dataset_auto")
    dataset_path_full = os.path.join(CONFIG_HISTORY_DIR, dataset_file)
    with open(dataset_path_full, "w") as f:
        toml.dump(dataset_config, f)
    return dataset_path_full

def create_training_config(output_dir, dataset_path, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
                           transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
                           gradient_accumulation_steps, gradient_clipping, warmup_steps, eval_before_first_step,
                           eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps, checkpoint_every_n_minutes,
                           activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
                           video_clip_mode):
    """Create and save the training configuration in TOML format."""
    training_config = {
        "output_dir": output_dir,
        "dataset": dataset_path,
        "epochs": epochs,
        "micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "warmup_steps": warmup_steps,
        "eval_every_n_epochs": eval_every,
        "eval_before_first_step": eval_before_first_step,
        "eval_micro_batch_size_per_gpu": eval_micro_batch_size_per_gpu,
        "eval_gradient_accumulation_steps": eval_gradient_accumulation_steps,
        "save_every_n_epochs": save_every,
        "checkpoint_every_n_minutes": checkpoint_every_n_minutes,
        "activation_checkpointing": activation_checkpointing,
        "partition_method": partition_method,
        "save_dtype": save_dtype,
        "caching_batch_size": caching_batch_size,
        "steps_per_print": steps_per_print,
        "video_clip_mode": video_clip_mode,
        "model": {
            "type": "hunyuan-video",
            "transformer_path": transformer_path,
            "vae_path": vae_path,
            "llm_path": llm_path,
            "clip_path": clip_path,
            "dtype": dtype,
            "transformer_dtype": "float8",
            "timestep_sample_method": "logit_normal"
        },
        "adapter": {
            "type": "lora",
            "rank": rank,
            "dtype": dtype
        },
        "optimizer": {
            "type": optimizer_type,
            "lr": lr,
            "betas": eval(betas),
            "weight_decay": weight_decay,
            "eps": eps
        }
    }
    training_file = generate_unique_filename("hunyuan_video_auto")
    training_path_full = os.path.join(CONFIG_HISTORY_DIR, training_file)
    with open(training_path_full, "w") as f:
        toml.dump(training_config, f)
    return training_path_full

def stop_training():
    """Stop the training process and all child processes."""
    global training_process
    with training_lock:
        if training_process is not None:
            try:
                # Attempt graceful termination
                training_process.terminate()
                try:
                    training_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if not terminated
                    training_process.kill()
                    training_process.wait(timeout=5)
                # Terminate child processes
                kill_child_processes(training_process.pid)
                training_process = None
                return "Training process and all child processes have been stopped."
            except Exception as e:
                return f"Error stopping training: {str(e)}"
        return "No training process is running."

def train_lora(dataset_path, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
               transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
               gradient_accumulation_steps, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets,
               gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
               checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
               video_clip_mode):
    """Run the training command and stream logs."""
    global training_process
    with training_lock:
        if training_process is not None:
            yield "Training is already in progress."
            return

        # Create configurations
        dataset_config_path = create_dataset_config(
            dataset_path, 
            num_repeats, 
            resolutions, 
            enable_ar_bucket, 
            min_ar, 
            max_ar, 
            num_ar_buckets, 
            frame_buckets
        )
        training_config_path = create_training_config(
            output_dir, 
            dataset_config_path, 
            epochs, 
            batch_size, 
            lr, 
            save_every, 
            eval_every, 
            rank, 
            dtype,
            transformer_path, 
            vae_path, 
            llm_path, 
            clip_path, 
            optimizer_type, 
            betas, 
            weight_decay, 
            eps,
            gradient_accumulation_steps,
            gradient_clipping,
            warmup_steps,
            eval_before_first_step,
            eval_micro_batch_size_per_gpu,
            eval_gradient_accumulation_steps,
            checkpoint_every_n_minutes,
            activation_checkpointing,
            partition_method,
            save_dtype,
            caching_batch_size,
            steps_per_print,
            video_clip_mode
        )

        # Training command (replace with the actual command)
        conda_activate_path = "/opt/conda/etc/profile.d/conda.sh"
        conda_env_name = "pyenv"

        command = (
            f"bash -c 'source {conda_activate_path} && "
            f"conda activate {conda_env_name} && "
            f"cd /workspace/diffusion-pipe && "
            f"NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 "
            f"train.py --deepspeed --config {training_config_path}'"
        )

        # Start the training process
        training_process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True  # Start the process in a new session
        )

    # Stream the logs
    try:
        while True:
            output_line = training_process.stdout.readline()
            error_line = training_process.stderr.readline()
            if output_line:
                yield output_line.strip()
            if error_line:
                yield error_line.strip()
            if output_line == "" and error_line == "" and training_process.poll() is not None:
                break
    except Exception as e:
        yield f"Error during training: {str(e)}"
    finally:
        with training_lock:
            training_process = None
            yield "Training process has finished."

# -----------------------------
# Dataset Upload and Display
# -----------------------------

def get_existing_datasets():
    """Retrieve a list of existing datasets."""
    datasets = [d for d in os.listdir(BASE_DATASET_DIR) if os.path.isdir(os.path.join(BASE_DATASET_DIR, d))]
    return datasets

def upload_dataset(files, current_dataset, action):
    """
    Handle uploaded dataset files and store them in a unique directory.
    Action can be 'add' (add files to current dataset) or 'finalize' (finalize the dataset).
    """
    if not files:
        return current_dataset, "No files uploaded."

    if action == "start":
        # Start a new dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = os.path.join(BASE_DATASET_DIR, f"dataset_{timestamp}")
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir, f"Started new dataset: {dataset_dir}"
    
    if not current_dataset:
        return current_dataset, "Please start a new dataset before uploading files."

    # Calculate the total size of the current dataset
    total_size = 0
    for root, dirs, files_in_dir in os.walk(current_dataset):
        for f in files_in_dir:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)

    # Calculate the size of the new files
    new_files_size = 0
    for file in files:
        new_files_size += os.path.getsize(file.name)

    # Check if adding these files would exceed the limit
    if IS_RUNPOD and (total_size + new_files_size) > MAX_UPLOAD_SIZE:
        return current_dataset, f"Upload would exceed the 500MB limit on Runpod. Please upload smaller files or finalize the dataset."

    uploaded_files = []

    for file in files:
        file_path = file.name
        filename = os.path.basename(file_path)
        dest_path = os.path.join(current_dataset, filename)

        if zipfile.is_zipfile(file_path):
            # If the file is a ZIP, extract its contents
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(current_dataset)
                uploaded_files.append(f"{filename} (extracted)")
            except zipfile.BadZipFile:
                uploaded_files.append(f"{filename} (invalid ZIP)")
                continue
        else:
            # Check if the file is a supported format
            if filename.lower().endswith(('.mp4', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.txt')):
                shutil.copy(file_path, dest_path)
                uploaded_files.append(filename)
            else:
                uploaded_files.append(f"{filename} (unsupported format)")
                continue

    return current_dataset, f"Uploaded files: {', '.join(uploaded_files)}"

def finalize_dataset(current_dataset):
    """Finalize the dataset creation."""
    if not current_dataset or not os.path.exists(current_dataset):
        return "No dataset to finalize.", current_dataset
    # Here you can add any finalization steps if needed
    return f"Dataset {current_dataset} has been finalized.", current_dataset

def show_media(dataset_dir):
    """Display uploaded images and .mp4 videos in a single gallery."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        # Return an empty list if the dataset_dir is invalid
        return []

    # List of image and .mp4 video files
    media_files = [
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.mp4'))
    ]

    media_paths = [os.path.join(dataset_dir, f) for f in media_files[:MAX_MEDIA]]

    # Return the list of media paths for gr.Gallery
    return media_paths

# -----------------------------
# Download Functions
# -----------------------------

def download_output_zip():
    """Create a zip file with the training outputs for download."""
    zip_filename = "output.zip"  # Relative path; placed in the current working directory
    zip_path = os.path.join(os.getcwd(), zip_filename)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, OUTPUT_DIR)
                zf.write(filepath, arcname)
    return zip_path

def download_dataset_config_zip(dataset_dir):
    """Create a zip file with the dataset and configurations for download."""
    zip_filename = "dataset_configs.zip"  # Relative path; placed in the current working directory
    zip_path = os.path.join(os.getcwd(), zip_filename)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if dataset_dir and os.path.exists(dataset_dir):
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, os.path.dirname(dataset_dir))
                    zf.write(filepath, arcname)
        for root, dirs, files in os.walk(CONFIG_HISTORY_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.join("config_history", os.path.relpath(filepath, CONFIG_HISTORY_DIR))
                zf.write(filepath, arcname)
    return zip_path

def download_dataset_action(dataset_dir, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets_input):
    """Action to download the dataset and configurations."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        return None  # Gradio will handle this as no file to download
    if not num_repeats:
        num_repeats = 10
    try:
        # Parse resolutions
        resolutions = json.loads(resolutions_input)
        if not isinstance(resolutions, list) or not all(isinstance(i, int) for i in resolutions):
            raise ValueError
    except:
        # If parsing fails, use default value or return error
        resolutions = [512]  # Default value
    try:
        # Parse frame_buckets
        frame_buckets = json.loads(frame_buckets_input)
        if not isinstance(frame_buckets, list) or not all(isinstance(i, int) for i in frame_buckets):
            raise ValueError
    except:
        # If parsing fails, use default value or return error
        frame_buckets = [1, 33, 65]  # Default value
    create_dataset_config(dataset_dir, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets)
    return download_dataset_config_zip(dataset_dir)

# -----------------------------
# Gradio Interface Construction
# -----------------------------

def build_interface():
    """Build the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# LoRA Training Interface for Hunyuan Video")

        # 1. Step 1: Dataset Management
        gr.Markdown("### Step 1: Dataset Management\nChoose to create a new dataset or select an existing one.")

        with gr.Row():
            dataset_option = gr.Radio(
                choices=["Create New Dataset", "Select Existing Dataset"],
                value="Create New Dataset",
                label="Dataset Option"
            )

        # Container for dataset creation
        with gr.Row(elem_id="create_new_dataset_container"):
            with gr.Column():
                start_dataset_button = gr.Button("Start New Dataset")
                upload_files = gr.File(
                    label="Upload Images (.jpg, .png, .gif, .bmp, .webp), Videos (.mp4), Captions (.txt) or a ZIP archive",
                    file_types=[".jpg", ".png", ".gif", ".bmp", ".webp", ".mp4", ".txt", ".zip"],
                    file_count="multiple",
                    type="file",
                    interactive=True
                )
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                finalize_button = gr.Button("Finalize Dataset")
                dataset_path_display = gr.Textbox(label="Current Dataset Path", interactive=False)

        # Container for selecting existing dataset
        with gr.Row(elem_id="select_existing_dataset_container"):
            with gr.Column():
                existing_datasets = gr.Dropdown(
                    choices=get_existing_datasets(),
                    label="Select Existing Dataset",
                    interactive=True
                )
                selected_dataset_display = gr.Textbox(label="Selected Dataset Path", interactive=False)

        # Update visibility based on dataset_option
        def toggle_dataset_option(option):
            if option == "Create New Dataset":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        dataset_option.change(
            fn=toggle_dataset_option,
            inputs=dataset_option,
            outputs=[gr.components.Component.update(), gr.components.Component.update()]
        )

        # Initialize visibility
        toggle_dataset_option(dataset_option.value)

        # Logic to handle dataset creation
        def handle_start_dataset():
            return upload_dataset(None, None, "start")

        def handle_upload(files, current_dataset):
            return upload_dataset(files, current_dataset, "add")

        def handle_finalize(current_dataset):
            message, dataset = finalize_dataset(current_dataset)
            return message, dataset

        # State to keep track of the current dataset being created
        current_dataset_state = gr.State(None)

        # Start New Dataset
        start_dataset_button.click(
            fn=lambda: handle_start_dataset(),
            inputs=None,
            outputs=[dataset_path_display, upload_status],
            _js=None
        )

        # Upload Files
        upload_files.upload(
            fn=lambda files, current_dataset: handle_upload(files, current_dataset),
            inputs=[upload_files, current_dataset_state],
            outputs=[dataset_path_display, upload_status],
            queue=True
        )

        # Finalize Dataset
        finalize_button.click(
            fn=lambda current_dataset: handle_finalize(current_dataset),
            inputs=current_dataset_state,
            outputs=[upload_status, current_dataset_state]
        )

        # Select Existing Dataset
        def handle_select_existing(selected_dataset):
            if selected_dataset:
                return selected_dataset
            return ""

        existing_datasets.change(
            fn=handle_select_existing,
            inputs=existing_datasets,
            outputs=selected_dataset_display
        )

        # 2. Media Gallery
        gr.Markdown("### Media Gallery")
        gallery = gr.Gallery(
            label="Uploaded Media",
            show_label=False,
            elem_id="gallery",
            columns=3,
            rows=2,
            object_fit="contain",
            height="auto"
        )

        # Display media in gallery based on dataset selection
        def update_gallery(option, new_dataset, selected_dataset):
            if option == "Create New Dataset" and new_dataset:
                return show_media(new_dataset)
            elif option == "Select Existing Dataset" and selected_dataset:
                return show_media(selected_dataset)
            return []

        dataset_option.change(
            fn=update_gallery,
            inputs=[dataset_option, dataset_path_display, selected_dataset_display],
            outputs=gallery
        )

        dataset_path_display.change(
            fn=lambda path: show_media(path),
            inputs=dataset_path_display,
            outputs=gallery
        )

        selected_dataset_display.change(
            fn=lambda path: show_media(path),
            inputs=selected_dataset_display,
            outputs=gallery
        )

        # 3. Step 2: Training
        gr.Markdown("### Step 2: Training\nConfigure your training parameters and start or stop the training process.")
        with gr.Column():
            # Output Directory
            output_dir_box = gr.Textbox(
                label="Output Directory",
                value=OUTPUT_DIR,
                info="Directory for training outputs",
                interactive=False
            )

            # Training Parameters
            gr.Markdown("#### Training Parameters")
            with gr.Row():
                with gr.Column(scale=1):
                    epochs = gr.Number(
                        label="Epochs",
                        value=1000,
                        info="Total number of training epochs"
                    )
                    batch_size = gr.Number(
                        label="Batch Size",
                        value=1,
                        info="Batch size per GPU"
                    )
                    lr = gr.Number(
                        label="Learning Rate",
                        value=2e-5,
                        step=0.0001,
                        info="Optimizer learning rate"
                    )
                    save_every = gr.Number(
                        label="Save Every N Epochs",
                        value=2,
                        info="Frequency to save checkpoints"
                    )
                    eval_every = gr.Number(
                        label="Evaluate Every N Epochs",
                        value=1,
                        info="Frequency to perform evaluations"
                    )
                    rank = gr.Number(
                        label="LoRA Rank",
                        value=32,
                        info="LoRA adapter rank"
                    )
                    dtype = gr.Dropdown(
                        label="LoRA Dtype",
                        choices=['float32', 'float16', 'bfloat16', 'float8'],
                        value="bfloat16"
                    )
                    gradient_accumulation_steps = gr.Number(
                        label="Gradient Accumulation Steps",
                        value=4,
                        info="Micro-batch accumulation steps"
                    )

            # Dataset Configuration Fields
            gr.Markdown("#### Dataset Configuration")
            with gr.Row():
                enable_ar_bucket = gr.Checkbox(
                    label="Enable AR Bucket",
                    value=True,
                    info="Enable aspect ratio bucketing"
                )
                min_ar = gr.Number(
                    label="Minimum Aspect Ratio",
                    value=0.5,
                    step=0.1,
                    info="Minimum aspect ratio for AR buckets"
                )
            with gr.Row():
                max_ar = gr.Number(
                    label="Maximum Aspect Ratio",
                    value=2.0,
                    step=0.1,
                    info="Maximum aspect ratio for AR buckets"
                )
                num_ar_buckets = gr.Number(
                    label="Number of AR Buckets",
                    value=7,
                    step=1,
                    info="Number of aspect ratio buckets"
                )
            frame_buckets = gr.Textbox(
                label="Frame Buckets",
                value="[1, 33, 65]",
                info="Frame buckets as a JSON list. Example: [1, 33, 65]"
            )

            # Dataset Duplication and Resolutions
            with gr.Row():
                with gr.Column(scale=1):
                    num_repeats = gr.Number(
                        label="Dataset Num Repeats",
                        value=10,
                        info="Number of times to duplicate the dataset"
                    )
                with gr.Column(scale=1):
                    resolutions_input = gr.Textbox(
                        label="Resolutions",
                        value="[512]",
                        info="Resolutions to train on, given as a list. Example: [512] or [512, 768, 1024]"
                    )

            # Optimizer Parameters
            gr.Markdown("#### Optimizer Parameters")
            with gr.Row():
                with gr.Column(scale=1):
                    optimizer_type = gr.Textbox(
                        label="Optimizer Type",
                        value="adamw_optimizer",
                        info="Type of optimizer"
                    )
                    betas = gr.Textbox(
                        label="Betas",
                        value="[0.9, 0.99]",
                        info="Betas for the optimizer"
                    )
                    weight_decay = gr.Number(
                        label="Weight Decay",
                        value=0.01,
                        step=0.0001,
                        info="Weight decay for regularization"
                    )
                    eps = gr.Number(
                        label="Epsilon",
                        value=1e-8,
                        step=0.0000001,
                        info="Epsilon for the optimizer"
                    )

            # Additional Training Parameters
            gr.Markdown("#### Additional Training Parameters")
            with gr.Row():
                with gr.Column(scale=1):
                    gradient_clipping = gr.Number(
                        label="Gradient Clipping",
                        value=1.0,
                        step=0.1,
                        info="Value for gradient clipping"
                    )
                    warmup_steps = gr.Number(
                        label="Warmup Steps",
                        value=100,
                        step=10,
                        info="Number of warmup steps"
                    )
                    eval_before_first_step = gr.Checkbox(
                        label="Evaluate Before First Step",
                        value=True,
                        info="Perform evaluation before the first training step"
                    )
                    eval_micro_batch_size_per_gpu = gr.Number(
                        label="Eval Micro Batch Size Per GPU",
                        value=1,
                        info="Batch size for evaluation per GPU"
                    )
                    eval_gradient_accumulation_steps = gr.Number(
                        label="Eval Gradient Accumulation Steps",
                        value=1,
                        info="Gradient accumulation steps for evaluation"
                    )
                    checkpoint_every_n_minutes = gr.Number(
                        label="Checkpoint Every N Minutes",
                        value=120,
                        info="Frequency to create checkpoints (in minutes)"
                    )
                    activation_checkpointing = gr.Checkbox(
                        label="Activation Checkpointing",
                        value=True,
                        info="Enable activation checkpointing to save memory"
                    )
                    partition_method = gr.Textbox(
                        label="Partition Method",
                        value="parameters",
                        info="Method for partitioning (e.g., parameters)"
                    )
                    save_dtype = gr.Dropdown(
                        label="Save Dtype",
                        choices=['bfloat16', 'float16', 'float32'],
                        value="bfloat16",
                        info="Data type to save model checkpoints"
                    )
                    caching_batch_size = gr.Number(
                        label="Caching Batch Size",
                        value=1,
                        info="Batch size for caching"
                    )
                    steps_per_print = gr.Number(
                        label="Steps Per Print",
                        value=1,
                        info="Frequency to print training steps"
                    )
                    video_clip_mode = gr.Textbox(
                        label="Video Clip Mode",
                        value="single_middle",
                        info="Mode for video clipping (e.g., single_middle)"
                    )

            # Model Paths
            gr.Markdown("#### Model Paths")
            with gr.Row():
                with gr.Column(scale=1):
                    transformer_path = gr.Textbox(
                        label="Transformer Path",
                        value=f"{MODEL_DIR}/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors",
                        info="Path to the transformer model weights for Hunyuan Video."
                    )
                    vae_path = gr.Textbox(
                        label="VAE Path",
                        value=f"{MODEL_DIR}/hunyuan_video_vae_fp32.safetensors",
                        info="Path to the VAE model file."
                    )
                    llm_path = gr.Textbox(
                        label="LLM Path",
                        value=f"{MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer",
                        info="Path to the LLM's text tokenizer and encoder."
                    )
                    clip_path = gr.Textbox(
                        label="CLIP Path",
                        value=f"{MODEL_DIR}/clip-vit-large-patch14",
                        info="Path to the CLIP model directory."
                    )

            # Start Training Button and Log Box
            with gr.Row():
                with gr.Column(scale=1):
                    train_button = gr.Button("Start Training")
            with gr.Row():
                with gr.Column(scale=1):
                    output = gr.Textbox(
                        label="Output Logs",
                        lines=20,
                        interactive=False,
                        elem_id="log_box"
                    )

        # Initialize States
        is_training_state = gr.State(False)
        logs_state = gr.State("")
        current_dataset_creation = gr.State(None)

        # 4. Training Button Action
        def toggle_training(is_training, logs, dataset_option_selected, new_dataset, selected_dataset, output_dir_val, epochs_val, batch_size_val, lr_val, save_every_val, eval_every_val,
                            rank_val, dtype_val, transformer_path_val, vae_path_val, llm_path_val, clip_path_val,
                            optimizer_type_val, betas_val, weight_decay_val, eps_val, gradient_accumulation_steps_val,
                            num_repeats_val, resolutions_input_val, enable_ar_bucket_val, min_ar_val, max_ar_val, num_ar_buckets_val, frame_buckets_val,
                            gradient_clipping_val, warmup_steps_val, eval_before_first_step_val, eval_micro_batch_size_per_gpu_val,
                            eval_gradient_accumulation_steps_val, checkpoint_every_n_minutes_val, activation_checkpointing_val,
                            partition_method_val, save_dtype_val, caching_batch_size_val, steps_per_print_val, video_clip_mode_val):
            if is_training:
                # Stop training
                message = stop_training()
                logs += message + "\n"
                is_training = False
                return (logs, is_training, "Start Training")
            else:
                # Determine the dataset path based on selection
                if dataset_option_selected == "Create New Dataset":
                    dataset_path = new_dataset
                else:
                    dataset_path = selected_dataset

                if not dataset_path or not os.path.exists(dataset_path):
                    message = "Please upload a valid dataset before starting training."
                    logs += message + "\n"
                    return (logs, is_training, "Start Training")
                
                # Parse resolutions as list of integers
                try:
                    # Try to parse using JSON
                    resolutions = json.loads(resolutions_input_val)
                    if not isinstance(resolutions, list) or not all(isinstance(i, int) for i in resolutions):
                        raise ValueError
                except:
                    # If parsing fails, return error
                    message = "Resolutions must be a list of integers. Example: [512] or [512, 768, 1024]"
                    logs += message + "\n"
                    return (logs, is_training, "Start Training")

                # Parse frame_buckets as list of integers
                try:
                    frame_buckets_parsed = json.loads(frame_buckets_val)
                    if not isinstance(frame_buckets_parsed, list) or not all(isinstance(i, int) for i in frame_buckets_parsed):
                        raise ValueError
                except:
                    # If parsing fails, return error
                    message = "Frame Buckets must be a list of integers. Example: [1, 33, 65]"
                    logs += message + "\n"
                    return (logs, is_training, "Start Training")

                # Start training
                is_training = True
                logs = ""
                generator = train_lora(
                    dataset_path, 
                    output_dir_val, 
                    epochs_val, 
                    batch_size_val, 
                    lr_val, 
                    save_every_val, 
                    eval_every_val,
                    rank_val, 
                    dtype_val, 
                    transformer_path_val, 
                    vae_path_val, 
                    llm_path_val, 
                    clip_path_val, 
                    optimizer_type_val,
                    betas_val, 
                    weight_decay_val, 
                    eps_val, 
                    gradient_accumulation_steps_val, 
                    num_repeats_val, 
                    resolutions,
                    enable_ar_bucket_val, 
                    min_ar_val, 
                    max_ar_val, 
                    num_ar_buckets_val, 
                    frame_buckets_parsed,
                    gradient_clipping_val,
                    warmup_steps_val,
                    eval_before_first_step_val,
                    eval_micro_batch_size_per_gpu_val,
                    eval_gradient_accumulation_steps_val,
                    checkpoint_every_n_minutes_val,
                    activation_checkpointing_val,
                    partition_method_val,
                    save_dtype_val,
                    caching_batch_size_val,
                    steps_per_print_val,
                    video_clip_mode_val
                )
                for log in generator:
                    logs += log + "\n"
                    yield (logs, is_training, "Stop Training")

        train_button.click(
            fn=toggle_training,
            inputs=[
                is_training_state, logs_state,
                dataset_option, dataset_path_display, selected_dataset_display,
                output_dir_box, epochs, batch_size, lr, save_every, eval_every,
                rank, dtype, transformer_path, vae_path, llm_path, clip_path,
                optimizer_type, betas, weight_decay, eps, gradient_accumulation_steps,
                num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets,
                gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu,
                eval_gradient_accumulation_steps, checkpoint_every_n_minutes, activation_checkpointing,
                partition_method, save_dtype, caching_batch_size, steps_per_print, video_clip_mode
            ],
            outputs=[output, is_training_state, train_button],
            queue=False
        )

        # 5. Step 3: Download Outputs
        gr.Markdown("### Step 3: Download Outputs\nDownload the training results and configurations.")
        with gr.Row():
            download_output_button = gr.Button("Download Output")
            output_file = gr.File(label="Download Output File")
            download_output_button.click(
                fn=download_output_zip,
                inputs=[],
                outputs=output_file
            )
            download_dataset_button = gr.Button("Download Dataset & Configs")
            dataset_file = gr.File(label="Download Dataset & Configs File")
            download_dataset_button.click(
                fn=download_dataset_action,
                inputs=[selected_dataset_display, num_repeats, resolutions_input, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets],
                outputs=dataset_file
            )

        # 6. Auto-Scroll in Log Box
        gr.HTML("""
        <script>
        const logBox = document.getElementById("log_box");
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                logBox.scrollTop = logBox.scrollHeight;
            });
        });
        observer.observe(logBox, { childList: true, subtree: true, characterData: true });
        </script>
        """)

    return demo

# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    demo = build_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=None,
        share=False,
        max_file_size=MAX_UPLOAD_SIZE,  # Set based on IS_RUNPOD
        allowed_paths=["/workspace", ".", "/"]
    )
