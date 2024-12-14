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
MODEL_DIR = "/models"
BASE_DATASET_DIR = "/datasets"
OUTPUT_DIR = "/output"
CONFIG_HISTORY_DIR = "/config_history"

# Create directories if they don't exist
os.makedirs(CONFIG_HISTORY_DIR, exist_ok=True)
os.makedirs(BASE_DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum number of images to display in the gallery
MAX_IMAGES = 50

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

def create_dataset_config(dataset_path, num_repeats, resolutions):
    """Create and save the dataset configuration in TOML format."""
    dataset_config = {
        "resolutions": resolutions,  # Uses the provided list of resolutions
        "enable_ar_bucket": True,
        "min_ar": 0.5,
        "max_ar": 2.0,
        "num_ar_buckets": 7,
        "frame_buckets": [1, 33, 65],
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
                           gradient_accumulation_steps):
    """Create and save the training configuration in TOML format."""
    training_config = {
        "output_dir": output_dir,
        "dataset": dataset_path,
        "epochs": epochs,
        "micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "warmup_steps": 100,
        "eval_every_n_epochs": eval_every,
        "eval_before_first_step": True,
        "eval_micro_batch_size_per_gpu": 1,
        "eval_gradient_accumulation_steps": 1,
        "save_every_n_epochs": save_every,
        "checkpoint_every_n_minutes": 120,
        "activation_checkpointing": True,
        "partition_method": "parameters",
        "save_dtype": "bfloat16",
        "caching_batch_size": 1,
        "steps_per_print": 1,
        "video_clip_mode": "single_middle",
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
               gradient_accumulation_steps, num_repeats, resolutions):
    """Run the training command and stream logs."""
    global training_process
    with training_lock:
        if training_process is not None:
            yield "Training is already in progress."
            return

        # Create configurations
        dataset_config_path = create_dataset_config(dataset_path, num_repeats, resolutions)
        training_config_path = create_training_config(
            output_dir, dataset_config_path, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
            transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
            gradient_accumulation_steps
        )

        # Training command (replace with the actual command)
        conda_activate_path = "/opt/conda/etc/profile.d/conda.sh"
        conda_env_name = "pyenv"

        command = (
            f"bash -c 'source {conda_activate_path} && "
            f"conda activate {conda_env_name} && "
            f"cd /diffusion-pipe && "
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
            output = training_process.stdout.readline()
            error = training_process.stderr.readline()
            if output:
                yield output.strip()
            if error:
                yield error.strip()
            if output == "" and error == "" and training_process.poll() is not None:
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

def upload_dataset(files):
    """Handle uploaded dataset files and store them in a unique directory."""
    if not files:
        return "", "No files uploaded."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join(BASE_DATASET_DIR, f"dataset_{timestamp}")
    os.makedirs(dataset_dir, exist_ok=True)
    uploaded_files = []

    for file_path in files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dataset_dir, filename)

        if zipfile.is_zipfile(file_path):
            # If the file is a ZIP, extract its contents
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                uploaded_files.append(f"{filename} (extracted)")
            except zipfile.BadZipFile:
                uploaded_files.append(f"{filename} (invalid ZIP)")
                continue
        else:
            # If it's not a ZIP, copy the file directly
            shutil.copy(file_path, dest_path)
            uploaded_files.append(filename)

    return dataset_dir, f"Dataset uploaded to {dataset_dir}: {', '.join(uploaded_files)}"

def show_images(dataset_dir):
    """Display uploaded images in a gallery."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        # Return empty list if dataset_dir is invalid
        return []

    # List only image files
    image_files = [
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))
    ]
    image_paths = [os.path.join(dataset_dir, img) for img in image_files[:MAX_IMAGES]]
    return image_paths

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

def download_dataset_action(dataset_dir, num_repeats, resolutions_input):
    """Action to download the dataset and configurations."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        return ""  # Gradio will handle this as no file to download
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
    create_dataset_config(dataset_dir, num_repeats, resolutions)
    return download_dataset_config_zip(dataset_dir)

# -----------------------------
# Gradio Interface Construction
# -----------------------------

def build_interface():
    """Build the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# LoRA Training Interface for Hunyuan Video")
        
        # 1. Step 1: Dataset Upload
        gr.Markdown("### Step 1: Dataset\nUpload your dataset (images and captions) either as individual files or as a ZIP archive.")
        with gr.Row():
            with gr.Column():
                images = gr.File(
                    label="Upload Images (.jpg, .png, .gif, .bmp, .webp) and Captions (.txt) or a ZIP archive",
                    file_types=[".jpg", ".png", ".gif", ".bmp", ".webp", ".txt", ".zip"],
                    file_count="multiple",
                    type="filepath"
                )
            with gr.Column():
                dataset_status = gr.Textbox(label="Upload Status", interactive=False)
                dataset_path_box = gr.Textbox(label="Dataset Path", interactive=False)
        
        # Upload files
        images.upload(
            fn=upload_dataset,
            inputs=images,
            outputs=[dataset_path_box, dataset_status]
        )
        
        # 2. Image Gallery
        gr.Markdown("### Image Gallery")
        gallery = gr.Gallery(
            label="Uploaded Images",
            show_label=False,
            elem_id="gallery",
            columns=3,
            rows=1,
            object_fit="contain",
            height="auto"
        )
        
        # Display images in gallery
        dataset_path_box.change(
            fn=show_images,
            inputs=dataset_path_box,
            outputs=gallery
        )
        
        # 3. Step 2: Training
        gr.Markdown("### Step 2: Training\nConfigure your training parameters and start or stop the training process.")
        with gr.Row():
            with gr.Column(scale=1):
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value=OUTPUT_DIR,
                    info="Directory for training outputs",
                    interactive=False
                )
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
                
                # Group num_repeats and resolutions_input side by side
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
                
                # Model Paths
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
            with gr.Column(scale=1):
                train_button = gr.Button("Start Training")
                output = gr.Textbox(
                    label="Output Logs",
                    lines=20,
                    interactive=False,
                    elem_id="log_box"
                )
        
        # Initialize States
        is_training_state = gr.State(False)
        logs_state = gr.State("")
        
        # 4. Training Button Action
        def toggle_training(is_training, logs, dataset_path, output_dir, epochs, batch_size, lr, save_every, eval_every,
                            rank, dtype, transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas,
                            weight_decay, eps, gradient_accumulation_steps, num_repeats, resolutions_input):
            if is_training:
                # Stop training
                message = stop_training()
                logs += message + "\n"
                is_training = False
                yield (logs, is_training, "Start Training")
            else:
                if not dataset_path or not os.path.exists(dataset_path):
                    message = "Please upload a valid dataset before starting training."
                    logs += message + "\n"
                    yield (logs, is_training, "Start Training")
                else:
                    # Parse resolutions as list of integers
                    try:
                        # Try to parse using JSON
                        resolutions = json.loads(resolutions_input)
                        if not isinstance(resolutions, list) or not all(isinstance(i, int) for i in resolutions):
                            raise ValueError
                    except:
                        # If parsing fails, return error
                        message = "Resolutions must be a list of integers. Example: [512] or [512, 768, 1024]"
                        logs += message + "\n"
                        yield (logs, is_training, "Start Training")
                        return

                    # Start training
                    is_training = True
                    logs = ""
                    generator = train_lora(dataset_path, output_dir, epochs, batch_size, lr, save_every, eval_every,
                                           rank, dtype, transformer_path, vae_path, llm_path, clip_path, optimizer_type,
                                           betas, weight_decay, eps, gradient_accumulation_steps, num_repeats, resolutions)
                    for log in generator:
                        logs += log + "\n"
                        yield (logs, is_training, "Stop Training")
        
        train_button.click(
            fn=toggle_training,
            inputs=[
                is_training_state, logs_state, dataset_path_box, output_dir, epochs, batch_size, lr, save_every, eval_every,
                rank, dtype, transformer_path, vae_path, llm_path, clip_path,
                optimizer_type, betas, weight_decay, eps, gradient_accumulation_steps, num_repeats, resolutions_input
            ],
            outputs=[output, is_training_state, train_button]
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
                inputs=[dataset_path_box, num_repeats, resolutions_input],
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
        max_file_size=2 * gr.FileSize.GB,
        allowed_paths=["/datasets", "/output", "/config_history", "/models", ".", "/app"]
    )
