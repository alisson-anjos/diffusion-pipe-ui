import gradio as gr
import os
import shutil
import subprocess
import toml
from datetime import datetime
import threading
from PIL import Image, ImageDraw, ImageFont
import zipfile

MODEL_DIR = os.getenv("MODEL_DIR", "/models")
BASE_DATASET_DIR = "/datasets"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/output")
CONFIG_HISTORY_DIR = "/config_history"
os.makedirs(CONFIG_HISTORY_DIR, exist_ok=True)
os.makedirs(BASE_DATASET_DIR, exist_ok=True)

training_process = None
training_lock = threading.Lock()


def generate_unique_filename(base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.toml"


def create_dataset_config(dataset_path, num_repeats):
    """Generate dataset TOML configuration."""
    dataset_config = {
        "resolutions": [512],
        "enable_ar_bucket": True,
        "min_ar": 0.5,
        "max_ar": 2.0,
        "num_ar_buckets": 7,
        "frame_buckets": [1, 33, 65],
        "directory": {
            "path": dataset_path,
            "num_repeats": num_repeats
        }
    }
    dataset_file = generate_unique_filename("dataset_auto")
    dataset_path_full = os.path.join(CONFIG_HISTORY_DIR, dataset_file)
    with open(dataset_path_full, "w") as f:
        toml.dump(dataset_config, f)
    return dataset_path_full


def create_training_config(output_dir, dataset_path, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
                           transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
                           gradient_accumulation_steps):
    """Generate training TOML configuration."""
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


def train_lora(dataset_path, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
               transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
               gradient_accumulation_steps, num_repeats):
    """Run the training command and stream logs."""
    global training_process

    with training_lock:
        if training_process is not None:
            yield "Training is already in progress."
            return

        dataset_config_path = create_dataset_config(dataset_path, num_repeats)
        training_config_path = create_training_config(
            output_dir, dataset_config_path, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
            transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
            gradient_accumulation_steps
        )

        conda_activate_path = "/opt/conda/etc/profile.d/conda.sh"
        conda_env_name = "pyenv"  # Replace if needed

        command = (
            f"bash -c 'source {conda_activate_path} && "
            f"conda activate {conda_env_name} && "
            f"cd /diffusion-pipe && "
            f"NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus=1 "
            f"train.py --deepspeed --config {training_config_path}'"
        )

        training_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    logs = ""
    try:
        for line in training_process.stdout:
            logs += line
            yield logs

        for line in training_process.stderr:
            logs += line
            yield logs
    finally:
        with training_lock:
            training_process = None


def stop_training():
    """Stop the training process."""
    global training_process

    with training_lock:
        if training_process is not None:
            training_process.terminate()
            training_process = None
            return "Training process terminated."
        return "No training process is running."


def upload_dataset(files):
    """Handle uploaded dataset files and store them in a unique directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join(BASE_DATASET_DIR, f"dataset_{timestamp}")
    os.makedirs(dataset_dir, exist_ok=True)
    uploaded_files = []
    for file_path in files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dataset_dir, filename)
        shutil.copy2(file_path, dest_path)
        uploaded_files.append(dest_path)
    return dataset_dir, f"Dataset uploaded to {dataset_dir}: {', '.join(uploaded_files)}"


def _train_generator_wrapper(generator, initial_logs):
    """Wrap the train_lora generator to keep track of the logs so we can add final message."""
    all_logs = initial_logs
    for chunk in generator:
        all_logs = chunk
        yield all_logs, True, gr.update()


def toggle_training(is_training, dataset_path, output_dir, epochs, batch_size, lr, save_every, eval_every,
                    rank, dtype, transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas,
                    weight_decay, eps, gradient_accumulation_steps, num_repeats):
    """Toggle training state. If not training, start. If training, stop."""
    if is_training:
        msg = stop_training()
        yield msg, False, gr.update(value="Start Training")
    else:
        all_logs = "Starting training...\n"
        yield all_logs, True, gr.update(value="Stop Training")

        for log_chunk, training_state, btn_state in _train_generator_wrapper(
            train_lora(dataset_path, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
                       transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas,
                       weight_decay, eps, gradient_accumulation_steps, num_repeats), all_logs):
            all_logs = log_chunk
            yield all_logs, True, gr.update()
        all_logs += "\nTraining finished."
        yield all_logs, False, gr.update(value="Start Training")


def download_output_zip():
    """Create a zip file from OUTPUT_DIR and return its path."""
    zip_path = "/tmp/output.zip"
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
    """Create a zip file containing the dataset directory and the config history."""
    zip_path = "/tmp/dataset_configs.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add dataset directory
        if dataset_dir and os.path.exists(dataset_dir):
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, os.path.dirname(dataset_dir))
                    zf.write(filepath, arcname)

        # Add config history
        for root, dirs, files in os.walk(CONFIG_HISTORY_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.join("config_history", os.path.relpath(filepath, CONFIG_HISTORY_DIR))
                zf.write(filepath, arcname)
    return zip_path


def download_dataset_action(dataset_dir, num_repeats):
    # If dataset_dir is not set or doesn't exist, return None
    if not dataset_dir or not os.path.exists(dataset_dir):
        return None

    # Create a fresh dataset config to ensure we have config ready
    if not num_repeats:
        num_repeats = 10  # default if none provided
    create_dataset_config(dataset_dir, num_repeats)

    return download_dataset_config_zip(dataset_dir)


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# LoRA Training Interface for Hunyuan Video")

        with gr.Row():
            # Upload and dataset info
            dataset_upload = gr.File(label="Upload Dataset Files", file_types=[".jpg", ".png", ".txt"], file_count="multiple", type="filepath")
            dataset_status = gr.Textbox(label="Upload Status", interactive=False)
            dataset_path_box = gr.Textbox(label="Dataset Path", interactive=False)

            dataset_upload.upload(
                upload_dataset,
                inputs=dataset_upload,
                outputs=[dataset_path_box, dataset_status]
            )

        with gr.Row():
            # Download Dataset & Configs button moved above training parameters
            download_dataset_button = gr.Button("Download Dataset & Configs")
            dataset_file = gr.File(label="Download Dataset & Configs File")
        
        with gr.Row():
            output_dir = gr.Textbox(label="Output Directory", value=OUTPUT_DIR)

        with gr.Row():
            transformer_path = gr.Textbox(label="Transformer Path", value=f"{MODEL_DIR}/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors")
            vae_path = gr.Textbox(label="VAE Path", value=f"{MODEL_DIR}/hunyuan_video_vae_bf16.safetensors")

        with gr.Row():
            llm_path = gr.Textbox(label="LLM Path", value=f"{MODEL_DIR}/llava-llama-3-8b-text-encoder-tokenizer")
            clip_path = gr.Textbox(label="CLIP Path", value=f"{MODEL_DIR}/clip-vit-large-patch14")

        with gr.Row():
            epochs = gr.Number(label="Epochs", value=1000)
            batch_size = gr.Number(label="Batch Size", value=1)
            lr = gr.Number(label="Learning Rate", value=2e-5, step=0.0001)

        with gr.Row():
            save_every = gr.Number(label="Save Every N Epochs", value=2)
            eval_every = gr.Number(label="Evaluate Every N Epochs", value=1)

        with gr.Row():
            rank = gr.Number(label="LoRA Rank", value=32)
            dtype = gr.Dropdown(label="LoRA Dtype", choices=['float32', 'float16', 'bfloat16', 'float8'], value="bfloat16")

        with gr.Row():
            gradient_accumulation_steps = gr.Number(label="Gradient Accumulation Steps", value=4)
            num_repeats = gr.Number(label="Dataset Num Repeats", value=10)

        with gr.Row():
            optimizer_type = gr.Textbox(label="Optimizer Type", value="adamw_optimi")
            betas = gr.Textbox(label="Betas", value="[0.9, 0.99]")
            weight_decay = gr.Number(label="Weight Decay", value=0.01, step=0.0001)
            eps = gr.Number(label="Epsilon", value=1e-8, step=0.0000001)

        is_training = gr.State(False)
        train_button = gr.Button("Start Training")
        output = gr.Textbox(label="Output Logs", lines=40)

        train_button.click(
            toggle_training,
            inputs=[
                is_training, dataset_path_box, output_dir, epochs, batch_size, lr, save_every, eval_every,
                rank, dtype, transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas,
                weight_decay, eps, gradient_accumulation_steps, num_repeats
            ],
            outputs=[output, is_training, train_button]
        )

        # Button to download output
        with gr.Row():
            download_output_button = gr.Button("Download Output")
            output_file = gr.File(label="Download Output File")

        download_output_button.click(
            download_output_zip,
            inputs=[],
            outputs=output_file
        )

        # Download dataset & configs (now generates config if needed)
        download_dataset_button.click(
            download_dataset_action,
            inputs=[dataset_path_box, num_repeats],
            outputs=dataset_file
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get('GRADIO_PORT', 7860))
)
