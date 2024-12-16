# project/training.py

import threading
import subprocess
import psutil
import os
from diffusion_pipe_ui.utils import generate_unique_filename
from diffusion_pipe_ui.config import CONFIG_HISTORY_DIR
import toml
from diffusion_pipe_ui.dataset import create_dataset_config
from datetime import datetime

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
            dataset_path=dataset_path, 
            dataset_name=os.path.basename(dataset_path),
            num_repeats=num_repeats, 
            resolutions=resolutions, 
            enable_ar_bucket=enable_ar_bucket, 
            min_ar=min_ar, 
            max_ar=max_ar, 
            num_ar_buckets=num_ar_buckets, 
            frame_buckets=frame_buckets
        )
        training_config_path = create_training_config(
            output_dir=output_dir, 
            dataset_path=dataset_config_path, 
            epochs=epochs, 
            batch_size=batch_size, 
            lr=lr, 
            save_every=save_every, 
            eval_every=eval_every, 
            rank=rank, 
            dtype=dtype,
            transformer_path=transformer_path, 
            vae_path=vae_path, 
            llm_path=llm_path, 
            clip_path=clip_path, 
            optimizer_type=optimizer_type, 
            betas=betas, 
            weight_decay=weight_decay, 
            eps=eps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clipping=gradient_clipping,
            warmup_steps=warmup_steps,
            eval_before_first_step=eval_before_first_step,
            eval_micro_batch_size_per_gpu=eval_micro_batch_size_per_gpu,
            eval_gradient_accumulation_steps=eval_gradient_accumulation_steps,
            checkpoint_every_n_minutes=checkpoint_every_n_minutes,
            activation_checkpointing=activation_checkpointing,
            partition_method=partition_method,
            save_dtype=save_dtype,
            caching_batch_size=caching_batch_size,
            steps_per_print=steps_per_print,
            video_clip_mode=video_clip_mode
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
