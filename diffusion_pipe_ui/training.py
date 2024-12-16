# project/training.py

import threading
import subprocess
import psutil
import os
import signal
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

def kill_process_group(pid):
    """Kill the process group including all child processes."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass  # Process already terminated

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
                # Kill the entire process group
                kill_process_group(training_process.pid)
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

        # Training command
        conda_activate_path = "/opt/conda/etc/profile.d/conda.sh"
        conda_env_name = "pyenv"
        num_gpus = os.getenv("NUM_GPUS", "1")  # Get NUM_GPUS from environment variable, default to 1

        command = (
            f"bash -c 'source {conda_activate_path} && "
            f"conda activate {conda_env_name} && "
            f"cd /workspace/diffusion-pipe-ui/diffusion-pipe && "  # Updated path to use submodule
            f"NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus={num_gpus} "
            f"train.py --deepspeed --config {training_config_path}'"
        )

        # Start the training process with process group control
        training_process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            preexec_fn=os.setsid  # Create new process group
        )

    # Stream the logs with non-blocking reads
    import select
    import fcntl
    import errno

    # Set non-blocking mode for stdout and stderr
    for pipe in [training_process.stdout, training_process.stderr]:
        fd = pipe.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    try:
        while training_process.poll() is None:
            reads = [training_process.stdout.fileno(), training_process.stderr.fileno()]
            readable, _, _ = select.select(reads, [], [], 0.1)

            for fd in readable:
                if fd == training_process.stdout.fileno():
                    try:
                        line = training_process.stdout.readline()
                        if line:
                            yield line.strip()
                    except IOError as e:
                        if e.errno != errno.EAGAIN:
                            raise

                if fd == training_process.stderr.fileno():
                    try:
                        line = training_process.stderr.readline()
                        if line:
                            yield line.strip()
                    except IOError as e:
                        if e.errno != errno.EAGAIN:
                            raise

        # Read any remaining output
        remaining_stdout = training_process.stdout.read()
        if remaining_stdout:
            for line in remaining_stdout.splitlines():
                yield line.strip()

        remaining_stderr = training_process.stderr.read()
        if remaining_stderr:
            for line in remaining_stderr.splitlines():
                yield line.strip()

    except Exception as e:
        yield f"Error during training: {str(e)}"
    finally:
        with training_lock:
            if training_process is not None:
                kill_process_group(training_process.pid)
                training_process = None
            yield "Training process has finished."
