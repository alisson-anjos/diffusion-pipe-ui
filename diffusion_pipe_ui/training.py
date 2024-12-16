# project/training.py

import threading
import os
import signal
import sys
import subprocess
from diffusion_pipe_ui.utils import generate_unique_filename
from diffusion_pipe_ui.config import CONFIG_HISTORY_DIR
import toml
from diffusion_pipe_ui.dataset import create_dataset_config

# Adicionar o diretório raiz ao path para poder importar
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DIFFUSION_PIPE_DIR = os.path.join(ROOT_DIR, 'diffusion-pipe')

class TrainingManager:
    def __init__(self):
        self.training_lock = threading.Lock()
        self.stop_flag = False
        self.training_process = None

    def stop_training(self):
        """Stop the training process safely."""
        with self.training_lock:
            if self.training_process:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self.training_process.pid), signal.SIGTERM)
                self.training_process = None
                return "Training process has been stopped."
            return "No training process is running."

    def train_lora(self, dataset_path, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
                   transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
                   gradient_accumulation_steps, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets,
                   gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
                   checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
                   video_clip_mode):
        """Run the training process using the original diffusion-pipe train.py."""
        
        with self.training_lock:
            if self.training_process:
                yield "Training is already in progress."
                return

            # Create dataset configuration
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

            # Create training configuration
            training_config = {
                "output_dir": output_dir,
                "dataset": dataset_config_path,
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

            # Save training config to a TOML file
            training_file = generate_unique_filename("hunyuan_video_auto", extension=".toml")
            training_path_full = os.path.join(CONFIG_HISTORY_DIR, training_file)
            with open(training_path_full, "w") as f:
                toml.dump(training_config, f)

            # Reset stop flag
            self.stop_flag = False

            # Set environment variables for training
            env = os.environ.copy()
            env.update({
                'NCCL_P2P_DISABLE': '1',
                'NCCL_IB_DISABLE': '1',
                'PYTHONPATH': f"{DIFFUSION_PIPE_DIR}:{DIFFUSION_PIPE_DIR}/submodules/HunyuanVideo:{os.environ.get('PYTHONPATH', '')}"
            })

            # Get number of GPUs from environment variable
            num_gpus = os.environ.get('NUM_GPUS', '1')

            # Prepare deepspeed command
            cmd = [
                "deepspeed",
                f"--num_gpus={num_gpus}",
                os.path.join(DIFFUSION_PIPE_DIR, "train.py"),
                "--deepspeed",
                "--config", training_path_full
            ]

            # Start training process
            try:
                # Start process in a new process group so we can kill it and its children
                self.training_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env=env,
                    preexec_fn=os.setsid if os.name != 'nt' else None,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                )

                # Stream output
                while True:
                    output = self.training_process.stdout.readline()
                    if output == '' and self.training_process.poll() is not None:
                        break
                    if output:
                        yield output.strip()

                # Check if process completed successfully
                if self.training_process.returncode == 0:
                    yield "Training completed successfully."
                else:
                    yield f"Training failed with return code {self.training_process.returncode}"

            except Exception as e:
                yield f"Error during training: {str(e)}"
            finally:
                if self.training_process:
                    self.training_process = None

training_manager = TrainingManager()

def stop_training():
    """Stop the training process."""
    return training_manager.stop_training()

def train_lora(*args, **kwargs):
    """Start the training process."""
    return training_manager.train_lora(*args, **kwargs)
