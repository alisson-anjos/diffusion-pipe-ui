# project/training.py

import threading
import os
import signal
from diffusion_pipe_ui.utils import generate_unique_filename
from diffusion_pipe_ui.config import CONFIG_HISTORY_DIR
import toml
from diffusion_pipe_ui.dataset import create_dataset_config
import sys
import torch
from datetime import datetime
import json

# Adicionar o diretório raiz ao path para poder importar
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DIFFUSION_PIPE_DIR = os.path.join(ROOT_DIR, 'diffusion-pipe')

# Ensure the diffusion-pipe directory exists in sys.path
if DIFFUSION_PIPE_DIR not in sys.path:
    sys.path.insert(0, DIFFUSION_PIPE_DIR)

# Now we can import using relative paths from the diffusion-pipe directory
from models.hunyuan_video import HunyuanVideoPipeline
from utils.dataset import DatasetManager, Dataset, PipelineDataLoader
from utils.common import is_main_process, DTYPE_MAP
import deepspeed
from deepspeed import comm as dist

# -----------------------------
# Training Process Management
# -----------------------------

class TrainingManager:
    def __init__(self):
        self.training_lock = threading.Lock()
        self.stop_flag = False
        self.model_engine = None
        self.training_thread = None

    def stop_training(self):
        """Stop the training process safely."""
        with self.training_lock:
            if self.training_thread and self.training_thread.is_alive():
                self.stop_flag = True
                self.training_thread.join()
                self.model_engine = None
                self.training_thread = None
                return "Training process has been stopped."
            return "No training process is running."

    def train_lora(self, dataset_path, output_dir, epochs, batch_size, lr, save_every, eval_every, rank, dtype,
                   transformer_path, vae_path, llm_path, clip_path, optimizer_type, betas, weight_decay, eps,
                   gradient_accumulation_steps, num_repeats, resolutions, enable_ar_bucket, min_ar, max_ar, num_ar_buckets, frame_buckets,
                   gradient_clipping, warmup_steps, eval_before_first_step, eval_micro_batch_size_per_gpu, eval_gradient_accumulation_steps,
                   checkpoint_every_n_minutes, activation_checkpointing, partition_method, save_dtype, caching_batch_size, steps_per_print,
                   video_clip_mode):
        """Run the training process directly using the diffusion-pipe code."""
        
        with self.training_lock:
            if self.training_thread and self.training_thread.is_alive():
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

            # Create training config
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

            training_file = generate_unique_filename("hunyuan_video_auto")
            training_path_full = os.path.join(CONFIG_HISTORY_DIR, training_file)
            with open(training_path_full, "w") as f:
                toml.dump(training_config, f)

            # Reset stop flag
            self.stop_flag = False

            def training_process():
                try:
                    # Initialize distributed training
                    deepspeed.init_distributed()
                    torch.cuda.set_device(dist.get_rank())

                    # Set config defaults
                    config = training_config
                    
                    # Create model instance
                    model = HunyuanVideoPipeline(config)

                    # Load model and setup dataset
                    model.load_diffusion_model()
                    
                    if adapter_config := config.get('adapter', None):
                        peft_config = model.configure_adapter(adapter_config)
                    else:
                        peft_config = None

                    # Setup dataset
                    dataset_manager = DatasetManager(model, regenerate_cache=False, caching_batch_size=config['caching_batch_size'])
                    train_data = Dataset(toml.load(config['dataset']), model.name)
                    dataset_manager.register(train_data)
                    dataset_manager.cache()

                    # Initialize DeepSpeed
                    ds_config = {
                        'train_micro_batch_size_per_gpu': config.get('micro_batch_size_per_gpu', 1),
                        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
                        'gradient_clipping': config.get('gradient_clipping', 1.0),
                        'steps_per_print': config.get('steps_per_print', 1),
                    }

                    # Convert model to layers and initialize pipeline
                    layers = model.to_layers()
                    pipeline_model = deepspeed.pipe.PipelineModule(
                        layers=layers,
                        num_stages=config.get('pipeline_stages', 1),
                        partition_method=config.get('partition_method', 'parameters'),
                    )

                    # Initialize optimizer and model engine
                    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]
                    optimizer = torch.optim.AdamW(parameters_to_train, **config['optimizer'])
                    
                    self.model_engine, _, _, _ = deepspeed.initialize(
                        args=type('Args', (), {'local_rank': -1})(),
                        model=pipeline_model,
                        optimizer=optimizer,
                        config=ds_config
                    )

                    # Setup training dataloader
                    train_data.post_init(
                        self.model_engine.grid.get_data_parallel_rank(),
                        self.model_engine.grid.get_data_parallel_world_size(),
                        self.model_engine.train_micro_batch_size_per_gpu(),
                        self.model_engine.gradient_accumulation_steps(),
                    )

                    train_dataloader = PipelineDataLoader(
                        train_data,
                        self.model_engine.gradient_accumulation_steps(),
                        model
                    )

                    self.model_engine.set_dataloader(train_dataloader)

                    # Training loop
                    step = 1
                    epoch = train_dataloader.epoch
                    while not self.stop_flag:
                        self.model_engine.reset_activation_shape()
                        loss = self.model_engine.train_batch().item()
                        
                        if is_main_process():
                            yield f"Step {step}, Loss: {loss:.4f}, Epoch: {epoch}"
                        
                        train_dataloader.sync_epoch()
                        new_epoch = train_dataloader.epoch
                        
                        if new_epoch != epoch:
                            epoch = new_epoch
                            if epoch > epochs:
                                break
                        
                        step += 1

                except Exception as e:
                    yield f"Error during training: {str(e)}"
                finally:
                    if self.model_engine:
                        self.model_engine = None
                    yield "Training process has finished."

            # Start training in a separate thread
            self.training_thread = threading.Thread(target=lambda: None)
            for log in training_process():
                yield log

training_manager = TrainingManager()

def stop_training():
    """Stop the training process."""
    return training_manager.stop_training()

def train_lora(*args, **kwargs):
    """Start the training process."""
    return training_manager.train_lora(*args, **kwargs)
