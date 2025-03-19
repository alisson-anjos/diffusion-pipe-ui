from typing import Optional
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../submodules/HunyuanVideo'))

import torch
from torch import nn
import peft
from peft.tuners._buffer_dict import BufferDict
from transformers import CLIPTextModel, AutoModel
import deepspeed
from deepspeed.runtime.pipe.schedule import (
    SendGrad, RecvActivation, SendActivation, RecvGrad, LoadMicroBatch, ForwardPass, BackwardPass,
    ReduceTiedGrads, ReduceGrads, OptimizerStep,
)
from deepspeed import comm as dist
from deepspeed.utils import groups

import hyvideo.text_encoder
from hyvideo.constants import PRECISION_TO_TYPE, TEXT_ENCODER_PATH


def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
    """
    Move the adapter of the given name to the device of the base layer.
    """
    if device is None:
        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                device = weight.device
                dtype = weight.dtype
                break
        else:
            # no break encountered: could not determine the device
            return

    meta = torch.device("meta")

    # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
    # move this specific adapter to the device, as the other adapters could be on different devices
    # see #1639
    for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
        adapter_layer = getattr(self, adapter_layer_name, None)
        if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
            continue
        if adapter_name not in adapter_layer:
            continue
        if any(p.device == meta for p in adapter_layer.parameters()):
            continue

        if ((weight.dtype.is_floating_point or weight.dtype.is_complex)
            # This is the part I added.
            and not (weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.float8_e5m2)):
            adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
        else:
            adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
):
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    if logger is not None:
        logger.info(
            f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}"
        )

    torch_dtype = 'auto'
    if text_encoder_precision is not None:
        torch_dtype = PRECISION_TO_TYPE[text_encoder_precision]

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=torch_dtype)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = AutoModel.from_pretrained(
            text_encoder_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype
        )
        text_encoder.final_layer_norm = text_encoder.norm
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def train_schedule_steps(self):
    prev_micro_batch_id = -1
    total_steps = 2 * (self.micro_batches + self.stages - 1)
    for step_id in range(total_steps):
        # Map the step of the pipeline to the micro-batch id and also whether it is a
        # forward or backward pass step.
        micro_batch_id, is_forward = self._step_to_micro_batch(step_id)

        if self._valid_micro_batch(prev_micro_batch_id):
            prev_buffer = self._buffer_idx(prev_micro_batch_id)
        if self._valid_micro_batch(micro_batch_id):
            curr_buffer = self._buffer_idx(micro_batch_id)

        cmds = []

        # First/last stage loads
        if self.stage_id == 0 or self.stage_id == self.stages - 1:
            if is_forward and self._valid_micro_batch(micro_batch_id):
                cmds.append(LoadMicroBatch(curr_buffer))

        # Exchange activations
        if is_forward:
            if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.prev_stage):
                cmds.append(SendGrad(prev_buffer))
            if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.prev_stage):
                cmds.append(RecvActivation(curr_buffer))
        else:
            if self._valid_micro_batch(micro_batch_id) and self._valid_stage(self.next_stage):
                cmds.append(RecvGrad(curr_buffer))
            if self._valid_micro_batch(prev_micro_batch_id) and self._valid_stage(self.next_stage):
                cmds.append(SendActivation(prev_buffer))

        # Computation
        if self._valid_micro_batch(micro_batch_id):
            if is_forward:
                cmds.append(ForwardPass(curr_buffer))
            else:
                cmds.append(BackwardPass(curr_buffer))

        # Model step at the end of the batch
        if step_id == total_steps - 1:
            cmds.append(ReduceTiedGrads())
            cmds.append(ReduceGrads())
            cmds.append(OptimizerStep())

        # Prepare state for next time
        prev_micro_batch_id = micro_batch_id
        yield cmds


def broadcast_model(self):
    for n, p in self.module.named_parameters():
        if torch.is_tensor(p) and p.requires_grad:
            orig_device = p.device
            move_to_gpu = (orig_device != self.device)
            if move_to_gpu:
                p.data = p.data.to(self.device)
            dist.broadcast(p.data, groups._get_broadcast_src_rank(), group=self.seq_data_parallel_group)
            if move_to_gpu:
                p.data = p.data.to(orig_device)


def apply_patches():
    # Prevent PEFT from downcasting LoRA weights to fp8 only for this script to upcast them again.
    # TODO: probably should send a PR to PEFT. Default behavior looks like a mistake to me.
    peft.tuners.tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = _move_adapter_to_device_of_base_layer

    # Use torch_dtype to avoid needlessly loading the text encoder in float32, only to cast it right after.
    hyvideo.text_encoder.load_text_encoder = load_text_encoder

    # LoadMicroBatch before sending / receiving activations so we can avoid a deadlock and broadcast the target
    # from the first stage to the last stage. InferenceSchedule already has the commands in the right order
    # and doesn't need this.
    deepspeed.runtime.pipe.schedule.TrainSchedule.steps = train_schedule_steps

    # This does two things:
    # 1. For block swapping, some parameters will be on CPU when the DeepSpeedEngine is constructed. So we patch this to
    #    first move those parameters to GPU, then back again when broadcasting the model weights from rank 0.
    # 2. We skip broadcasting for parameters that don't require grad. These weights are static and always the same because
    #    they were loaded from disk, so we can safely skip broadcasting and it's faster.
    deepspeed.runtime.engine.DeepSpeedEngine._broadcast_model = broadcast_model
