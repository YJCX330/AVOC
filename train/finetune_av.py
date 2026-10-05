import json
import os
from dataclasses import dataclass, field
from functools import partial
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModel, AutoProcessor, AutoTokenizer, Trainer
from transformers.integrations import deepspeed, is_deepspeed_zero3_enabled
from transformers.trainer import *
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled

from dataset_av import AudioVideoDataset, av_data_collator

import logging
logger = logging.getLogger(__name__)

# ======================================================================
# arguments
# ======================================================================

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="openbmb/MiniCPM-o-2_6"
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to training data JSON."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to evaluation data JSON."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=False)
    tune_audio: Optional[bool] = field(default=True)
    tune_compression: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=1)
    use_image_id: Optional[bool] = field(default=False)
    max_num_frames: Optional[int] = field(default=32)
    compression_enabled: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable token compression. None = use checkpoint config."},
    )
    compression_topk: Optional[int] = field(default=None)
    compression_topk_ratio: Optional[float] = field(default=None)
    compression_topk_ratio_end: Optional[float] = field(default=None)
    compression_text_guided_weight: Optional[float] = field(default=None)
    compression_gumbel_temperature: Optional[float] = field(default=None)
    compression_gumbel_temperature_end: Optional[float] = field(default=None)
    compression_debug_log: bool = field(
        default=False,
        metadata={"help": "Print selected token indices and per-token score gradients each training step (rank-0 only)."},
    )
    compression_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate learning rate for compression module. If None, uses the main learning_rate."},
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = (
        r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None


# ======================================================================
# trainer
# ======================================================================

class AVTrainer(Trainer):
    """Trainer that passes the full input dict to model.forward(data=...).

    Includes optional compression ratio and temperature annealing.
    """

    # def _get_train_sampler(self):
    #     from torch.utils.data import SequentialSampler
    #     return SequentialSampler(self.train_dataset)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        compression_lr = getattr(self.args, "compression_lr", None)
        if compression_lr is None:
            return super().create_optimizer()

        comp_param_names = set()
        for name, _ in self.model.named_parameters():
            if "compression_module" in name:
                comp_param_names.add(name)

        no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight")
        decay_main, no_decay_main = [], []
        decay_comp, no_decay_comp = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_no_decay = any(nd in name for nd in no_decay_keywords)
            if name in comp_param_names:
                (no_decay_comp if is_no_decay else decay_comp).append(param)
            else:
                (no_decay_main if is_no_decay else decay_main).append(param)

        groups = [
            {"params": decay_main,    "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
            {"params": no_decay_main,  "weight_decay": 0.0,                   "lr": self.args.learning_rate},
            {"params": decay_comp,     "weight_decay": self.args.weight_decay, "lr": compression_lr},
            {"params": no_decay_comp,  "weight_decay": 0.0,                   "lr": compression_lr},
        ]
        groups = [g for g in groups if g["params"]]

        for i, g in enumerate(groups):
            rank0_print(f"Optimizer group {i}: lr={g['lr']}, wd={g['weight_decay']}, #params={len(g['params'])}")

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    def _get_compression_module(self, model):
        m = model
        if hasattr(m, "module"):
            m = m.module
        return getattr(m, "compression_module", None)

    def _anneal_compression(self, model):
        """Randomly sample topk_ratio between start and end each step.

        Also supports exponential annealing of gumbel_temperature.
        """
        import random
        comp = self._get_compression_module(model)
        if comp is None:
            return
        m = model.module if hasattr(model, "module") else model
        cfg = m.config
        max_steps = self.state.max_steps
        step = self.state.global_step
        anneal_end_step = int(0.8 * max_steps) if max_steps > 0 else 1

        ratio_end = getattr(cfg, "compression_topk_ratio_end", None)
        if ratio_end is not None and getattr(cfg, "compression_topk", None) is None:
            ratio_start = cfg.compression_topk_ratio
            lo = min(ratio_start, ratio_end)
            hi = max(ratio_start, ratio_end)
            comp.topk_ratio = random.uniform(lo, hi)

        temp_end = getattr(cfg, "compression_gumbel_temperature_end", None)
        if temp_end is not None:
            temp_start = cfg.compression_gumbel_temperature
            if temp_end < temp_start and step < anneal_end_step:
                progress = step / max(anneal_end_step, 1)
                comp.gumbel_temperature = temp_start * (temp_end / temp_start) ** progress
            elif step >= anneal_end_step:
                comp.gumbel_temperature = temp_end

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)

        comp = self._get_compression_module(model)
        if comp is not None and labels is not None:
            # Labels travel through forward() so _compress_multimodal_tokens
            # can resize them together with the compressed embedding.
            # Do NOT let the LLM compute loss internally — _build_labels
            # already pre-shifts labels by one position, and the LLM would
            # apply a second shift, causing a double-shift error.
            inputs["labels"] = labels

        if not self.args.use_lora:
            outputs = self.model(data=inputs, use_cache=False)
        else:
            with self.model._enable_peft_forward_hooks(**inputs):
                outputs = self.model.base_model(data=inputs, use_cache=False)

        if comp is not None and labels is not None:
            # Retrieve the compression-resized labels from the model output
            compressed_labels = inputs.get("labels", labels)
            loss_fct = nn.CrossEntropyLoss()
            logits = outputs.logits.view(-1, outputs.logits.size(-1)).contiguous()
            compressed_labels = compressed_labels.view(-1).long().contiguous().to(logits.device)
            loss = loss_fct(logits, compressed_labels)
        elif labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            _model = self.model.module if hasattr(self.model, "module") else self.model
            logits = outputs.logits.view(-1, _model.config.vocab_size).contiguous()
            labels = labels.view(-1).long().contiguous().to(logits.device)
            loss = loss_fct(logits, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = all(inputs.get(k) is not None for k in self.label_names) if self.label_names else False
        return_loss = inputs.get("return_loss", self.can_return_loss)
        loss_without_labels = True if not self.label_names and return_loss else False

        inputs = self._prepare_inputs(inputs)
        _model = self.model.module if hasattr(self.model, "module") else self.model
        if ignore_keys is None and hasattr(_model, "config"):
            ignore_keys = getattr(_model.config, "keys_to_ignore_at_inference", [])

        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(n) for n in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                logits = tuple(v for k, v in outputs.items() if k not in (ignore_keys or []) + ["loss"]) if isinstance(outputs, dict) else outputs[1:]
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                logits = tuple(v for k, v in outputs.items() if k not in (ignore_keys or [])) if isinstance(outputs, dict) else outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return (loss, logits, labels)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        self._anneal_compression(model)

        sample_indices = inputs.pop("sample_indices", None)
        seq_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else -1
        if sample_indices is not None:
            print(f"[Step {self.state.global_step}][RANK {self.args.local_rank}] sample_indices={sample_indices}, seq_len={seq_len}", flush=True)

        max_effective_len = 27000
        comp = self._get_compression_module(model)
        if comp is not None and seq_len > 0:
            effective_len = seq_len * comp.topk_ratio
            if effective_len > max_effective_len:
                comp.topk_ratio = max_effective_len / seq_len

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported = (PreTrainedModel,)
        if is_peft_available():
            from peft import PeftModel as _PM
            supported = (PreTrainedModel, _PM)

        if not isinstance(self.model, supported):
            if state_dict is None:
                state_dict = self.model.state_dict()
            if isinstance(unwrap_model(self.model), supported):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                )
            else:
                logger.info("Model is not PreTrainedModel; saving state dict only.")
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


# ======================================================================
# helpers
# ======================================================================

local_rank = 0

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def get_parameter_number(model):
    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        if n == 0 and hasattr(p, "ds_numel"):
            n = p.ds_numel
        total += n
        if p.requires_grad:
            trainable += n
    return {"Total": total, "Trainable": trainable}


def safe_save_model_for_hf_trainer(trainer, output_dir, bias="none"):
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer.save_model(output_dir)


# ======================================================================
# main
# ======================================================================

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if getattr(training_args, "deepspeed", None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
            logging.warning("FSDP / ZeRO-3 are incompatible with QLoRA.")

    # ---- pre-load: override compression config so model __init__ sees it ----
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    if training_args.compression_enabled is not None:
        model_config.compression_enabled = training_args.compression_enabled

    for attr in (
        "compression_topk",
        "compression_topk_ratio",
        "compression_topk_ratio_end",
        "compression_text_guided_weight",
        "compression_gumbel_temperature",
        "compression_gumbel_temperature_end",
    ):
        value = getattr(training_args, attr)
        if value is not None:
            setattr(model_config, attr, value)
    model_config.compression_debug_log = training_args.compression_debug_log

    if getattr(model_config, "compression_enabled", False):
        rank0_print(
            "Compression enabled: "
            f"topk={getattr(model_config, 'compression_topk', None)}, "
            f"ratio={getattr(model_config, 'compression_topk_ratio', None)}, "
            f"debug_log={getattr(model_config, 'compression_debug_log', False)}"
        )
    else:
        rank0_print("Compression disabled.")

    # ---- load model (audio ON, tts OFF) ----
    model_config.init_vision = True
    model_config.init_audio = True
    model_config.init_tts = False
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    # ---- freeze / unfreeze components ----
    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
        if hasattr(model, "resampler"):
            model.resampler.requires_grad_(False)
        rank0_print("Vision encoder frozen.")

    if not training_args.tune_llm:
        model.llm.requires_grad_(False)
        rank0_print("LLM frozen.")

    if not training_args.tune_audio:
        model.apm.requires_grad_(False)
        if hasattr(model, "audio_projection_layer"):
            model.audio_projection_layer.requires_grad_(False)
        rank0_print("Audio encoder frozen.")

    if hasattr(model, "compression_module") and model.compression_module is not None:
        if not training_args.tune_compression:
            model.compression_module.requires_grad_(False)
            rank0_print("Compression module frozen.")
        else:
            model.compression_module.requires_grad_(True)
            rank0_print("Compression module trainable.")

    # ---- LoRA ----
    if training_args.use_lora:
        if training_args.tune_llm:
            raise ValueError("Cannot use LoRA and tune_llm=True simultaneously.")

        rank0_print("Using LoRA for fine-tuning.")
        for p in model.llm.parameters():
            p.requires_grad = False

        modules_to_save = ["embed_tokens", "resampler"]
        if training_args.tune_vision:
            modules_to_save.append("vpm")
        if training_args.tune_audio:
            modules_to_save.extend(["apm", "audio_projection_layer", "audio_avg_pooler"])
        if training_args.tune_compression and hasattr(model, "compression_module") and model.compression_module is not None:
            modules_to_save.append("compression_module")

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, "get_input_embeddings"):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=training_args.gradient_checkpointing,
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))

    # ---- prepare processor ----
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    # ---- dataset ----
    rank0_print("Loading training data …")
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = AudioVideoDataset(
        train_json,
        processor,
        max_length=training_args.model_max_length,
        max_slice_nums=training_args.max_slice_nums,
        use_image_id=training_args.use_image_id,
        max_num_frames=training_args.max_num_frames,
    )

    eval_dataset = None
    if data_args.eval_data_path:
        rank0_print("Loading evaluation data …")
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = AudioVideoDataset(
            eval_json,
            processor,
            max_length=training_args.model_max_length,
            max_slice_nums=training_args.max_slice_nums,
            use_image_id=training_args.use_image_id,
            max_num_frames=training_args.max_num_frames,
        )
    # from fpdb import ForkedPdb; ForkedPdb().set_trace()
    # ---- trainer ----
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    trainer = AVTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(
            av_data_collator,
            max_length=training_args.model_max_length,
        ),
    )

    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt is not None:
        resume_ckpt = True if resume_ckpt.lower() == "true" else resume_ckpt
    if training_args.local_rank <= 0:
        print(f"[Resume] raw arg = {training_args.resume_from_checkpoint!r}, "
              f"resolved = {resume_ckpt!r}, output_dir = {training_args.output_dir!r}")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias,
    )


if __name__ == "__main__":
    train()
