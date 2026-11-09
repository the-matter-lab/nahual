# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading for inference and evaluation.

--checkpoint takes a training checkpoint (step_N.pt with model, EMA, optimizer,
scheduler and config), as released on the Hugging Face Hub. The EMA weights are
used unless ema=False (--no-ema). A directory with model.safetensors + config.json
also loads.
"""
import json
from pathlib import Path
import warnings
import numpy as np
import torch
from nahual.config import Config
from nahual.model import NextTokenDiffusion, DiffusionMLP, Qwen3RotaryEmbedding
CHAT_BEFORE = np.array([151644, 872, 198], dtype=np.int64)
CHAT_AFTER = np.array([151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271], dtype=np.int64)
MODEL_CLASSES = {'next_token': NextTokenDiffusion, 'diffmlp': DiffusionMLP}
EXPORT_WEIGHTS = 'model.safetensors'
EXPORT_CONFIG = 'config.json'


def _build(config, state_dict, device, tied=None):
    if config.model_impl not in MODEL_CLASSES:
        raise ValueError(f'Unsupported architecture: {config.model_impl}')
    with torch.device('meta'):
        model = MODEL_CLASSES[config.model_impl](config)
    tied = tied or {}
    missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=not tied)
    if set(missing) != set(tied) or unexpected:
        raise RuntimeError(f'Checkpoint does not match {config.model_impl}: missing={missing}, unexpected={unexpected}')
    for alias, source in tied.items():  # e.g. lm_head.weight shares embed_tokens.weight
        owner, _, name = alias.rpartition('.')
        src_owner, _, src_name = source.rpartition('.')
        setattr(model.get_submodule(owner), name, getattr(model.get_submodule(src_owner), src_name))
    # RoPE frequencies are nonpersistent and therefore absent from the checkpoint.
    model.rotary_emb = Qwen3RotaryEmbedding(config)
    return model.to(device).eval()


def is_exported_checkpoint(path):
    path = Path(path)
    return (path / EXPORT_WEIGHTS).is_file() if path.is_dir() else path.suffix == '.safetensors'


def load_exported_model(path, device='cuda'):
    """Load a release directory (model.safetensors + config.json) or its safetensors file."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    path = Path(path)
    weights = path / EXPORT_WEIGHTS if path.is_dir() else path
    info = json.loads((weights.parent / EXPORT_CONFIG).read_text())
    with safe_open(str(weights), framework='pt') as handle:
        metadata = handle.metadata() or {}
    tied = json.loads(metadata.get('tied_parameters', '{}'))
    return _build(Config.from_dict(info), load_file(str(weights)), device, tied)


def load_checkpoint_model(path, device='cuda', ema=True):
    if is_exported_checkpoint(path):
        if not ema:
            warnings.warn('Exported checkpoints contain only EMA weights; --no-ema is ignored.', stacklevel=2)
        return load_exported_model(path, device)
    ckpt = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    config = Config.from_dict(ckpt['config'])
    key = 'ema_state_dict' if ema and 'ema_state_dict' in ckpt else 'model_state_dict'
    return _build(config, ckpt[key], device)

def load_model(checkpoint_path, device='cuda'):
    return (load_checkpoint_model(checkpoint_path, device, False),
            load_checkpoint_model(checkpoint_path, device, True))


class CheckpointBackend:
    name = "checkpoint"

    def generate(self, task, dataset, args):
        """Generate predictions for the task and save them."""
        config = task.config
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for generation phase")

        from evals.api import Generation
        from evals.backends.batched import generate_predictions
        from evals.artifacts import make_save_path, load_generate_meta_dict, save_predictions, save_generate_meta
        import time
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        start_time = time.time()

        # Load model
        from evals.catalog import resolve
        loader = resolve(args.model_loader) if getattr(args,'model_loader',None) else load_checkpoint_model
        gen_model = loader(args.checkpoint, device=args.device, ema=not args.no_ema)


        # Create save path
        save_path, run_name = make_save_path(config.output_dir, args.run_name, config.dataset)
        args.run_name = run_name
        args.save_path = save_path

        # Track peak GPU memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # The task's custom_generate, if set, replaces the default loop.
        if callable(task.custom_generate):
            predictions = task.custom_generate(gen_model, dataset, config, args)
        else:
            predictions = generate_predictions(gen_model, dataset, config, args, task=task)

        # Read peak GPU memory
        peak_gpu_memory_gb = None
        if torch.cuda.is_available():
            peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        duration = time.time() - start_time
        print(f"Generation took {duration:.2f}s ({duration/60:.2f}min)")
        if peak_gpu_memory_gb is not None:
            print(f"Peak GPU memory: {peak_gpu_memory_gb:.2f} GB")

        # Never save a preempted generation as complete.
        saved_meta = load_generate_meta_dict(save_path)
        if saved_meta and saved_meta.get("complete") is False:
            raise InterruptedError("Generation interrupted; partial predictions retained for resume")

        # Save predictions and generation metadata
        save_predictions(save_path, predictions)
        save_generate_meta(
            save_path,
            duration,
            peak_gpu_memory_gb,
            complete=True,
            num_predictions=len(predictions),
            total_expected=len(predictions),
        )

        return Generation(predictions, save_path, duration, peak_gpu_memory_gb)


