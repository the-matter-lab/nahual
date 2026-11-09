# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one checkpoint on one task: `python -m evals run --backend checkpoint`."""
import argparse
import json
import os
from evals.api import TaskConfig
from evals.catalog import get_task, get_spec
from evals.artifacts import (make_save_path, save_predictions, load_predictions, load_generate_meta)

def run_generate(task, config, args):
    from evals.backends.checkpoint import CheckpointBackend
    from evals.engine import generate
    result = generate(task, CheckpointBackend(), args)
    return result.predictions, result.save_path, result.duration, result.peak_gpu_memory_gb

def run_metrics(task, config, predictions, save_path, duration, args, peak_gpu_memory_gb=None):
    from evals.engine import score
    return score(task, predictions, save_path, duration, args, peak_gpu_memory_gb=peak_gpu_memory_gb)

def get_parser():
    parser = argparse.ArgumentParser(description="Nahual evaluation pipeline")
    parser.add_argument("--task", type=str, required=True, help="Task name (see `python -m evals list`)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--model-loader", default=None, help="Optional module:loader(path, device, ema) for another model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--run_name", "--run-name", type=str, default=None, help="Name for this eval run")
    parser.add_argument("--run_dir", "--run-dir", type=str, default=None, help="Existing run dir (for --phase metrics)")
    parser.add_argument("--no_ema", "--no-ema", action="store_true", help="Use base model instead of EMA")
    parser.add_argument("--phase", type=str, default=None, choices=["generate", "metrics"],
                        help="Run only one phase (default: both)")

    # Override task defaults
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", "--output-dir", type=str, default=None)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=None)
    parser.add_argument("--samples_per_prompt", "--samples-per-prompt", type=int, default=None)
    parser.add_argument("--prompt_batch_size", "--prompt-batch-size", type=int, default=None)
    parser.add_argument("--num_examples", "--num-examples", type=int, default=None)
    parser.add_argument("--num_steps", "--num-steps", type=int, default=None)
    parser.add_argument("--k_values", "--k-values", type=lambda s: [int(k) for k in s.split(',')], default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--merge_shards", "--merge-shards", type=str, default=None,
                        help="Comma-separated shard run names to merge into this run")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated dataset indices (e.g., '0,5,7,12'). "
                             "Takes precedence over --offset/--num_examples.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature for discrete tokens. 0 is greedy. "
                             "Default: the task's default_temperature, else 1.0.")

    parser.add_argument("--task-options", type=json.loads, default={}, help="JSON object of benchmark-specific settings")
    return parser

def _apply_overrides(config: TaskConfig, args):
    """Apply CLI overrides to task config."""
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    if args.samples_per_prompt is not None:
        config.samples_per_prompt = args.samples_per_prompt
    if args.prompt_batch_size is not None:
        config.prompt_batch_size = args.prompt_batch_size
    if args.num_examples is not None:
        config.num_examples = args.num_examples
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if getattr(args, 'k_values', None) is not None:
        if any(k < 1 or k > config.samples_per_prompt for k in args.k_values):
            raise ValueError('k_values must be between 1 and samples_per_prompt')
        config.k_values = args.k_values

def run_eval_pipeline(args=None):
    """Main entry point: run the full eval pipeline."""
    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    # Lazily import only the task needed for this run.
    get_spec(args.task).require_backend("checkpoint")
    task = get_task(args.task)
    config = task.config

    # Apply CLI overrides
    _apply_overrides(config, args)
    if not isinstance(args.task_options, dict):
        raise ValueError("--task-options must be a JSON object")
    config.extra.update(args.task_options)

    # Parse --indices into list[int]
    if getattr(args, 'indices', None):
        args.indices = [int(x) for x in args.indices.split(",")] if isinstance(args.indices,str) else args.indices
    else:
        args.indices = None

    phase = args.phase

    if phase == "metrics":
        # Metrics-only: load existing predictions (or merge shards)
        merge_shards = getattr(args, 'merge_shards', None)

        if merge_shards:
            # Merge predictions from multiple shards into one
            shard_names = merge_shards.split(",")
            save_path, run_name = make_save_path(config.output_dir, args.run_name, config.dataset)
            args.run_name = run_name

            print(f"Merging {len(shard_names)} shards...")
            all_predictions = []
            shard_durations = []
            shard_peak_mems = []
            for shard_name in shard_names:
                shard_path = os.path.join(config.output_dir, shard_name)
                shard_preds = load_predictions(shard_path)
                all_predictions.extend(shard_preds)
                d, m = load_generate_meta(shard_path)
                shard_durations.append(d)
                if m is not None:
                    shard_peak_mems.append(m)
                print(f"  {shard_name}: {len(shard_preds)} examples ({d:.0f}s)")

            # Sort by prompt_index to restore original order
            all_predictions.sort(key=lambda x: x['prompt_index'])
            indices = [p['prompt_index'] for p in all_predictions]
            if len(indices) != len(set(indices)):
                raise ValueError('Overlapping evaluation shards')
            if args.indices is not None and set(indices) != set(args.indices):
                raise ValueError('Merged predictions do not match the requested subset')
            print(f"Total: {len(all_predictions)} examples")

            # Save merged predictions
            save_predictions(save_path, all_predictions)
            predictions = all_predictions
            # Shards ran in parallel, so wall-clock time is the slowest shard.
            duration = max(shard_durations) if shard_durations else 0
            peak_gpu_memory_gb = max(shard_peak_mems) if shard_peak_mems else None
        else:
            run_dir = args.run_dir
            if run_dir is None:
                raise ValueError("--run_dir or --merge_shards required for --phase metrics")
            save_path = run_dir
            predictions = load_predictions(run_dir)
            duration, peak_gpu_memory_gb = load_generate_meta(run_dir)
            if duration == 0:
                # Fallback: try loading from existing metrics.json
                try:
                    from evals.artifacts import load_metrics as _load_metrics
                    old = _load_metrics(run_dir)
                    duration = old.get("duration_seconds", 0)
                except Exception:
                    pass

        run_metrics(task, config, predictions, save_path, duration, args, peak_gpu_memory_gb=peak_gpu_memory_gb)
    elif phase == "generate":
        # Generate only
        run_generate(task, config, args)
    else:
        # Both phases
        predictions, save_path, duration, peak_gpu_memory_gb = run_generate(task, config, args)
        run_metrics(task, config, predictions, save_path, duration, args, peak_gpu_memory_gb=peak_gpu_memory_gb)


if __name__ == "__main__":
    run_eval_pipeline()
