# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save and load evaluation outputs: predictions, metrics and generation metadata."""

import json
import os
import subprocess
from datetime import datetime

import numpy as np
import torch


def make_save_path(output_dir, run_name=None, dataset=None):
    """Create and return the save path for an eval run."""
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.basename(dataset).split('.')[0] if dataset else "eval"
        run_name = f"{dataset_name}_{timestamp}"

    save_path = os.path.join(output_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path, run_name


def _atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _atomic_json_dump(obj, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=4, default=_json_default)
    os.replace(tmp_path, path)


def save_predictions(save_path, predictions):
    """Save predictions.pt with XYZ objects preserved."""
    path = os.path.join(save_path, "predictions.pt")
    _atomic_torch_save(predictions, path)
    print(f"Saved predictions to {path}")


def load_predictions(save_path):
    """Load predictions.pt."""
    path = os.path.join(save_path, "predictions.pt")
    return torch.load(path, weights_only=False)


def _get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_metrics(save_path, metrics, task_name, checkpoint, config, duration, num_prompts,
                 peak_gpu_memory_gb=None):
    """Save metrics.json with the run's settings and provenance."""
    output = {
        "task": task_name,
        "checkpoint": checkpoint,
        "dataset": config.dataset,
        "git_commit": _get_git_commit(),
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration,
        "peak_gpu_memory_gb": round(peak_gpu_memory_gb, 2) if peak_gpu_memory_gb is not None else None,
        "hyperparams": {
            "max_new_tokens": config.max_new_tokens,
            "samples_per_prompt": config.samples_per_prompt,
            "prompt_batch_size": config.prompt_batch_size,
            "num_steps": config.num_steps,
            "num_examples": config.num_examples,
            "k_values": config.k_values,
            "task_options": config.extra,
        },
        "num_prompts": num_prompts,
        "metrics": metrics,
    }

    json_path = os.path.join(save_path, "metrics.json")
    _atomic_json_dump(output, json_path)
    print(f"Saved metrics to {json_path}")


def save_generate_meta(save_path, duration, peak_gpu_memory_gb=None, complete=True,
                       num_predictions=None, total_expected=None):
    """Save generation metadata (duration, peak memory) alongside predictions."""
    meta = {
        "duration_seconds": duration,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "complete": complete,
    }
    if num_predictions is not None:
        meta["num_predictions"] = num_predictions
    if total_expected is not None:
        meta["total_expected"] = total_expected
    path = os.path.join(save_path, "generate_meta.json")
    _atomic_json_dump(meta, path)


def load_generate_meta(save_path):
    """Load generation metadata. Returns (duration, peak_gpu_memory_gb)."""
    meta = load_generate_meta_dict(save_path)
    return meta.get("duration_seconds", 0), meta.get("peak_gpu_memory_gb")


def load_generate_meta_dict(save_path):
    """Load generation metadata as a dict."""
    path = os.path.join(save_path, "generate_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def is_generation_complete(save_path, expected_num_predictions=None):
    """Return True when a shard/run has a fully written predictions checkpoint."""
    predictions_path = os.path.join(save_path, "predictions.pt")
    if not os.path.exists(predictions_path):
        return False

    meta = load_generate_meta_dict(save_path)
    complete = meta.get("complete")
    num_predictions = meta.get("num_predictions")

    if complete is False:
        return False
    if expected_num_predictions is not None and num_predictions is not None:
        return num_predictions >= expected_num_predictions

    # Without a completion flag, an existing predictions.pt counts as complete.
    return True


def load_metrics(save_path):
    """Load metrics.json."""
    json_path = os.path.join(save_path, "metrics.json")
    with open(json_path) as f:
        return json.load(f)


def save_task_artifacts(save_path, artifacts):
    """Save the task's extra .pt artifacts in the run directory."""
    for name, value in artifacts.items():
        if os.path.basename(name) != name or not name.endswith('.pt'):
            raise ValueError(f'Task artifact must be a local .pt filename: {name!r}')
        _atomic_torch_save(value, os.path.join(save_path, name))
