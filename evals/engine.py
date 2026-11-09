# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate-then-score pipeline shared by all generation backends."""
from evals.api import Backend, Task


def generate(task: Task, backend: Backend, args):
    return backend.generate(task, task.load_dataset(), args)


def score(task: Task, predictions, save_path, duration, args, *, model=None, peak_gpu_memory_gb=None):
    from evals.artifacts import save_predictions, save_metrics, save_task_artifacts
    task.setup(args=args)
    # Load the dataset only for tasks that override post_process.
    if type(task).post_process is not Task.post_process:
        task.post_process(predictions, task.load_dataset(), save_path)
    for result in predictions:
        for sample in result['samples']:
            sample['metric'] = task.metric(result['ground_truth'], sample['prediction'], prompt=result['prompt'])
    metrics = task.aggregate(predictions, duration)
    save_predictions(save_path, predictions)
    save_metrics(save_path, metrics, task.config.name,
                 checkpoint=model if model is not None else getattr(args,'checkpoint',None),
                 config=task.config, duration=duration, num_prompts=len(predictions),
                 peak_gpu_memory_gb=peak_gpu_memory_gb)
    save_task_artifacts(save_path, task.artifacts())
    return metrics


def run(task: Task, backend: Backend, args):
    generated = generate(task, backend, args)
    if generated.save_path is None or getattr(args,'phase',None) == 'generate':
        return generated, None
    metrics = score(task, generated.predictions, generated.save_path, generated.duration,
                    args, model=getattr(backend,'identity',None),
                    peak_gpu_memory_gb=generated.peak_gpu_memory_gb)
    return generated, metrics
