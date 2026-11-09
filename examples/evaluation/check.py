# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with ``python -m examples.evaluation.check``. Needs no model or data."""
from examples.evaluation.task import ExactText
from evals.api import TaskConfig


def main():
    task = ExactText(TaskConfig(name='example', dataset='unused', output_dir='unused'))
    samples = [{'prediction': ['water']}, {'prediction': ['methane']}]
    for sample in samples:
        sample['metric'] = task.metric(['water'], sample['prediction'])
    results = [{'prompt_index': 0, 'prompt': ['Name H2O'], 'ground_truth': ['water'], 'samples': samples}]
    metrics = task.aggregate(results, duration=0)
    assert metrics['accuracy'] == 0.5
    print(metrics)


if __name__ == '__main__':
    main()
