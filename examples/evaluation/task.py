# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal contributor example: score exact text answers, with no chemistry deps."""
from evals.api import Task


class ExactText(Task):
    def load_dataset(self):
        # Replace with a dataset loader, a local file, or the inherited chemistry loader.
        return [dict(instruction=[{"type":"text","text":f"Name {formula}"}],
                     response=[{"type":"text","text":answer}])
                for formula,answer in [("H2O","water"),("CH4","methane")]]

    def metric(self, ground_truth, prediction, prompt=None):
        correct = ' '.join(ground_truth).strip() == ' '.join(prediction).strip()
        return {'score': float(correct), 'valid': True, 'info': {}}

    def aggregate(self, results, duration):
        values = [sample['metric']['score'] for result in results for sample in result['samples']]
        return {'accuracy': sum(values) / len(values) if values else 0.0,
                'n_samples': len(values), 'duration_seconds': duration}
