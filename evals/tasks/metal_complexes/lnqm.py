# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from evals.api import Task
from evals.metrics.geometry import (
    connectivity_match,
    rmsd_metric,
    topk_connectivity_match_aggregation,
    topk_rmsd_aggregation,
)


class LnqmTask(Task):
    def __init__(self, config):
        self.config = config

    def metric(self, ground_truth, prediction, prompt=None):
        rmsd_res = rmsd_metric(ground_truth, prediction)
        connectivity_res = connectivity_match(ground_truth, prediction)

        info = dict(rmsd_res.get('info', {}))
        info['same_connectivity'] = connectivity_res['match']
        if connectivity_res['info'].get('msg'):
            info['connectivity_msg'] = connectivity_res['info']['msg']

        return {
            'score': rmsd_res['score'],
            'valid': rmsd_res['valid'],
            'info': info,
        }

    def aggregate(self, results, duration):
        metrics = topk_rmsd_aggregation(results, self.config.k_values, duration)
        metrics.update(topk_connectivity_match_aggregation(results, self.config.k_values))
        return metrics
