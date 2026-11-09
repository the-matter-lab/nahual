# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from evals.api import Task
from evals.metrics.periodic import gmae_metric, gmae_aggregate


class SurfaceAdsorptionTask(Task):
    def metric(self, ground_truth, prediction, prompt=None):
        return gmae_metric(ground_truth, prediction, prompt)

    def aggregate(self, results, duration):
        return gmae_aggregate(results, self.config.k_values, duration)
