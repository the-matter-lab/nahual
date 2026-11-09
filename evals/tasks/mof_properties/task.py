# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from evals.api import Task
from evals.tasks.mof_properties.scoring import qm9_prop_metric, qm9_prop_aggregate


class MOFPropertyTask(Task):
    def metric(self, ground_truth, prediction, prompt=None):
        return qm9_prop_metric(ground_truth, prediction, self.config.name)

    def aggregate(self, results, duration):
        return qm9_prop_aggregate(results, self.config.name, duration)
