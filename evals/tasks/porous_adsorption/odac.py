# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

from evals.api import Task
from evals.tasks.porous_adsorption.scoring import odac_aggregate, odac_metric
from nahual.tokens import make_seq


class OdacTask(Task):
    def __init__(self, config):
        self.config = config

    def setup(self, args=None):
        pass

    @staticmethod
    def _load_references(row):
        all_responses = row.get("all_responses")
        if not all_responses:
            return None

        references = []
        for response in all_responses:
            if isinstance(response, str):
                response = json.loads(response)
            references.append(make_seq(response))
        return {"responses": references}

    def post_process(self, predictions, dataset, save_path):

        if "all_responses" not in getattr(dataset, "column_names", []):
            return predictions

        for result in predictions:
            row = dataset[result["prompt_index"]]
            references = self._load_references(row)
            if references is not None:
                result["ground_truth"] = references

        return predictions

    def metric(self, ground_truth, prediction, prompt=None):
        return odac_metric(ground_truth, prediction, prompt)

    def aggregate(self, results, duration):
        return odac_aggregate(results, self.config.k_values, duration)
