# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NEB task with constrained decoding.

Same as ``neb``, but all ground-truth response tokens are teacher-forced and
only the coordinates are sampled. This measures geometry error without
format or length errors. The forced atom list is the same in every frame and
in the prompt, so it gives the model no extra information.
"""

import numpy as np

from evals.tasks.reaction_paths.neb import NebTask
from nahual.tokens import STOP, tokenize_seq


class NebConstrainedTask(NebTask):
    def build_forced_response(self, prompt_segs, ground_truth_segs):
        """Ground-truth response tokens followed by ``STOP``, forced after the prompt."""
        gt_tokens, _ = tokenize_seq(ground_truth_segs)
        return np.concatenate([gt_tokens, np.array([STOP], dtype=np.int64)], axis=0)
