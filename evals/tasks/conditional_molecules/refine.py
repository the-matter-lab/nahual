# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run self-oracle selection after conditional molecule evaluation."""
import sys


def commands(spec, request, limit):
    return [('self_oracle', [sys.executable, '-m', 'evals.tasks.conditional_molecules.self_oracle',
             '--checkpoint', request['checkpoint'], '--props', spec.config.extra['property'],
             '--run_suffix', request['run_name'], '--pred-run-name', request['run_name'],
             '--num_examples', str(limit), '--self_batch_size', str(request.get('self_batch_size', 64))])]
