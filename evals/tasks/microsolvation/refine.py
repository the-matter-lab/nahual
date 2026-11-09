# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Crude xTB relaxation of microsolvation samples (final requests, or when requested)."""
from pathlib import Path
import sys


def commands(spec, request, limit):
    if not request['final'] and not request.get('relax_microsolvation', False):
        return []
    return [('xtb_relax', [sys.executable, '-m', 'evals.tasks.microsolvation.relax',
             '--predictions_pt', str(Path(spec.config.output_dir) / request['run_name'] / 'predictions.pt'),
             '--run_name', request['run_name'] + '_xtb_relaxed_crude', '--nprocs', '16'])]
