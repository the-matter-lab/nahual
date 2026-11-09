# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from evals.api import Task
from evals.metrics.geometry import first_xyz

# DiffCSP (Jiao et al., NeurIPS 2023, §5.1) reports MP-20 match rate and
# matched-only RMSE with StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10).
# We use the same tolerances, which are looser than pymatgen's defaults.
DIFFCSP_MATCHER_TOL = {"ltol": 0.3, "stol": 0.5, "angle_tol": 10}


class CspTask(Task):
    def __init__(self, config):
        self.config = config

    def setup(self, args=None):
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from nahual.inorganic import match_and_rms
        self._matcher = StructureMatcher(**DIFFCSP_MATCHER_TOL)
        self._match_and_rms = match_and_rms

    def metric(self, ground_truth, prediction, prompt=None):
        truth_xyz = first_xyz(ground_truth)
        pred_xyz = first_xyz(prediction)

        # A sample decoded without a lattice is an invalid crystal: a non-match.
        if truth_xyz is None or pred_xyz is None or not pred_xyz.has_lattice:
            return {'score': float('inf'), 'valid': False,
                    'info': {'match': False, 'rms': None, 'msg': 'Missing data'}}

        true_struc = truth_xyz.to_pymatgen()
        pred_struc = pred_xyz.to_pymatgen()

        if true_struc is None or pred_struc is None:
            return {'score': float('inf'), 'valid': False,
                    'info': {'match': False, 'rms': None, 'msg': 'Conversion failed'}}

        match, rms = self._match_and_rms(true_struc, pred_struc, matcher=self._matcher)
        # Score is the RMS, or inf if there is no match.
        score = float(rms) if rms is not None else float('inf')
        return {'score': score, 'valid': True,
                'info': {'match': bool(match), 'rms': rms, 'msg': 'Success'}}

    def aggregate(self, results, duration):
        top1_matches, top20_matches = [], []
        top1_rmses, top20_rmses = [], []

        for res in results:
            samples = res['samples']
            matches = [s['metric']['info']['match'] for s in samples]
            rmses = [s['metric']['info']['rms'] for s in samples]

            if matches:
                top1_matches.append(matches[0])
                top1_rmses.append(rmses[0])
                top20_matches.append(any(matches))
                valid_rmses = [r for r in rmses if r is not None]
                top20_rmses.append(min(valid_rmses) if valid_rmses else None)

        def calc_stats(matches, rmses):
            if not matches:
                return 0.0, float('nan')
            match_rate = float(np.mean(matches))
            valid = [r for r in rmses if r is not None]
            avg_rmse = float(np.mean(valid)) if valid else float('nan')
            return match_rate, avg_rmse

        t1_match, t1_rmse = calc_stats(top1_matches, top1_rmses)
        t20_match, t20_rmse = calc_stats(top20_matches, top20_rmses)

        return {
            "top1_match_rate": t1_match,
            "top1_rmse": t1_rmse,
            "top20_match_rate": t20_match,
            "top20_rmse": t20_rmse,
            "matcher_tolerances": dict(DIFFCSP_MATCHER_TOL),
            "duration_seconds": duration,
        }
