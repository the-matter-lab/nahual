# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QM9 EDM property prediction tasks qm9_edm_prop_{alpha,gap,homo,lumo,mu,Cv}.

Reports MAE on the test set, parsing values from the model's text output.
"""

import re

import numpy as np

from evals.metrics.geometry import full_text
from evals.api import Task


# Property configs: (display_name, unit, regex_pattern)
EDM_PROP_CONFIGS = {
    "alpha": ("polarizability",  "Bohr^3",      re.compile(r'(-?[\d\.]+)\s*Bohr\^3')),
    "gap":   ("HOMO-LUMO gap",   "meV",          re.compile(r'(-?[\d\.]+)\s*meV')),
    "homo":  ("HOMO energy",     "meV",          re.compile(r'(-?[\d\.]+)\s*meV')),
    "lumo":  ("LUMO energy",     "meV",          re.compile(r'(-?[\d\.]+)\s*meV')),
    "mu":    ("dipole moment",   "D",            re.compile(r'(-?[\d\.]+)\s*D\b')),
    "Cv":    ("heat capacity",   "cal/(mol*K)",  re.compile(r'(-?[\d\.]+)\s*cal/\(mol\*K\)')),
}


def _extract(text, pattern):
    if not isinstance(text, str):
        return None
    m = pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _prop_metric(ground_truth, prediction, prop_key):
    """Placeholder. Metrics are computed from the text in aggregation."""
    return {'score': 0.0, 'valid': True, 'info': {'msg': 'ok'}}


def _prop_aggregate(results, prop_key, duration):
    display_name, unit, pattern = EDM_PROP_CONFIGS[prop_key]

    ae_at_1 = []
    ae_at_n = []
    truths_at_1 = []
    preds_at_1 = []
    n_parse_fail_truth = 0
    n_parse_fail_pred = 0
    n_total = 0

    for res in results:
        truth_text = full_text(res['ground_truth'])
        truth_val = _extract(truth_text, pattern)
        if truth_val is None:
            n_parse_fail_truth += 1
            continue

        n_total += 1
        sample_errors = []
        sample_preds = []

        for sample in res['samples']:
            pred_text = full_text(sample['prediction'])
            pred_val = _extract(pred_text, pattern)
            if pred_val is None:
                n_parse_fail_pred += 1
                sample_errors.append(None)
                sample_preds.append(None)
            else:
                sample_errors.append(abs(pred_val - truth_val))
                sample_preds.append(pred_val)

        if sample_errors and sample_errors[0] is not None:
            ae_at_1.append(sample_errors[0])
            truths_at_1.append(truth_val)
            preds_at_1.append(sample_preds[0])

        valid_errors = [e for e in sample_errors if e is not None]
        if valid_errors:
            ae_at_n.append(float(np.mean(valid_errors)))

    n_samples = len(results[0]['samples']) if results else 0

    # R² score
    r2 = float('nan')
    if truths_at_1:
        y_true, y_pred = np.array(truths_at_1), np.array(preds_at_1)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot > 0:
            r2 = float(1 - ss_res / ss_tot)

    ae_arr = np.array(ae_at_1) if ae_at_1 else np.array([])
    rmse = float(np.sqrt(np.mean(ae_arr ** 2))) if ae_at_1 else float('nan')

    metrics = {
        f"mae@1_mean_{unit}": float(np.mean(ae_at_1)) if ae_at_1 else float('nan'),
        f"mae@1_median_{unit}": float(np.median(ae_at_1)) if ae_at_1 else float('nan'),
        f"rmse@1_{unit}": rmse,
        "r2@1": r2,
        f"mae@{n_samples}_mean_{unit}": float(np.mean(ae_at_n)) if ae_at_n else float('nan'),
        f"mae@{n_samples}_median_{unit}": float(np.median(ae_at_n)) if ae_at_n else float('nan'),
        "n_examples": n_total,
        "n_valid_at_1": len(ae_at_1),
        "n_valid_at_n": len(ae_at_n),
        "n_parse_fail_truth": n_parse_fail_truth,
        "n_parse_fail_pred": n_parse_fail_pred,
        "samples_per_prompt": n_samples,
        "duration_seconds": duration,
    }

    print(f"\n{display_name} prediction results ({n_total} examples):")
    print(f"  MAE@1:  mean={metrics[f'mae@1_mean_{unit}']:.4f} {unit},  median={metrics[f'mae@1_median_{unit}']:.4f} {unit}")
    print(f"  RMSE@1: {rmse:.4f} {unit}")
    print(f"  R²@1:   {r2:.4f}")
    if n_parse_fail_truth:
        print(f"  Warning: {n_parse_fail_truth} truth values could not be parsed")
    if n_parse_fail_pred:
        print(f"  Warning: {n_parse_fail_pred} predictions could not be parsed")

    return metrics


# --- Register all 6 tasks ---

def _make_prop_task_class(prop_key):
    class EdmPropTask(Task):
        def __init__(self, config):
            self.config = config

        def setup(self, args=None):
            pass

        def metric(self, ground_truth, prediction, prompt=None):
            return _prop_metric(ground_truth, prediction, prop_key)

        def aggregate(self, results, duration):
            return _prop_aggregate(results, prop_key, duration)

    EdmPropTask.__name__ = f"QM9EdmProp{prop_key.capitalize()}Task"
    return EdmPropTask


def create_task(config):
    return _make_prop_task_class(config.extra["property"])(config)
