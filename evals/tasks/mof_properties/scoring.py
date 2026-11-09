# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared logic for property prediction evaluation (QM9 + MOF tasks)."""

import re
import numpy as np
from evals.metrics.geometry import full_text


# Numbers must start the string or follow whitespace, so corrupt decodings
# like '4.5 a93 D' or '2.-636 D' do not parse.
PROPERTY_PATTERNS = {
    'qm9_prop_gap': {
        'pattern': re.compile(r'(?:HOMO-LUMO gap|gap) (?:is |of )?([\d\.]+)\s*eV', re.IGNORECASE),
        'unit': 'eV',
        'name': 'HOMO-LUMO gap',
    },
    'qm9_prop_mu': {
        'pattern': re.compile(r'(?:dipole moment|dipole) (?:is |of )?([\d\.]+)\s*debye', re.IGNORECASE),
        'unit': 'debye',
        'name': 'dipole moment',
    },
    'coremof_pore_volume': {
        'pattern': re.compile(r'(?:^|\s)([\d\.]+)\s*cm3/g', re.IGNORECASE),
        'unit': 'cm3/g',
        'name': 'pore volume',
    },
    'coremof_surface_area': {
        'pattern': re.compile(r'(?:^|\s)([\d\.]+)\s*m2/g', re.IGNORECASE),
        'unit': 'm2/g',
        'name': 'surface area',
    },
    'qmof_bandgap': {
        'pattern': re.compile(r'(?:^|\s)([\d\.]+)\s*eV', re.IGNORECASE),
        'unit': 'eV',
        'name': 'bandgap',
    },
    # EDM tasks (6 DFT properties)
    'qm9_edm_prop_alpha': {
        'pattern': re.compile(r'(?:^|\s)(-?[\d\.]+)\s*Bohr\^3', re.IGNORECASE),
        'unit': 'Bohr^3',
        'name': 'polarizability',
    },
    'qm9_edm_prop_gap': {
        'pattern': re.compile(r'(?:^|\s)(-?[\d\.]+)\s*meV', re.IGNORECASE),
        'unit': 'meV',
        'name': 'HOMO-LUMO gap',
    },
    'qm9_edm_prop_homo': {
        'pattern': re.compile(r'(?:^|\s)(-?[\d\.]+)\s*meV', re.IGNORECASE),
        'unit': 'meV',
        'name': 'HOMO energy',
    },
    'qm9_edm_prop_lumo': {
        'pattern': re.compile(r'(?:^|\s)(-?[\d\.]+)\s*meV', re.IGNORECASE),
        'unit': 'meV',
        'name': 'LUMO energy',
    },
    'qm9_edm_prop_mu': {
        'pattern': re.compile(r'(?:^|\s)(-?[\d\.]+)\s*D\b'),
        'unit': 'D',
        'name': 'dipole moment',
    },
    'qm9_edm_prop_Cv': {
        'pattern': re.compile(r'(?:^|\s)(-?[\d\.]+)\s*cal/\(mol\*K\)'),
        'unit': 'cal/(mol*K)',
        'name': 'heat capacity',
    },
}


def extract_property(text, pattern):
    if not isinstance(text, str):
        return None
    match = pattern.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def qm9_prop_metric(ground_truth, prediction, task_name):
    """Placeholder. Metrics are computed from the text in aggregation."""
    return {'score': 0.0, 'valid': True, 'info': {'msg': 'ok'}}


def _r2_score(y_true, y_pred):
    """Coefficient of determination (R²)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return float(1 - ss_res / ss_tot)


def qm9_prop_aggregate(results, task_name, duration):
    """Parse true and predicted values from text and compute MAE, RMSE and R²."""
    prop_config = PROPERTY_PATTERNS[task_name]
    pattern = prop_config['pattern']
    unit = prop_config['unit']
    name = prop_config['name']

    ae_at_1 = []
    ae_at_n = []
    truths_at_1 = []
    preds_at_1 = []
    n_parse_fail_truth = 0
    n_parse_fail_pred = 0
    n_total = 0

    for res in results:
        truth_text = full_text(res['ground_truth'])
        truth_val = extract_property(truth_text, pattern)

        if truth_val is None:
            n_parse_fail_truth += 1
            continue

        n_total += 1
        sample_errors = []
        sample_preds = []

        for sample in res['samples']:
            pred_text = full_text(sample['prediction'])
            pred_val = extract_property(pred_text, pattern) if pred_text else None

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

    ae_at_1_arr = np.array(ae_at_1) if ae_at_1 else np.array([])
    r2_at_1 = _r2_score(truths_at_1, preds_at_1) if truths_at_1 else float('nan')
    rmse_at_1 = float(np.sqrt(np.mean(ae_at_1_arr ** 2))) if ae_at_1 else float('nan')

    metrics = {
        f"mae@1_mean_{unit}": float(np.mean(ae_at_1)) if ae_at_1 else float('nan'),
        f"mae@1_median_{unit}": float(np.median(ae_at_1)) if ae_at_1 else float('nan'),
        f"rmse@1_{unit}": rmse_at_1,
        "r2@1": r2_at_1,
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

    print(f"\n{name} prediction results ({n_total} examples):")
    print(f"  MAE@1:  mean={metrics[f'mae@1_mean_{unit}']:.4f} {unit},  median={metrics[f'mae@1_median_{unit}']:.4f} {unit}")
    print(f"  RMSE@1: {rmse_at_1:.4f} {unit}")
    print(f"  R²@1:   {r2_at_1:.4f}")
    print(f"  MAE@{n_samples}:  mean={metrics[f'mae@{n_samples}_mean_{unit}']:.4f} {unit},  "
          f"median={metrics[f'mae@{n_samples}_median_{unit}']:.4f} {unit}")
    if n_parse_fail_truth:
        print(f"  Warning: {n_parse_fail_truth} truth values could not be parsed")
    if n_parse_fail_pred:
        print(f"  Warning: {n_parse_fail_pred} predictions could not be parsed")

    return metrics
