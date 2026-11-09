# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QM9 EDM conditional generation tasks qm9_edm_cond_{alpha,gap,homo,lumo,mu,Cv}.

Evaluates on first_half, which is held out from training. A pretrained EGNN
predicts the property of each generated molecule, and the task reports the MAE
between that prediction and the target.
"""

import json
import re

import numpy as np

from evals.metrics.geometry import first_xyz, rmsd_metric
from evals.api import Task

try:
    from nahual.metrics import check_validity_smiles
    HAS_VALIDITY = True
except ImportError:
    HAS_VALIDITY = False


# Property configs: (display_name, unit, prompt_regex)
EDM_COND_CONFIGS = {
    "alpha": ("polarizability",  "Bohr^3",      re.compile(r'polarizability of (-?[\d\.]+)\s*Bohr\^3', re.IGNORECASE)),
    "gap":   ("HOMO-LUMO gap",   "meV",          re.compile(r'HOMO-LUMO gap of (-?[\d\.]+)\s*meV', re.IGNORECASE)),
    "homo":  ("HOMO energy",     "meV",          re.compile(r'HOMO energy of (-?[\d\.]+)\s*meV', re.IGNORECASE)),
    "lumo":  ("LUMO energy",     "meV",          re.compile(r'LUMO energy of (-?[\d\.]+)\s*meV', re.IGNORECASE)),
    "mu":    ("dipole moment",   "D",            re.compile(r'dipole moment of (-?[\d\.]+)\s*D\b')),
    "Cv":    ("heat capacity",   "cal/(mol*K)",  re.compile(r'heat capacity of (-?[\d\.]+)\s*cal/\(mol\*K\)')),
}


def _instruction_to_text(instruction):
    """Extract text from HF dataset instruction field."""
    if isinstance(instruction, str):
        instruction = json.loads(instruction)
    parts = []
    for seg in instruction:
        if isinstance(seg, dict) and seg.get("type") == "text" and seg.get("text"):
            parts.append(seg["text"])
    return " ".join(parts)


def _check_valid(xyz):
    if not HAS_VALIDITY or xyz is None:
        return None
    try:
        valid, _ = check_validity_smiles(xyz)
        return valid
    except Exception:
        return None


def _mean_or_nan(vals):
    return float(np.mean(vals)) if vals else float("nan")


def _median_or_nan(vals):
    return float(np.median(vals)) if vals else float("nan")


def _make_cond_task_class(prop_key):
    display_name, unit, pattern = EDM_COND_CONFIGS[prop_key]

    class EdmCondTask(Task):
        def __init__(self, config):
            self.config = config
            self._egnn_model = None
            self._egnn_mean = None
            self._egnn_mad = None

        def setup(self, args=None):
            pass

        def metric(self, ground_truth, prediction, prompt=None):
            return rmsd_metric(ground_truth, prediction)

        def post_process(self, results, dataset, save_path):
            """Parse each prompt's target value and store each sample's EGNN prediction."""
            import torch
            from evals.tasks.conditional_molecules.predictor import load_egnn_predictor, score_molecules

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading EGNN predictor for {prop_key} on {device}...")
            model, mean, mad = load_egnn_predictor(prop_key, device=device)

            samples = []
            for res in results:
                # Parse target value from prompt
                row = dataset[int(res["prompt_index"])]
                prompt_text = _instruction_to_text(row["instruction"])
                m = pattern.search(prompt_text)
                res["target_value"] = float(m.group(1)) if m else None

                samples.extend(res['samples'])
            values = score_molecules(
                model, [first_xyz(s['prediction']) for s in samples], mean, mad,
                prop_key, device=device, batch_size=self.config.extra.get('egnn_batch_size', 128))
            for sample, value in zip(samples, values):
                sample['egnn_pred'] = value

        def aggregate(self, results, duration):
            n_total = len(results)
            n_valid_xyz = 0
            n_rdkit_valid = 0
            n_rdkit_checked = 0
            abs_errors = []
            best_abs_errors = []
            n_parse_fail_target = 0
            n_egnn_fail = 0

            for res in results:
                target_val = res.get("target_value")
                if target_val is None:
                    n_parse_fail_target += 1
                    continue

                for sample in res["samples"][:1]:
                    pred_xyz = first_xyz(sample["prediction"])
                    if pred_xyz is None:
                        continue

                    n_valid_xyz += 1

                    valid = _check_valid(pred_xyz)
                    if valid is not None:
                        n_rdkit_checked += 1
                        if valid:
                            n_rdkit_valid += 1

                    egnn_pred = sample.get("egnn_pred")
                    if egnn_pred is not None:
                        abs_errors.append(abs(egnn_pred - target_val))
                    else:
                        n_egnn_fail += 1

                sample_errors = [
                    abs(s["egnn_pred"] - target_val)
                    for s in res["samples"]
                    if s.get("egnn_pred") is not None
                ]
                if sample_errors:
                    best_abs_errors.append(min(sample_errors))

            metrics = {
                "n_total": n_total,
                "n_valid_xyz": n_valid_xyz,
                "valid_xyz_rate": n_valid_xyz / n_total if n_total else 0.0,
                "rdkit_validity_rate": (n_rdkit_valid / n_rdkit_checked) if n_rdkit_checked > 0 else float("nan"),
                f"mae_{unit}": _mean_or_nan(abs_errors),
                f"medae_{unit}": _median_or_nan(abs_errors),
                f"mae_best5_{unit}": _mean_or_nan(best_abs_errors),
                f"medae_best5_{unit}": _median_or_nan(best_abs_errors),
                "n_evaluated": len(abs_errors),
                "n_evaluated_best5": len(best_abs_errors),
                "n_parse_fail_target": n_parse_fail_target,
                "n_egnn_fail": n_egnn_fail,
                "duration_seconds": duration,
            }

            print(f"\n{display_name} conditional generation results ({n_total} examples):")
            print(f"  Valid XYZ rate: {metrics['valid_xyz_rate']:.2%}")
            if n_rdkit_checked > 0:
                print(f"  RDKit validity: {metrics['rdkit_validity_rate']:.2%}")
            print(f"  EGNN evaluated: {len(abs_errors)}")
            print(f"  MAE:  mean={metrics[f'mae_{unit}']:.4f} {unit},  median={metrics[f'medae_{unit}']:.4f} {unit}")
            print(f"  best-of-5 MAE:  mean={metrics[f'mae_best5_{unit}']:.4f} {unit},  median={metrics[f'medae_best5_{unit}']:.4f} {unit}  (n={metrics['n_evaluated_best5']})")
            if n_parse_fail_target:
                print(f"  Warning: {n_parse_fail_target} target values could not be parsed")
            if n_egnn_fail:
                print(f"  Warning: {n_egnn_fail} EGNN predictions failed")

            return metrics

    EdmCondTask.__name__ = f"QM9EdmCond{prop_key.capitalize()}Task"
    return EdmCondTask


# --- Register all 6 tasks ---


def create_task(config):
    return _make_cond_task_class(config.extra["property"])(config)
