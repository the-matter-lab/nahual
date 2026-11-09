# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ODAC metrics: periodic RMSD of the adsorbate and of the full system.

The adsorbate is the ground-truth atoms after the prompt's framework atoms.
``periodic_rmsd`` assigns atoms per element, so atom order within an element
need not match.
"""

import copy
from collections import Counter

import numpy as np

from evals.metrics.geometry import all_xyzs, calculate_rmsd_metrics, first_xyz, get_topk_min
from evals.metrics.periodic import periodic_rmsd
from nahual.tokens import XYZ


def _same_composition(a, b):
    return Counter(a.tolist()) == Counter(b.tolist())


def _pick_pred_xyz(prediction, expected_atoms):
    """The first prediction XYZ whose atoms equal expected_atoms, else the largest XYZ.

    Predictions can hold several XYZ blocks, and the first is often an intermediate.
    """
    xyzs = all_xyzs(prediction)
    if not xyzs:
        return None
    for xyz in xyzs:
        if np.array_equal(xyz.atoms, expected_atoms):
            return xyz
    return max(xyzs, key=lambda x: len(x.atoms))


def _with_lattice_tail(xyz, tail_atoms, tail_coords):
    """Periodic XYZ with the lattice header of xyz followed by the given atoms."""
    return XYZ(
        atoms=np.concatenate([xyz.atoms[:4], tail_atoms]),
        coords=np.concatenate([xyz.coords[:4], tail_coords]),
    )


def _odac_metric_single(ground_truth, prediction, prompt=None):
    """ODAC metric against a single reference structure."""
    prompt_xyz = first_xyz(prompt)
    truth_xyz = first_xyz(ground_truth)

    def err(msg):
        return {
            "score": float("inf"),
            "valid": False,
            "info": {
                "ads_rmsd": float("inf"),
                "full_rmsd": float("inf"),
                "msg": msg,
            },
        }

    if prompt_xyz is None or truth_xyz is None:
        return err("Missing prompt or truth XYZ")
    pred_xyz = _pick_pred_xyz(prediction, truth_xyz.atoms)
    if pred_xyz is None:
        return err("No XYZ found")
    if not getattr(prompt_xyz, "has_lattice", False):
        return err("Prompt is non-periodic")
    if not getattr(truth_xyz, "has_lattice", False):
        return err("Truth is non-periodic")
    if not getattr(pred_xyz, "has_lattice", False):
        return err("Prediction is non-periodic")

    ads_len = len(truth_xyz.atoms) - len(prompt_xyz.atoms)
    if ads_len <= 0:
        return err("Truth has no adsorbate tail")

    ads_atoms = truth_xyz.atoms[-ads_len:]
    if not np.array_equal(truth_xyz.atoms[:-ads_len], prompt_xyz.atoms):
        return err("Truth framework does not match prompt")
    if not _same_composition(pred_xyz.atoms, truth_xyz.atoms):
        return err("Composition mismatch")
    if truth_xyz.coords.shape != pred_xyz.coords.shape:
        return err("Shape mismatch")
    if len(pred_xyz.atoms) != len(prompt_xyz.atoms) + ads_len:
        return err("Prediction does not contain the expected adsorbate tail")
    if not _same_composition(pred_xyz.atoms[-ads_len:], ads_atoms):
        return err("Prediction adsorbate tail composition mismatch")

    ads_truth = _with_lattice_tail(truth_xyz, truth_xyz.atoms[-ads_len:], truth_xyz.coords[-ads_len:])
    ads_pred = _with_lattice_tail(pred_xyz, pred_xyz.atoms[-ads_len:], pred_xyz.coords[-ads_len:])

    ads_rmsd = periodic_rmsd(ads_truth, ads_pred)
    full_rmsd = periodic_rmsd(truth_xyz, pred_xyz)
    if np.isnan(ads_rmsd) or np.isnan(full_rmsd):
        return err("RMSD is NaN")

    return {
        "score": ads_rmsd,
        "valid": True,
        "info": {
            "ads_rmsd": ads_rmsd,
            "full_rmsd": full_rmsd,
            "msg": "Success",
        },
    }


def _get_reference_ground_truths(ground_truth):
    """Normalize ODAC ground truth into a list of reference sequences."""
    if isinstance(ground_truth, dict) and "responses" in ground_truth:
        return list(ground_truth["responses"])
    return [ground_truth]


def odac_metric(ground_truth, prediction, prompt=None):
    """Score a prediction against its closest ODAC reference."""
    references = _get_reference_ground_truths(ground_truth)
    if not references:
        return {
            "score": float("inf"),
            "valid": False,
            "info": {
                "ads_rmsd": float("inf"),
                "full_rmsd": float("inf"),
                "msg": "Missing ODAC references",
            },
        }

    best_metric = None
    best_idx = None
    for ref_idx, ref in enumerate(references):
        metric = _odac_metric_single(ref, prediction, prompt)
        if best_metric is None or metric["score"] < best_metric["score"]:
            best_metric = metric
            best_idx = ref_idx

    result = copy.deepcopy(best_metric)
    result["info"]["matched_reference_index"] = best_idx
    result["info"]["num_references"] = len(references)
    return result


def odac_aggregate(results, k_values, duration):
    """Top-k adsorbate and full-system RMSD."""
    topk_ads = {k: [] for k in k_values}
    topk_full = {k: [] for k in k_values}
    ref_counts = []

    for res in results:
        samples = res["samples"]
        ads_scores = [s["metric"]["info"]["ads_rmsd"] for s in samples]
        full_scores = [s["metric"]["info"]["full_rmsd"] for s in samples]
        ref_counts.append(len(_get_reference_ground_truths(res.get("ground_truth"))))

        for k in k_values:
            topk_ads[k].append(get_topk_min(ads_scores, k))
            topk_full[k].append(get_topk_min(full_scores, k))

    metrics = {}
    for k in k_values:
        ads_mean, ads_median, ads_fail = calculate_rmsd_metrics(topk_ads[k])
        full_mean, full_median, full_fail = calculate_rmsd_metrics(topk_full[k])
        metrics[f"top{k}_ads_rmsd_mean"] = ads_mean
        metrics[f"top{k}_ads_rmsd_median"] = ads_median
        metrics[f"top{k}_ads_fail_rate"] = ads_fail
        metrics[f"top{k}_full_rmsd_mean"] = full_mean
        metrics[f"top{k}_full_rmsd_median"] = full_median
        metrics[f"top{k}_full_fail_rate"] = full_fail

    if ref_counts:
        ref_counts_arr = np.asarray(ref_counts, dtype=np.float64)
        metrics["mean_references_per_prompt"] = float(ref_counts_arr.mean())
        metrics["median_references_per_prompt"] = float(np.median(ref_counts_arr))
        metrics["multi_reference_prompt_rate"] = float(np.mean(ref_counts_arr > 1))

    metrics["duration_seconds"] = duration
    return metrics
