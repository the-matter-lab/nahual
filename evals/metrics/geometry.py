# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Geometry metrics and top-k aggregation helpers."""

import numpy as np

from nahual.tokens import XYZ
from nahual.compare import compare_xyz
from nahual.metrics import get_bond_graph, build_graph, are_isomorphic


# --- Sequence extraction helpers ---

def first_xyz(seq):
    """Extract the first XYZ object from a parsed sequence."""
    if seq is None:
        return None
    for item in seq:
        if isinstance(item, XYZ):
            return item
    return None


def last_xyz(seq):
    """Extract the last XYZ object from a parsed sequence (e.g. CoT final answer)."""
    if seq is None:
        return None
    for item in reversed(seq):
        if isinstance(item, XYZ):
            return item
    return None


def all_xyzs(seq):
    """Extract all XYZ objects from a parsed sequence."""
    if seq is None:
        return []
    return [item for item in seq if isinstance(item, XYZ)]


def full_text(seq):
    """Concatenate all text segments from a parsed sequence."""
    if seq is None:
        return ""
    return " ".join(item for item in seq if isinstance(item, str))


def nth_xyz(seq, n):
    """Extract the nth XYZ object (0-indexed) from a parsed sequence."""
    xyzs = all_xyzs(seq)
    return xyzs[n] if n < len(xyzs) else None


# --- Standard metric functions ---

def rmsd_metric(ground_truth, prediction):
    """RMSD between the first XYZ of the ground truth and of the prediction.

    Both are parsed segment lists. Returns a dict with 'score', 'valid' and 'info'.
    """
    truth_xyz = first_xyz(ground_truth)
    pred_xyz = first_xyz(prediction)

    if pred_xyz is None:
        return {'score': float('inf'), 'valid': False, 'info': {'msg': 'No XYZ found'}}

    if truth_xyz is None:
        return {'score': float('inf'), 'valid': False, 'info': {'msg': 'No truth XYZ'}}

    if not np.array_equal(truth_xyz.atoms, pred_xyz.atoms):
        return {'score': float('inf'), 'valid': False, 'info': {'msg': 'Atom mismatch'}}

    if truth_xyz.coords.shape != pred_xyz.coords.shape:
        return {'score': float('inf'), 'valid': False, 'info': {'msg': 'Shape mismatch'}}

    rmsd, _ = compare_xyz(truth_xyz, pred_xyz, chirality_matters=False)

    if np.isnan(rmsd):
        return {'score': float('inf'), 'valid': False, 'info': {'msg': 'RMSD is NaN'}}

    return {'score': float(rmsd), 'valid': True, 'info': {}}


def connectivity_match(ground_truth, prediction):
    """Check whether first XYZs share the same atom-labeled bond graph."""
    truth_xyz = first_xyz(ground_truth)
    pred_xyz = first_xyz(prediction)

    if pred_xyz is None:
        return {'match': False, 'valid': False, 'info': {'msg': 'No XYZ found'}}

    if truth_xyz is None:
        return {'match': False, 'valid': False, 'info': {'msg': 'No truth XYZ'}}

    try:
        truth_graph = build_graph(
            get_bond_graph(truth_xyz.atoms, truth_xyz.coords),
            truth_xyz.atoms,
        )
        pred_graph = build_graph(
            get_bond_graph(pred_xyz.atoms, pred_xyz.coords),
            pred_xyz.atoms,
        )
    except Exception as exc:
        return {
            'match': False,
            'valid': False,
            'info': {'msg': f'Connectivity check failed: {exc}'},
        }

    return {'match': bool(are_isomorphic(truth_graph, pred_graph)), 'valid': True, 'info': {}}


# --- Aggregation helpers ---

def calculate_rmsd_metrics(rmsd_list):
    """Return (mean, median, failure rate) of an RMSD list.

    The mean excludes inf, the median includes it, and the failure rate is the fraction of inf.
    """
    rmsd_list = np.array(rmsd_list)
    valid_mask = ~np.isinf(rmsd_list)
    valid_rmsds = rmsd_list[valid_mask]

    if len(valid_rmsds) == 0:
        return float('nan'), float('inf'), 1.0

    mean_val = float(np.mean(valid_rmsds))
    median_val = float(np.median(rmsd_list))
    fail_rate = 1.0 - (len(valid_rmsds) / len(rmsd_list))

    return mean_val, median_val, fail_rate


def get_topk_min(scores, k):
    """Minimum finite score among the first k samples, or inf."""
    if not scores:
        return float('inf')
    k_scores = [s for s in scores[:k] if s != float('inf')]
    return min(k_scores) if k_scores else float('inf')


def topk_rmsd_aggregation(results, k_values, duration):
    """Mean, median and failure rate of the best RMSD among the first k samples, per k."""
    topk_rmsds = {k: [] for k in k_values}

    for res in results:
        sample_scores = [s['metric']['score'] for s in res['samples']]
        for k in k_values:
            topk_rmsds[k].append(get_topk_min(sample_scores, k))

    metrics = {}
    for k in k_values:
        mean_val, median_val, fail_rate = calculate_rmsd_metrics(topk_rmsds[k])
        metrics[f"top{k}_rmsd_mean"] = mean_val
        metrics[f"top{k}_rmsd_median"] = median_val
        metrics[f"top{k}_fail_rate"] = fail_rate

    metrics["duration_seconds"] = duration
    return metrics


def topk_connectivity_match_aggregation(results, k_values):
    """Compute top-k graph-connectivity match rate from per-sample metric info."""
    metrics = {}

    for k in k_values:
        topk_matches = []
        for res in results:
            samples_k = res['samples'][:min(k, len(res['samples']))]
            topk_matches.append(
                any(s.get('metric', {}).get('info', {}).get('same_connectivity', False)
                    for s in samples_k)
            )
        metrics[f"top{k}_connectivity_match_rate"] = float(np.mean(topk_matches)) if topk_matches else 0.0

    return metrics
