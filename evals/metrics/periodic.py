# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared metric logic for GMAE tasks (periodic RMSD with adsorbate decomposition)."""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from evals.metrics.geometry import first_xyz, calculate_rmsd_metrics, get_topk_min
from nahual.tokens import XYZ


def _mic_distance_matrix(coords1, coords2, cell, pbc):
    """(N, M) minimum-image distances between (N, 3) and (M, 3) coordinates.

    cell is the (3, 3) lattice and pbc the (3,) periodic flags.
    """
    from ase.geometry import find_mic

    N = len(coords1)
    M = len(coords2)
    # Broadcast: diff[i,j] = coords2[j] - coords1[i]
    diff = coords2[None, :, :] - coords1[:, None, :]  # (N, M, 3)
    diff_flat = diff.reshape(-1, 3)
    _, dist_flat = find_mic(diff_flat, cell, pbc)
    return dist_flat.reshape(N, M)


def _hungarian_reorder(atoms1, coords1, atoms2, coords2, cell=None, pbc=None):
    """Reorder coords2 to best match coords1 by Hungarian matching within each element.

    Uses MIC distances if cell is given. Returns (reordered coords2, success), where
    success is False if the element counts differ.
    """
    if Counter(atoms1.tolist()) != Counter(atoms2.tolist()):
        return coords2, False

    n = len(atoms1)
    reordered = np.zeros_like(coords2)
    used_indices = set()

    for elem in np.unique(atoms1):
        idx1 = np.where(atoms1 == elem)[0]
        idx2 = np.where(atoms2 == elem)[0]

        if len(idx1) != len(idx2):
            return coords2, False

        if len(idx1) == 1:
            reordered[idx1[0]] = coords2[idx2[0]]
            used_indices.add(idx2[0])
            continue

        c1 = coords1[idx1]
        c2 = coords2[idx2]

        if cell is not None:
            cost = _mic_distance_matrix(c1, c2, cell, pbc)
        else:
            diff = c2[None, :, :] - c1[:, None, :]
            cost = np.sqrt(np.sum(diff**2, axis=-1))

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            reordered[idx1[r]] = coords2[idx2[c]]
            used_indices.add(idx2[c])

    return reordered, True


def periodic_rmsd(xyz1, xyz2):
    """Compute RMSD between two periodic XYZ objects.

    Handles atom reordering via Hungarian matching with MIC distances.
    """
    from ase.geometry import find_mic

    if not getattr(xyz1, 'has_lattice', False):
        # Non-periodic: Hungarian match with Cartesian distances
        n1, n2 = len(xyz1.coords), len(xyz2.coords)
        if n1 == 0 or n2 == 0:
            return float('inf')
        if Counter(xyz1.atoms.tolist()) != Counter(xyz2.atoms.tolist()):
            return float('inf')

        reordered, ok = _hungarian_reorder(xyz1.atoms, xyz1.coords, xyz2.atoms, xyz2.coords)
        if not ok:
            return float('inf')
        diff = reordered - xyz1.coords
        return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))

    # Periodic case: first 4 positions are origin + lattice vectors
    n = min(len(xyz1.coords), len(xyz2.coords))
    if n < 5:
        return float('inf')

    origin1 = xyz1.coords[0]
    origin2 = xyz2.coords[0]
    cell = xyz1.coords[1:4] - origin1
    pbc = xyz1.pbc

    # Atom coordinates (skip origin + 3 lattice vectors)
    atoms1 = xyz1.atoms[4:n]
    atoms2 = xyz2.atoms[4:n]
    coords1 = xyz1.coords[4:n] - origin1
    coords2 = xyz2.coords[4:n] - origin2

    if Counter(atoms1.tolist()) != Counter(atoms2.tolist()):
        return float('inf')

    # Hungarian reorder with MIC distances
    reordered, ok = _hungarian_reorder(atoms1, coords1, atoms2, coords2, cell=cell, pbc=pbc)
    if not ok:
        return float('inf')

    diff = reordered - coords1
    _, diff_len = find_mic(diff, cell, pbc)
    return float(np.sqrt(np.mean(diff_len**2)))


def _same_composition(atoms1, atoms2):
    """Check if two atom arrays have the same element counts."""
    return Counter(atoms1.tolist()) == Counter(atoms2.tolist())


def gmae_metric(ground_truth, prediction, prompt=None):
    """GMAE metric: periodic RMSD for adsorbate and full system."""
    truth_xyz = first_xyz(ground_truth)
    pred_xyz = first_xyz(prediction)

    def err(msg):
        return {'score': float('inf'), 'valid': False,
                'info': {'ads_rmsd': float('inf'), 'full_rmsd': float('inf'), 'msg': msg}}

    if pred_xyz is None:
        return err("No XYZ found")
    if not _same_composition(truth_xyz.atoms, pred_xyz.atoms):
        return err("Composition mismatch")
    if truth_xyz.coords.shape != pred_xyz.coords.shape:
        return err("Shape mismatch")

    # The response is the prompt's periodic surface followed by the adsorbate.
    prompt_xyzs = [x for x in prompt if isinstance(x, XYZ)] if prompt else []
    surface = next((x for x in prompt_xyzs if getattr(x, 'has_lattice', False)), None)
    if surface is None:
        return err("No periodic surface in prompt")
    num_ads_atoms = len(truth_xyz.atoms) - len(surface.atoms)
    if num_ads_atoms <= 0:
        return err("No adsorbate atoms")

    if getattr(truth_xyz, 'has_lattice', False):
        ads_truth_atoms = np.concatenate([truth_xyz.atoms[:4], truth_xyz.atoms[-num_ads_atoms:]])
        ads_truth_coords = np.concatenate([truth_xyz.coords[:4], truth_xyz.coords[-num_ads_atoms:]])
        ads_pred_atoms = np.concatenate([pred_xyz.atoms[:4], pred_xyz.atoms[-num_ads_atoms:]])
        ads_pred_coords = np.concatenate([pred_xyz.coords[:4], pred_xyz.coords[-num_ads_atoms:]])
        ads_truth = XYZ(atoms=ads_truth_atoms, coords=ads_truth_coords)
        ads_pred = XYZ(atoms=ads_pred_atoms, coords=ads_pred_coords)
    else:
        ads_truth = XYZ(atoms=truth_xyz.atoms[-num_ads_atoms:], coords=truth_xyz.coords[-num_ads_atoms:])
        ads_pred = XYZ(atoms=pred_xyz.atoms[-num_ads_atoms:], coords=pred_xyz.coords[-num_ads_atoms:])

    ads_rmsd = periodic_rmsd(ads_truth, ads_pred)
    full_rmsd = periodic_rmsd(truth_xyz, pred_xyz)

    if np.isnan(ads_rmsd) or np.isnan(full_rmsd):
        return err("RMSD is NaN")

    return {'score': ads_rmsd, 'valid': True,
            'info': {'ads_rmsd': ads_rmsd, 'full_rmsd': full_rmsd, 'msg': 'Success'}}


def gmae_aggregate(results, k_values, duration):
    """GMAE aggregation: top-k for both adsorbate and full system RMSD."""
    topk_ads = {k: [] for k in k_values}
    topk_full = {k: [] for k in k_values}

    for res in results:
        samples = res['samples']
        ads_scores = [s['metric']['info']['ads_rmsd'] for s in samples]
        full_scores = [s['metric']['info']['full_rmsd'] for s in samples]

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

    metrics["duration_seconds"] = duration
    return metrics
