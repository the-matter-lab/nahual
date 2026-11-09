# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

import numpy as np

from evals.api import Task
from evals.metrics.geometry import all_xyzs, calculate_rmsd_metrics, get_topk_min
from evals.metrics.periodic import periodic_rmsd

# Å. Atoms that move more than this between the truth initial and final frames are adsorbate.
ADS_DISPLACEMENT_THRESHOLD = 0.5  # Å — atoms moving more than this between truth


def _adsorbate_mask(t_start, t_end, threshold=ADS_DISPLACEMENT_THRESHOLD):
    """Find adsorbate atoms by MIC displacement between the truth initial and final frames.

    Returns (mask, contiguous_at_end). The mask excludes the 4-row lattice header.
    contiguous_at_end is True if the moving atoms are the last atoms, as in OC20.
    """
    from ase.geometry import find_mic

    has_lattice = getattr(t_start, 'has_lattice', False)
    if has_lattice:
        origin_s = t_start.coords[0]
        origin_e = t_end.coords[0]
        cell = t_start.coords[1:4] - origin_s
        pbc = t_start.pbc
        coords_s = t_start.coords[4:] - origin_s
        coords_e = t_end.coords[4:] - origin_e
        _, dist = find_mic(coords_e - coords_s, cell, pbc)
    else:
        dist = np.linalg.norm(t_end.coords - t_start.coords, axis=-1)

    mask = dist > threshold
    n_moving = int(mask.sum())
    if n_moving == 0:
        return mask, False
    contiguous = bool(mask[-n_moving:].all())
    return mask, contiguous


def _slice_atoms(xyz, mask, has_lattice):
    """Copy of xyz with only the atoms in `mask`, keeping the lattice header."""
    out = deepcopy(xyz)
    if has_lattice:
        out.atoms = np.concatenate([xyz.atoms[:4], xyz.atoms[4:][mask]])
        out.coords = np.concatenate([xyz.coords[:4], xyz.coords[4:][mask]])
    else:
        out.atoms = xyz.atoms[mask]
        out.coords = xyz.coords[mask]
    return out


class NebTask(Task):
    def __init__(self, config):
        self.config = config

    def setup(self, args=None):
        pass

    def metric(self, ground_truth, prediction, prompt=None):
        """Mean periodic RMSD over NEB frames, plus linear-interpolation baselines.

        RMSD is reported for the full system and for the adsorbate atoms only.
        The baselines interpolate linearly between the truth initial and final
        frames, in Cartesian space (`base_*_rmsd`) or along the minimum-image
        displacement (`base_mic_*_rmsd`). They coincide for non-periodic systems.
        """
        from ase.geometry import find_mic

        truth_frames = all_xyzs(ground_truth)
        pred_frames = all_xyzs(prediction)

        def err(msg):
            return {'score': float('inf'), 'valid': False,
                    'info': {'pred_rmsd': float('inf'), 'base_rmsd': float('inf'),
                             'base_mic_rmsd': float('inf'),
                             'pred_ads_rmsd': float('inf'), 'base_ads_rmsd': float('inf'),
                             'base_mic_ads_rmsd': float('inf'),
                             'msg': msg}}

        if not pred_frames:
            return err("No frames found")

        n_frames = len(truth_frames)
        if len(pred_frames) != n_frames:
            return err(f"Frame count mismatch: {len(pred_frames)} vs {n_frames}")
        if n_frames == 0:
            return err("Zero frames")

        t_start, t_end = truth_frames[0], truth_frames[-1]
        has_lattice = getattr(t_start, 'has_lattice', False)
        ads_mask, ads_contiguous = _adsorbate_mask(t_start, t_end)
        n_ads_atoms = int(ads_mask.sum())
        ads_available = n_ads_atoms > 0

        # Minimum-image displacement from initial to final frame, or the plain difference without a lattice.
        if has_lattice:
            origin_s = t_start.coords[0]
            origin_e = t_end.coords[0]
            cell = t_start.coords[1:4] - origin_s
            pbc = t_start.pbc
            coords_s_rel = t_start.coords[4:] - origin_s
            coords_e_rel = t_end.coords[4:] - origin_e
            mic_delta, _ = find_mic(coords_e_rel - coords_s_rel, cell, pbc)
        else:
            mic_delta = t_end.coords - t_start.coords

        rmsds_pred, rmsds_base, rmsds_base_mic = [], [], []
        rmsds_pred_ads, rmsds_base_ads, rmsds_base_mic_ads = [], [], []

        for idx, (p, t) in enumerate(zip(pred_frames, truth_frames)):
            if not np.array_equal(p.atoms, t.atoms):
                return err(f"Atom mismatch at frame {idx}")
            if p.coords.shape != t.coords.shape:
                return err(f"Shape mismatch at frame {idx}")

            rmsd_p = periodic_rmsd(p, t)
            if np.isnan(rmsd_p):
                return err(f"NaN RMSD at frame {idx}")
            rmsds_pred.append(rmsd_p)

            alpha = idx / (n_frames - 1) if n_frames > 1 else 0.0

            base_xyz = deepcopy(t_start)
            base_xyz.coords = (1 - alpha) * t_start.coords + alpha * t_end.coords
            rmsds_base.append(periodic_rmsd(base_xyz, t))

            base_mic_xyz = deepcopy(t_start)
            if has_lattice:
                interp_atoms = coords_s_rel + alpha * mic_delta + origin_s
                base_mic_xyz.coords = np.concatenate(
                    [t_start.coords[:4], interp_atoms], axis=0
                )
            else:
                base_mic_xyz.coords = t_start.coords + alpha * mic_delta
            rmsds_base_mic.append(periodic_rmsd(base_mic_xyz, t))

            if ads_available:
                p_ads = _slice_atoms(p, ads_mask, has_lattice)
                t_ads = _slice_atoms(t, ads_mask, has_lattice)
                base_ads = _slice_atoms(base_xyz, ads_mask, has_lattice)
                base_mic_ads = _slice_atoms(base_mic_xyz, ads_mask, has_lattice)
                rmsd_p_ads = periodic_rmsd(p_ads, t_ads)
                if np.isnan(rmsd_p_ads):
                    return err(f"NaN ads RMSD at frame {idx}")
                rmsds_pred_ads.append(rmsd_p_ads)
                rmsds_base_ads.append(periodic_rmsd(base_ads, t_ads))
                rmsds_base_mic_ads.append(periodic_rmsd(base_mic_ads, t_ads))

        pred_mean = float(np.mean(rmsds_pred))
        base_mean = float(np.mean(rmsds_base))
        base_mic_mean = float(np.mean(rmsds_base_mic))
        if ads_available:
            pred_ads_mean = float(np.mean(rmsds_pred_ads))
            base_ads_mean = float(np.mean(rmsds_base_ads))
            base_mic_ads_mean = float(np.mean(rmsds_base_mic_ads))
        else:
            pred_ads_mean = float('inf')
            base_ads_mean = float('inf')
            base_mic_ads_mean = float('inf')

        return {'score': pred_mean, 'valid': True,
                'info': {'pred_rmsd': pred_mean,
                         'base_rmsd': base_mean, 'base_mic_rmsd': base_mic_mean,
                         'pred_ads_rmsd': pred_ads_mean,
                         'base_ads_rmsd': base_ads_mean, 'base_mic_ads_rmsd': base_mic_ads_mean,
                         'n_ads_atoms': n_ads_atoms, 'ads_contiguous_at_end': ads_contiguous,
                         'msg': 'Success'}}

    def _aggregate_subset(self, results, k_values):
        """Metrics for a subset of prompt results."""
        topk_pred = {k: [] for k in k_values}
        topk_pred_ads = {k: [] for k in k_values}
        base_rmsds, base_ads_rmsds = [], []
        base_mic_rmsds, base_mic_ads_rmsds = [], []
        n_with_ads, n_contiguous = 0, 0

        def _first_valid(samples, key):
            vals = [s['metric']['info'].get(key, float('inf')) for s in samples]
            vals = [v for v in vals if v != float('inf')]
            return vals[0] if vals else float('inf')

        for res in results:
            samples = res['samples']
            pred_scores = [s['metric']['score'] for s in samples]
            pred_ads_scores = [s['metric']['info'].get('pred_ads_rmsd', float('inf'))
                               for s in samples]

            for k in k_values:
                topk_pred[k].append(get_topk_min(pred_scores, k))
                topk_pred_ads[k].append(get_topk_min(pred_ads_scores, k))

            base_rmsds.append(_first_valid(samples, 'base_rmsd'))
            base_mic_rmsds.append(_first_valid(samples, 'base_mic_rmsd'))
            base_ads_rmsds.append(_first_valid(samples, 'base_ads_rmsd'))
            base_mic_ads_rmsds.append(_first_valid(samples, 'base_mic_ads_rmsd'))

            for s in samples:
                info = s['metric']['info']
                if info.get('n_ads_atoms', 0) > 0:
                    n_with_ads += 1
                    if info.get('ads_contiguous_at_end', False):
                        n_contiguous += 1

        out = {}
        for k in k_values:
            mean_val, median_val, fail_rate = calculate_rmsd_metrics(topk_pred[k])
            out[f"top{k}_pred_rmsd_mean"] = mean_val
            out[f"top{k}_pred_rmsd_median"] = median_val
            out[f"top{k}_fail_rate"] = fail_rate

            ads_mean, ads_median, ads_fail = calculate_rmsd_metrics(topk_pred_ads[k])
            out[f"top{k}_pred_ads_rmsd_mean"] = ads_mean
            out[f"top{k}_pred_ads_rmsd_median"] = ads_median
            out[f"top{k}_ads_fail_rate"] = ads_fail

        for prefix, lst in [("base", base_rmsds), ("base_mic", base_mic_rmsds),
                            ("base_ads", base_ads_rmsds), ("base_mic_ads", base_mic_ads_rmsds)]:
            m, med, _ = calculate_rmsd_metrics(lst)
            out[f"{prefix}_rmsd_mean"] = m
            out[f"{prefix}_rmsd_median"] = med

        if n_with_ads > 0:
            out["ads_contiguous_at_end_frac"] = n_contiguous / n_with_ads
        out["n_prompts"] = len(results)
        return out

    def aggregate(self, results, duration):
        k_values = self.config.k_values
        metrics = self._aggregate_subset(results, k_values)

        # Per-reaction-type metrics. Prompts 0-9 of oc20neb_combined_test_30 are
        # dissociations, 10-19 desorptions and 20-29 transfers.
        if self.config.dataset == "oc20neb_combined_test_30":
            split_ranges = {
                "dissociations": (0, 10),
                "desorptions":   (10, 20),
                "transfers":     (20, 30),
            }
            for split_name, (lo, hi) in split_ranges.items():
                subset = [r for r in results if lo <= r['prompt_index'] < hi]
                if not subset:
                    continue
                for key, val in self._aggregate_subset(subset, k_values).items():
                    metrics[f"{split_name}_{key}"] = val

        metrics["duration_seconds"] = duration
        return metrics
