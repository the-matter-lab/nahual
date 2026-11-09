# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QCG microsolvation task with 1 to 10 waters.

Each prompt in `qcg_solv_1to10water_xtbcrest_dynamic_apr2026_test` asks for a
target water count N, parsed from the prompt text. Reference formation
energies are in `data/gt_eform_1to10_dynamic_apr2026_test.csv`, keyed by
dataset row index.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd

from evals.api import Task
from evals.metrics.geometry import first_xyz, rmsd_metric, topk_rmsd_aggregation
from evals.tasks.microsolvation.xtb import (
    write_xyz,
    generate_xtb_scripts,
    parse_xtb_energy,
    calculate_eform,
)
from nahual.metrics import (
    extract_solute,
    count_waters,
    check_overlaps,
    get_bond_graph,
    build_graph,
    are_isomorphic,
)
from nahual.tokens import make_seq


_WATER_COUNT_RE = re.compile(
    r"solvated by\s+(\d+)\s+molecule[s]?\s+of\s+water", re.IGNORECASE
)


def _parse_target_n_water(instruction) -> int:
    """Extract N from the prompt text, e.g. "solvated by 7 molecules of water"."""
    if isinstance(instruction, str):
        instruction = json.loads(instruction)
    for seg in instruction:
        text = seg.get("text") if isinstance(seg, dict) else None
        if not text:
            continue
        m = _WATER_COUNT_RE.search(text)
        if m:
            return int(m.group(1))
    raise ValueError("could not parse target water count from instruction text")


def _solvation_check(pred_xyz, truth_solute, target_n_water):
    atoms = pred_xyz.atoms
    coords = pred_xyz.coords

    if check_overlaps(coords):
        return {
            "success": False,
            "reason": "Overlap",
            "solute_atoms": None,
            "solute_coords": None,
            "num_waters": 0,
            "target_n_water": target_n_water,
        }

    solute_atoms, solute_coords = extract_solute(atoms, coords)

    G_truth = build_graph(
        get_bond_graph(truth_solute.atoms, truth_solute.coords), truth_solute.atoms
    )
    G_pred = build_graph(get_bond_graph(solute_atoms, solute_coords), solute_atoms)

    if not are_isomorphic(G_truth, G_pred):
        return {
            "success": False,
            "reason": "Invalid Solute",
            "solute_atoms": solute_atoms,
            "solute_coords": solute_coords,
            "num_waters": count_waters(atoms, coords),
            "target_n_water": target_n_water,
        }

    num_waters = count_waters(atoms, coords)
    if num_waters != target_n_water:
        return {
            "success": False,
            "reason": f"Incorrect Water Count ({num_waters}, expected {target_n_water})",
            "solute_atoms": solute_atoms,
            "solute_coords": solute_coords,
            "num_waters": num_waters,
            "target_n_water": target_n_water,
        }

    return {
        "success": True,
        "reason": "Success",
        "solute_atoms": solute_atoms,
        "solute_coords": solute_coords,
        "num_waters": num_waters,
        "target_n_water": target_n_water,
    }


class Qcg1to10Task(Task):
    def __init__(self, config):
        self.config = config
        self._gt_eforms = {}
        self._gt_n_water = {}

    def setup(self, args=None):
        task_root = str(Path(__file__).resolve().parent)
        gt_path = os.path.join(
            task_root, "data", "gt_eform_1to10_dynamic_apr2026_test.csv"
        )
        if os.path.exists(gt_path):
            df = pd.read_csv(gt_path)
            df = df.dropna(subset=["e_form"])
            self._gt_eforms = df.set_index("dataset_row_idx")["e_form"].to_dict()
            self._gt_n_water = df.set_index("dataset_row_idx")["target_n_water"].to_dict()
            print(f"Loaded {len(self._gt_eforms)} ground-truth eforms from {gt_path}.")
        else:
            print(f"Warning: ground-truth CSV not found at {gt_path}.")

    def metric(self, ground_truth, prediction, prompt=None):
        return rmsd_metric(ground_truth, prediction)

    def post_process(self, results, dataset, save_path):
        """Check solvation, then compute xTB single-point formation energies."""
        xtb_dir = os.path.join(save_path, "xtb_files")
        os.makedirs(xtb_dir, exist_ok=True)
        xyz_dirs = []

        print("Post-processing: solvation validation...")
        for res in results:
            prompt_idx = res["prompt_index"]
            row = dataset[prompt_idx]
            instruction = row["instruction"]
            target_n = self._gt_n_water.get(prompt_idx)
            if target_n is None:
                target_n = _parse_target_n_water(instruction)
            if isinstance(instruction, str):
                instruction = json.loads(instruction)
            truth_solute = make_seq(instruction)[1]

            for s_idx, sample in enumerate(res["samples"]):
                pred_xyz = first_xyz(sample["prediction"])
                if pred_xyz is None:
                    sample["solv"] = {
                        "success": False,
                        "reason": "No XYZ",
                        "num_waters": 0,
                        "target_n_water": target_n,
                    }
                    continue

                solv_res = _solvation_check(pred_xyz, truth_solute, target_n)
                sample["solv"] = solv_res

                if solv_res["success"]:
                    sample_xtb_dir = os.path.join(xtb_dir, f"p{prompt_idx}_s{s_idx}")
                    os.makedirs(sample_xtb_dir, exist_ok=True)
                    write_xyz(
                        pred_xyz.atoms,
                        pred_xyz.coords,
                        os.path.join(sample_xtb_dir, "cluster.xyz"),
                    )
                    write_xyz(
                        solv_res["solute_atoms"],
                        solv_res["solute_coords"],
                        os.path.join(sample_xtb_dir, "solute.xyz"),
                    )
                    xyz_dirs.append(sample_xtb_dir)

        if xyz_dirs:
            script_path = os.path.join(save_path, "run_xtb_pred.sh")
            generate_xtb_scripts(xyz_dirs, script_path=script_path)
            print(f"Running xTB on {len(xyz_dirs)} clusters...")
            subprocess.run(["bash", script_path], check=True)

        for res in results:
            prompt_idx = res["prompt_index"]
            for s_idx, sample in enumerate(res["samples"]):
                solv = sample.get("solv", {})
                if solv.get("success"):
                    sample_xtb_dir = os.path.join(xtb_dir, f"p{prompt_idx}_s{s_idx}")
                    e_cluster = parse_xtb_energy(
                        os.path.join(sample_xtb_dir, "cluster.out")
                    )
                    e_solute = parse_xtb_energy(
                        os.path.join(sample_xtb_dir, "solute.out")
                    )
                    sample["pred_eform"] = calculate_eform(
                        e_cluster, e_solute, solv["num_waters"]
                    )
                else:
                    sample["pred_eform"] = None

        if os.path.exists(xtb_dir):
            shutil.rmtree(xtb_dir)

    def aggregate(self, results, duration):
        k_values = self.config.k_values
        rmsd_metrics = topk_rmsd_aggregation(results, k_values, duration)

        num_total = len(results)
        solv_topk = {k: 0 for k in k_values}
        energy_topk = {k: 0 for k in k_values}
        per_n_total: dict[int, int] = {}
        per_n_solv: dict[int, dict[int, int]] = {}
        per_n_energy: dict[int, dict[int, int]] = {}

        threshold = 2.0 / 627.5094740631  # 2 kcal/mol in Hartree
        for res in results:
            prompt_idx = res["prompt_index"]
            gt_eform = self._gt_eforms.get(prompt_idx)
            target_n = self._gt_n_water.get(prompt_idx)
            per_n_total[target_n] = per_n_total.get(target_n, 0) + 1
            per_n_solv.setdefault(target_n, {k: 0 for k in k_values})
            per_n_energy.setdefault(target_n, {k: 0 for k in k_values})

            for k in k_values:
                samples_k = res["samples"][: min(k, len(res["samples"]))]

                if any(s.get("solv", {}).get("success") for s in samples_k):
                    solv_topk[k] += 1
                    per_n_solv[target_n][k] += 1

                if gt_eform is not None:
                    for s in samples_k:
                        if (
                            s.get("solv", {}).get("success")
                            and s.get("pred_eform") is not None
                            and gt_eform + threshold >= s["pred_eform"]
                        ):
                            energy_topk[k] += 1
                            per_n_energy[target_n][k] += 1
                            break

        for k in k_values:
            rmsd_metrics[f"top{k}_solv_success"] = (
                solv_topk[k] / num_total if num_total else 0
            )
            rmsd_metrics[f"top{k}_energy_success"] = (
                energy_topk[k] / num_total if num_total else 0
            )

        for n in sorted(per_n_total):
            total_n = per_n_total[n]
            if total_n == 0:
                continue
            for k in k_values:
                rmsd_metrics[f"top{k}_solv_success_N{n}"] = per_n_solv[n][k] / total_n
                rmsd_metrics[f"top{k}_energy_success_N{n}"] = per_n_energy[n][k] / total_n
            rmsd_metrics[f"count_N{n}"] = total_n

        return rmsd_metrics
