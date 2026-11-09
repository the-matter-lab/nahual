# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Relax qcg_1to10 predictions with xTB, then re-score them.

Runs ``xtb --opt <level> --gfn 2 --alpb water`` on each predicted cluster in
parallel, replaces the prediction with the relaxed geometry, and re-runs the
qcg_1to10 metrics. The default opt level is ``crude``.

Usage:

    python -m evals.tasks.microsolvation.relax \\
        --predictions_pt evals/results/qcg_1to10/<RUN>/predictions.pt \\
        --run_name <RUN>_xtb_relaxed \\
        --timeout 300 --nprocs $(nproc)

Writes predictions.pt and metrics.json to evals/results/qcg_1to10/<run_name>/.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

REPO_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, REPO_ROOT)

from evals.artifacts import make_save_path, save_predictions, save_generate_meta
from nahual.tokens import XYZ, xyzfile

XTB_BIN = os.environ.get("NAHUAL_XTB", shutil.which("xtb") or "xtb")


def _first_xyz(prediction):
    if isinstance(prediction, list):
        for p in prediction:
            if hasattr(p, "atoms") and hasattr(p, "coords"):
                return p
    if hasattr(prediction, "atoms") and hasattr(prediction, "coords"):
        return prediction
    return None


def _run_xtb_relax(work_dir, timeout, opt_level="crude"):
    """Run ``xtb --opt <opt_level>`` in work_dir.

    Returns (xyz_text, energy in Hartree), or (None, None) on failure.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_STACKSIZE"] = "4G"
    cmd = [XTB_BIN, "input.xyz", "--opt", opt_level, "--gfn", "2", "--alpb", "water"]
    try:
        res = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None, None
    if res.returncode != 0:
        return None, None
    opt_path = os.path.join(work_dir, "xtbopt.xyz")
    if not os.path.exists(opt_path):
        return None, None
    try:
        with open(opt_path) as f:
            xyz_text = f.read()
    except OSError:
        return None, None
    energy = None
    energy_file = os.path.join(work_dir, "energy")
    if os.path.exists(energy_file):
        try:
            for line in reversed(open(energy_file).read().splitlines()):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        energy = float(parts[1])
                        break
                    except ValueError:
                        continue
        except OSError:
            pass
    return xyz_text, energy


def parse_xtb_xyz(xyz_text):
    """Parse xtbopt.xyz text into an XYZ dataclass (integer atomic numbers)."""
    from rdkit.Chem import GetPeriodicTable
    pt = GetPeriodicTable()
    lines = xyz_text.strip().split("\n")
    n = int(lines[0])
    atoms = []
    coords = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        sym = parts[0]
        try:
            z = int(sym)
        except ValueError:
            z = pt.GetAtomicNumber(sym)
        atoms.append(z)
        coords.append([float(x) for x in parts[1:4]])
    return XYZ(
        atoms=np.array(atoms, dtype=np.int64),
        coords=np.array(coords, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_pt", type=str, required=True,
                        help="Path to predictions.pt to relax.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Output run name. Default: <input run>_xtb_relaxed_<opt_level>.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(REPO_ROOT, "evals/results/qcg_1to10"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--nprocs", type=int, default=os.cpu_count())
    parser.add_argument("--opt_level", type=str, default="crude",
                        choices=["crude", "sloppy", "loose", "lax", "normal",
                                 "tight", "vtight", "extreme"],
                        help="xtb --opt level (default: crude).")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--dataset", type=str,
                        default="qcg_solv_1to10water_xtbcrest_dynamic_apr2026_test")
    args = parser.parse_args()

    assert os.path.exists(XTB_BIN), f"xtb not found at {XTB_BIN}"
    assert os.path.exists(args.predictions_pt), args.predictions_pt

    if args.run_name is None:
        # input dir name -> append _xtb_relaxed_<level>
        in_dir = os.path.basename(os.path.dirname(os.path.abspath(args.predictions_pt)))
        args.run_name = f"{in_dir}_xtb_relaxed_{args.opt_level}"

    print(f"Loading predictions from {args.predictions_pt}")
    predictions = torch.load(args.predictions_pt, map_location="cpu", weights_only=False)
    print(f"  {len(predictions)} prompts, "
          f"{sum(len(ex['samples']) for ex in predictions)} samples")

    save_path, run_name = make_save_path(args.output_dir, args.run_name, args.dataset)
    print(f"save_path: {save_path}")

    scratch_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    work_root = tempfile.mkdtemp(prefix="qcg_1to10_relax_", dir=scratch_root)
    print(f"work_root: {work_root}")

    # ------------------------------------------------------------------
    # Phase 1: write input.xyz for every sample with a parseable prediction
    # ------------------------------------------------------------------
    t_start = time.time()
    job_index = {}
    skipped = 0
    for ex in predictions:
        idx = ex["prompt_index"]
        for s_idx, sample in enumerate(ex["samples"]):
            xyz = _first_xyz(sample.get("prediction"))
            if xyz is None:
                skipped += 1
                continue
            job_dir = os.path.join(work_root, f"p{idx:06d}_s{s_idx}")
            os.makedirs(job_dir, exist_ok=True)
            with open(os.path.join(job_dir, "input.xyz"), "w") as f:
                f.write(xyzfile(np.asarray(xyz.atoms), np.asarray(xyz.coords)))
            job_index[(idx, s_idx)] = job_dir

    n_jobs = len(job_index)
    print(f"Prepared {n_jobs} jobs (skipped {skipped} samples with no XYZ).")

    # ------------------------------------------------------------------
    # Phase 2: run xtb --opt in parallel
    # ------------------------------------------------------------------
    print(f"Launching xTB --opt {args.opt_level} pool with nprocs={args.nprocs}, "
          f"timeout={args.timeout}s ...")
    relax_t0 = time.time()
    completed = 0
    xtb_results = {}
    with ProcessPoolExecutor(max_workers=args.nprocs) as pool:
        futs = {
            pool.submit(_run_xtb_relax, job_dir, args.timeout, args.opt_level): key
            for key, job_dir in job_index.items()
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                xyz_text, energy = fut.result()
            except Exception as e:
                xyz_text, energy = None, None
                print(f"  {key} worker exception: {e}")
            xtb_results[key] = (xyz_text, energy)
            completed += 1
            if completed % 500 == 0 or completed == n_jobs:
                el = time.time() - relax_t0
                rate = completed / el if el > 0 else 0
                eta = (n_jobs - completed) / rate if rate > 0 else 0
                ok = sum(1 for v in xtb_results.values() if v[0] is not None)
                print(f"  relax {completed}/{n_jobs}  ok={ok}  "
                      f"{rate:.1f}/s  ETA {eta/60:.1f}min")
    relax_dur = time.time() - relax_t0
    n_ok = sum(1 for v in xtb_results.values() if v[0] is not None)
    print(f"Relax phase done: {n_ok}/{n_jobs} succeeded in "
          f"{relax_dur:.1f}s ({relax_dur/60:.2f} min).")

    # ------------------------------------------------------------------
    # Phase 3: replace predictions with relaxed XYZs. Failed slots become [].
    # ------------------------------------------------------------------
    n_replaced = 0
    n_failed_relax = 0
    for ex in predictions:
        idx = ex["prompt_index"]
        for s_idx, sample in enumerate(ex["samples"]):
            key = (idx, s_idx)
            # Drop post-processing fields from the source run. Phase 4 recomputes them.
            for stale in ("solv", "pred_eform", "metric"):
                sample.pop(stale, None)
            if key not in xtb_results:
                continue
            xyz_text, _energy = xtb_results[key]
            if xyz_text is None:
                sample["prediction"] = []
                n_failed_relax += 1
                continue
            try:
                xyz_obj = parse_xtb_xyz(xyz_text)
            except Exception as e:
                print(f"  parse fail p{idx} s{s_idx}: {e}")
                sample["prediction"] = []
                n_failed_relax += 1
                continue
            sample["prediction"] = [xyz_obj]
            n_replaced += 1
    print(f"Replaced {n_replaced} predictions; {n_failed_relax} relax failures "
          f"(blanked); {skipped} samples had no input XYZ to begin with.")

    save_predictions(save_path, predictions)
    save_generate_meta(save_path, time.time() - t_start)
    shutil.rmtree(work_root, ignore_errors=True)

    if args.skip_metrics:
        print("--skip_metrics set, done.")
        return

    # ------------------------------------------------------------------
    # Phase 4: re-run the qcg_1to10 metrics on the relaxed predictions
    # ------------------------------------------------------------------
    print("Running metric pipeline (solv check + xTB single-points + aggregate)...")
    metric_t0 = time.time()
    from evals.catalog import discover_tasks, get_task
    from evals.checkpoint_cli import run_metrics
    discover_tasks()
    task = get_task("qcg_1to10")
    # Reuse the source run's dataset and hyperparameters.
    source_metrics = Path(args.predictions_pt).parent / 'metrics.json'
    if source_metrics.exists():
        source = json.loads(source_metrics.read_text())
        task.config.dataset = source['dataset']
        for key, value in source.get('hyperparams', {}).items():
            if key == 'task_options':
                task.config.extra.update(value)
            elif hasattr(task.config, key):
                setattr(task.config, key, value)

    class MetricArgs:
        checkpoint = (
            f"xtb_relaxed[{args.opt_level}]:{os.path.abspath(args.predictions_pt)}"
        )
        device = "cpu"

    duration = time.time() - t_start
    run_metrics(task, task.config, predictions, save_path, duration, MetricArgs())
    metric_dur = time.time() - metric_t0
    total = time.time() - t_start
    print(f"\nWallclock breakdown:")
    print(f"  relax phase:  {relax_dur:7.1f}s ({relax_dur/60:5.2f} min)")
    print(f"  metric phase: {metric_dur:7.1f}s ({metric_dur/60:5.2f} min)")
    print(f"  total:        {total:7.1f}s ({total/60:5.2f} min)")
    print(f"Saved metrics.json to {save_path}")


if __name__ == "__main__":
    main()
