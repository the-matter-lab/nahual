# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Diverse conformer generation evaluation: SMILES -> set of 3D conformers.

Metrics (following Torsional Diffusion / GeoMol evaluate_confs.py):
  - Coverage Recall (COV-R): fraction of true conformers matched within threshold
  - Average Minimum RMSD Recall (AMR-R): mean min RMSD from each true to nearest pred
  - Coverage Precision (COV-P): fraction of pred conformers matched within threshold
  - Average Minimum RMSD Precision (AMR-P): mean min RMSD from each pred to nearest true

RMSD uses compare_xyz (Kabsch + Hungarian assignment), which handles atom permutation.

Adaptive generation:
  Each sample is a sequence of up to ~10 conformers. For a molecule with K
  reference conformers, samples are read in order:
    - Remove hydrogens from each predicted conformer
    - Accept if the sorted heavy-atom composition matches the reference
    - Stop at 2K valid conformers or 4K conformers examined
    - If fewer than 2K are valid, keep the valid ones
    - If none are valid, the molecule is a failure (AMR=inf, COV=0)
  Each generate() call packs up to `prompt_batch_size` rows from several molecules.

Generation saves checkpoint.pt after each round and resumes from it on restart.
SIGTERM and SIGINT save a checkpoint before exiting, with a 15 s timeout.

Dataset format:
  - instruction: "Generate a diverse ensemble for this SMILES {smi}"
  - response: [text_seg(n_conformers), xyz1, xyz2, ...] (heavy atoms only)
"""

import csv
import hashlib
import math
import json
import os
import re
import signal
import sys
import threading
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

from nahual.compare import compare_xyz
from evals.api import Task, TaskConfig
from evals.metrics.geometry import all_xyzs
from evals.backends.batched import _to_numpy
from evals.backends.checkpoint import CHAT_BEFORE, CHAT_AFTER
from nahual.tokens import XYZ, tokenizer, parse_seq, tokenize_seq, make_seq

from pathlib import Path
COST_CSV_PATH = str(Path(__file__).resolve().parent / "data/test_smiles.csv")


def _remove_hydrogens(xyz: XYZ) -> XYZ:
    """Strip hydrogen atoms (atomic number 1) from an XYZ structure."""
    mask = xyz.atoms != 1
    return XYZ(atoms=xyz.atoms[mask], coords=xyz.coords[mask])


# Coverage thresholds in Å, as in Torsional Diffusion
COV_THRESHOLDS = np.arange(0, 2.5, 0.125)
# Thresholds printed in the summary
REPORT_THRESHOLDS = [0.5, 0.75, 1.0, 1.25]


def _parse_n_conformers(ground_truth) -> int | None:
    """Extract n_conformers from the first text segment of ground truth."""
    if not ground_truth:
        return None
    for item in ground_truth:
        if isinstance(item, str):
            m = re.match(r'^(\d+)$', item.strip())
            if m:
                return int(m.group(1))
    return None


def _collect_valid_conformers(samples, ref_sorted_atoms, target, max_attempts):
    """Collect predicted conformers whose heavy-atom composition matches ref_sorted_atoms.

    Stops at `target` valid conformers or `max_attempts` examined. Returns
    (valid heavy-atom XYZs, total_attempted, n_invalid, n_batches), where
    n_batches is the number of samples consumed.
    """
    valid_xyzs = []
    total_attempted = 0
    n_invalid = 0
    n_batches = 0
    done = False

    for sample in samples:
        if done:
            break
        n_batches += 1
        for xyz in all_xyzs(sample['prediction']):
            if total_attempted >= max_attempts or len(valid_xyzs) >= target:
                done = True
                break
            total_attempted += 1
            cleaned = _remove_hydrogens(xyz)
            if tuple(sorted(cleaned.atoms.tolist())) == ref_sorted_atoms:
                valid_xyzs.append(cleaned)
            else:
                n_invalid += 1

    return valid_xyzs, total_attempted, n_invalid, n_batches


def _pairwise_rmsd_matrix(true_xyzs, pred_xyzs):
    """RMSD matrix (n_true, n_pred) from compare_xyz. Failed entries are inf."""
    n_true = len(true_xyzs)
    n_pred = len(pred_xyzs)
    rmsd_matrix = np.full((n_true, n_pred), np.inf)

    for i in range(n_true):
        for j in range(n_pred):
            try:
                rmsd, _ = compare_xyz(true_xyzs[i], pred_xyzs[j], chirality_matters=False)
                rmsd_matrix[i, j] = rmsd
            except Exception:
                pass  # leave as inf

    return rmsd_matrix


def calc_performance_stats(rmsd_array, thresholds=COV_THRESHOLDS):
    """COV and AMR for recall and precision, as in Torsional Diffusion's calc_performance_stats:
      coverage_recall  = mean(min_over_pred(rmsd) < threshold)    per threshold
      amr_recall       = mean(min_over_pred(rmsd))
      coverage_precision = mean(min_over_true(rmsd) < threshold)  per threshold
      amr_precision    = mean(min_over_true(rmsd))
    """
    # Recall: closest prediction to each true conformer
    cov_recall = np.mean(
        rmsd_array.min(axis=1, keepdims=True) < thresholds, axis=0
    )
    amr_recall = float(rmsd_array.min(axis=1).mean())

    # Precision: closest true conformer to each prediction
    cov_precision = np.mean(
        rmsd_array.min(axis=0, keepdims=True) < np.expand_dims(thresholds, 1),
        axis=1,
    )
    amr_precision = float(rmsd_array.min(axis=0).mean())

    return cov_recall, amr_recall, cov_precision, amr_precision


def _failure_stats():
    """Stats for a molecule with no valid predictions."""
    return (
        np.zeros(len(COV_THRESHOLDS)),   # cov_recall = 0 at all thresholds
        np.inf,                           # amr_recall = inf
        np.zeros(len(COV_THRESHOLDS)),   # cov_precision = 0
        np.inf,                           # amr_precision = inf
    )


RMSD_BLOCK = 16  # reference conformers per parallel RMSD task


def _rmsd_block(item):
    """RMSD rows for a block of one molecule's reference conformers."""
    true_xyzs, pred_xyzs = item
    return _pairwise_rmsd_matrix(true_xyzs, pred_xyzs)


def _generate_rows(model, prompts, max_new_tokens, num_steps, device):
    """One generate() call over (tokens, coords) prompts of different lengths.

    Prompts are left-padded and the padding is masked, so each row is sampled as
    it would be alone. Returns one parsed sequence per row.
    """
    prompt_len = max(len(tokens) for tokens, _ in prompts)
    pad_id = tokenizer.pad_token_id or 0
    rows = len(prompts)

    batch_tokens = np.full((rows, prompt_len), pad_id, dtype=np.int64)
    batch_coords = np.zeros((rows, prompt_len, 3), dtype=np.float32)
    pad_mask = np.zeros((rows, prompt_len), dtype=bool)
    for b, (tokens, coords) in enumerate(prompts):
        batch_tokens[b, prompt_len - len(tokens):] = tokens
        batch_coords[b, prompt_len - len(tokens):] = coords
        pad_mask[b, prompt_len - len(tokens):] = True

    with torch.no_grad():
        batch_out_tokens, batch_out_coords = model.generate(
            prefix=(torch.tensor(batch_tokens).to(device), torch.tensor(batch_coords).to(device)),
            max_len=max_new_tokens + prompt_len,
            num_steps=num_steps,
            keep_prefix=False,
            pad_mask=torch.tensor(pad_mask).to(device),
            pbar=False,
        )

    batch_out_tokens = _to_numpy(batch_out_tokens)
    batch_out_coords = _to_numpy(batch_out_coords)
    predictions = []
    for b in range(rows):
        pred_seq = parse_seq(batch_out_tokens[b], batch_out_coords[b])
        predictions.append(pred_seq if pred_seq is not None else [])
    return predictions


def _molecule_state(dataset, idx):
    """Per-molecule sampling state. Checkpoints keep only the _PERSISTENT fields."""
    row = dataset[idx]
    instruction, response = row["instruction"], row["response"]
    if isinstance(instruction, str):
        instruction = json.loads(instruction)
    if isinstance(response, str):
        response = json.loads(response)
    prompt_segs = make_seq(instruction)
    response_segs = make_seq(response)

    n_conformers = _parse_n_conformers(response_segs)
    true_xyzs = all_xyzs(response_segs)
    k = n_conformers if n_conformers is not None else len(true_xyzs)

    sa, sx = tokenize_seq(prompt_segs)
    sa = np.concatenate([CHAT_BEFORE, sa, CHAT_AFTER], axis=0)
    sx = np.concatenate([
        np.zeros((len(CHAT_BEFORE), 3), dtype=np.float32),
        sx,
        np.zeros((len(CHAT_AFTER), 3), dtype=np.float32),
    ], axis=0)
    return {
        'prompt_index': idx, 'prompt': prompt_segs, 'ground_truth': response_segs,
        'samples': [], 'n_valid': 0, 'n_total_conformers': 0, 'n_rounds': 0,
        # Derived, rebuilt on resume:
        'k': k, 'target': 2 * k, 'max_attempts': 4 * k,
        'ref_sorted_atoms': tuple(sorted(true_xyzs[0].atoms.tolist())) if true_xyzs else None,
        'tokens': sa, 'coords': sx,
    }


_PERSISTENT = ('prompt_index', 'prompt', 'ground_truth', 'samples', 'n_valid', 'n_total_conformers', 'n_rounds')


def _molecule_done(state):
    return (state['n_valid'] >= state['target'] or state['n_total_conformers'] >= state['max_attempts']
            or len(state['samples']) >= state['max_attempts'] or state['ref_sorted_atoms'] is None)


def _consume(state, predictions):
    """Append new samples in order until the stopping rule is met. Later samples are discarded."""
    state['n_rounds'] += 1
    for pred_seq in predictions:
        state['samples'].append({'prediction': pred_seq})
        for xyz in all_xyzs(pred_seq):
            state['n_total_conformers'] += 1
            cleaned = _remove_hydrogens(xyz)
            if tuple(sorted(cleaned.atoms.tolist())) == state['ref_sorted_atoms']:
                state['n_valid'] += 1
            if state['n_valid'] >= state['target'] or state['n_total_conformers'] >= state['max_attempts']:
                return
        # Samples with no XYZ blocks also count against the attempt budget.
        if len(state['samples']) >= state['max_attempts']:
            return


def _rows_needed(state, cap):
    """Estimate the rows needed to reach the valid target or the attempt cap.

    Uses the observed valid and attempted conformers per row. A shortfall is
    covered by the next round.
    """
    seen = len(state['samples'])
    valid_rate = state['n_valid'] / seen if seen else 8.0
    attempt_rate = state['n_total_conformers'] / seen if seen else 8.0
    target_rows = (state['target'] - state['n_valid']) / max(valid_rate, 0.5)
    attempt_rows = (state['max_attempts'] - state['n_total_conformers']) / max(attempt_rate, 1.0)
    rows = math.ceil(min(target_rows, attempt_rows))
    return max(1, min(rows, cap, state['max_attempts'] - seen))


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def _config_hash(checkpoint_path, task_name, batch_size, max_new_tokens, assigned_indices):
    """Hash of the run config, used to detect stale checkpoints."""
    key = f"{checkpoint_path}|{task_name}|{batch_size}|{max_new_tokens}|{sorted(assigned_indices)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _save_checkpoint(save_path, completed, in_progress, assigned_indices, config_hash):
    """Atomically write checkpoint.pt."""
    state = {
        'completed': completed,
        'in_progress': in_progress,
        'assigned_indices': assigned_indices,
        'config_hash': config_hash,
        'rng_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp_path = os.path.join(save_path, 'checkpoint.pt.tmp')
    final_path = os.path.join(save_path, 'checkpoint.pt')
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)


def _load_checkpoint(save_path, config_hash):
    """Load checkpoint if it exists and config_hash matches.

    Returns (completed, in_progress) or None.
    """
    path = os.path.join(save_path, 'checkpoint.pt')
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, weights_only=False)
    except Exception as e:
        print(f"Warning: corrupt checkpoint.pt, starting fresh: {e}")
        return None
    if state.get('config_hash') != config_hash:
        print(f"Warning: checkpoint config_hash mismatch (stale checkpoint), starting fresh.")
        return None
    rng = state.get('rng_state')
    if rng is None:
        # Without the RNG state, resumed molecules could repeat earlier conformers.
        print('Checkpoint has no RNG state. Keeping completed molecules '
              'and restarting unfinished ones.', flush=True)
        return state['completed'], []
    torch.set_rng_state(rng['torch'])
    np.random.set_state(rng['numpy'])
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
    return state['completed'], state['in_progress']


class _SIGTERMHandler:
    """Save a checkpoint and exit on SIGTERM or SIGINT, with a 15 s timeout."""

    def __init__(self):
        self.triggered = False
        self._checkpoint_fn = None
        self._timer = None

    def register(self, checkpoint_fn):
        self._checkpoint_fn = checkpoint_fn
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame):
        if self.triggered:
            return  # already handling
        self.triggered = True
        print(f"\nReceived signal {signum}, saving checkpoint (15s timeout)...", flush=True)
        self._timer = threading.Timer(15.0, self._force_save_and_exit)
        self._timer.daemon = True
        self._timer.start()

    def _force_save_and_exit(self):
        print("Timeout reached, force-saving checkpoint and exiting.", flush=True)
        if self._checkpoint_fn:
            try:
                self._checkpoint_fn()
            except Exception as e:
                print(f"Error saving checkpoint on timeout: {e}", flush=True)
        os._exit(0)

    def cancel_timer(self):
        if self._timer:
            self._timer.cancel()


# ---------------------------------------------------------------------------
# Cost estimation for greedy load balancing
# ---------------------------------------------------------------------------

def compute_molecule_costs(dataset, csv_path=COST_CSV_PATH, num_examples=None):
    """Cost estimate K * n_atoms**2 per example, read from each row's response."""
    n = num_examples if num_examples is not None else len(dataset)
    n = min(n, len(dataset))

    costs = []
    for i in range(n):
        row = dataset[i]
        response = row["response"]
        if isinstance(response, str):
            response = json.loads(response)
        segs = make_seq(response)
        xyzs = all_xyzs(segs)
        n_atoms = len(xyzs[0].atoms) if xyzs else 1
        # Read K from the row itself.
        k = _parse_n_conformers(segs) or len(xyzs) or 1
        costs.append(k * n_atoms * n_atoms)

    return costs


class GeomconfDiverseTask(Task):
    def __init__(self, config: TaskConfig):
        self.config = config

    def artifacts(self):
        matrices = getattr(self, "_rmsd_matrices", None)
        return {"rmsd_matrices.pt": matrices} if matrices is not None else {}

    def setup(self, args=None):
        pass

    @staticmethod
    def compute_costs(dataset, num_examples=None):
        """Per-example cost estimates for greedy GPU load balancing."""
        return compute_molecule_costs(dataset, COST_CSV_PATH, num_examples)

    def custom_generate(self, model, dataset, config, args):
        """Packed adaptive generation with checkpointing.

        Each round fills up to `prompt_batch_size` rows, first with molecules in
        progress and then with new ones. Each molecule gets about as many rows as
        it still needs, and stops at 2K valid conformers or after 4K conformers
        or 4K samples.
        """
        rows_per_round = config.prompt_batch_size
        max_new_tokens = config.max_new_tokens
        num_steps = config.num_steps
        device = getattr(args, 'device', 'cuda')
        save_path = getattr(args, 'save_path', None)

        # Determine assigned indices
        indices_arg = getattr(args, 'indices', None)
        if indices_arg:
            all_indices = list(indices_arg)
        else:
            num_examples = len(dataset)
            offset = getattr(args, 'offset', 0)
            num_ex_override = getattr(args, 'num_examples', None) or config.num_examples
            if num_ex_override is not None:
                num_examples = min(num_examples - offset, num_ex_override)
            else:
                num_examples = num_examples - offset
            all_indices = list(range(offset, offset + num_examples))

        print(f"Packed adaptive generation for {len(all_indices)} examples "
              f"(indices={all_indices[:5]}{'...' if len(all_indices) > 5 else ''}), "
              f"rows_per_round={rows_per_round}, batch_policy=remaining_attempts")

        # Config hash for checkpoint validation
        ckpt_path = getattr(args, 'checkpoint', '') or ''
        chash = _config_hash(ckpt_path, config.name + '|packed', rows_per_round, max_new_tokens, all_indices)

        completed = []
        active = {}  # prompt_index -> state, in admission order
        if save_path:
            ckpt = _load_checkpoint(save_path, chash)
            if ckpt:
                completed, in_progress = ckpt
                for saved in in_progress or []:
                    state = _molecule_state(dataset, saved['prompt_index'])
                    state.update({key: saved[key] for key in _PERSISTENT})
                    active[state['prompt_index']] = state
                print(f"Resumed from checkpoint: {len(completed)} completed, {len(active)} in progress")

        def save():
            if save_path:
                in_progress = [{key: st[key] for key in _PERSISTENT} for st in active.values()]
                _save_checkpoint(save_path, completed, in_progress, all_indices, chash)

        sigterm = _SIGTERMHandler()
        sigterm.register(save)

        started = {r['prompt_index'] for r in completed} | set(active)
        pending = [i for i in all_indices if i not in started]

        def finish(state):
            print(f"  [{state['prompt_index']}] K={state['k']}, target=2K={state['target']}, "
                  f"valid={state['n_valid']}, total={state['n_total_conformers']}, "
                  f"rounds={state['n_rounds']}, samples={len(state['samples'])}", flush=True)
            completed.append({key: state[key] for key in ('prompt_index', 'prompt', 'ground_truth', 'samples')})

        with tqdm(total=len(all_indices), initial=len(completed)) as bar:
            while active or pending:
                plan, budget = [], rows_per_round
                for state in active.values():
                    if budget:
                        take = _rows_needed(state, budget)
                        plan.append((state, take))
                        budget -= take
                while budget and pending:
                    state = _molecule_state(dataset, pending.pop(0))
                    if _molecule_done(state):  # No reference atoms: nothing to sample.
                        finish(state)
                        bar.update(1)
                        continue
                    active[state['prompt_index']] = state
                    take = _rows_needed(state, budget)
                    plan.append((state, take))
                    budget -= take
                if not plan:
                    continue

                prompts = [(state['tokens'], state['coords']) for state, take in plan for _ in range(take)]
                predictions = _generate_rows(model, prompts, max_new_tokens, num_steps, device)
                offset = 0
                for state, take in plan:
                    _consume(state, predictions[offset:offset + take])
                    offset += take
                    if _molecule_done(state):
                        del active[state['prompt_index']]
                        finish(state)
                        bar.update(1)
                save()

                if sigterm.triggered:
                    sigterm.cancel_timer()
                    print(f"SIGTERM: checkpoint saved with {len(active)} molecules in progress.")
                    sys.exit(0)

        order = {idx: n for n, idx in enumerate(all_indices)}
        completed.sort(key=lambda r: order[r['prompt_index']])

        # All done: clean up checkpoint
        if save_path:
            checkpoint_file = os.path.join(save_path, 'checkpoint.pt')
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print("Removed checkpoint.pt (all molecules complete).")

        sigterm.cancel_timer()
        return completed

    def metric(self, ground_truth, prediction, prompt=None):
        """Count the conformers in one sample. COV and AMR are computed in aggregate()."""
        pred_xyzs = all_xyzs(prediction)

        return {
            'score': 0.0,
            'valid': len(pred_xyzs) > 0,
            'info': {
                'n_pred_conformers': len(pred_xyzs),
            },
        }

    def aggregate(self, results, duration):
        """Pool valid conformers across samples and compute COV and AMR per molecule.

        A molecule with no valid conformers is a failure (AMR=inf, COV=0).
        AMR means skip failures, and AMR medians count them as inf.
        example_invalid_rate is n_invalid / total_attempted.
        """
        # Phase 1: collect valid conformers per molecule
        work_items = []       # (true_xyzs, valid_pred_xyzs) or None for failures
        per_mol_info = []     # per-example stats
        n_total = 0
        n_failures = 0

        for res in results:
            n_total += 1
            gt = res['ground_truth']
            n_conformers = _parse_n_conformers(gt)
            true_xyzs = all_xyzs(gt)

            if not true_xyzs:
                n_failures += 1
                work_items.append(None)
                per_mol_info.append({
                    'failure': True, 'reason': 'no_ground_truth',
                    'n_valid': 0, 'target': 0, 'reached_target': False,
                    'total_attempted': 0, 'n_invalid': 0, 'n_batches': 0,
                    'example_invalid_rate': 0.0,
                })
                continue

            k = n_conformers if n_conformers is not None else len(true_xyzs)
            target = 2 * k
            max_attempts = 4 * k
            ref_sorted_atoms = tuple(sorted(true_xyzs[0].atoms.tolist()))

            valid_xyzs, total_attempted, n_invalid, n_batches = \
                _collect_valid_conformers(
                    res['samples'], ref_sorted_atoms, target, max_attempts
                )

            is_failure = len(valid_xyzs) == 0
            reached_target = len(valid_xyzs) >= target
            example_invalid_rate = n_invalid / total_attempted if total_attempted > 0 else 0.0

            info = {
                'failure': is_failure,
                'reason': 'no_valid_conformers' if is_failure else None,
                'n_valid': len(valid_xyzs),
                'target': target,
                'reached_target': reached_target,
                'total_attempted': total_attempted,
                'n_invalid': n_invalid,
                'n_batches': n_batches,
                'example_invalid_rate': example_invalid_rate,
            }
            per_mol_info.append(info)

            if is_failure:
                n_failures += 1
                work_items.append(None)
            else:
                work_items.append((true_xyzs, valid_xyzs))

        # Phase 2: RMSD matrices in parallel
        valid_work = [(i, item) for i, item in enumerate(work_items) if item is not None]
        n_workers = min(int(os.environ.get("NAHUAL_METRIC_WORKERS", "16")), max(len(valid_work), 1))
        print(f"\nComputing compare_xyz RMSD for {len(valid_work)} molecules "
              f"using {n_workers} workers...")

        # Split each matrix into blocks of reference rows, largest first, to balance workers.
        blocks = [(m, start) for m, (_, (true_xyzs, _)) in enumerate(valid_work)
                  for start in range(0, len(true_xyzs), RMSD_BLOCK)]
        tasks = [(valid_work[m][1][0][start:start + RMSD_BLOCK], valid_work[m][1][1])
                 for m, start in blocks]
        order = sorted(range(len(tasks)), key=lambda t: -len(tasks[t][0]) * len(tasks[t][1]))
        if n_workers > 1 and len(tasks) > 1:
            with Pool(n_workers) as pool:
                done = pool.map(_rmsd_block, [tasks[t] for t in order], chunksize=1)
        else:
            done = [_rmsd_block(tasks[t]) for t in order]
        rows = [None] * len(tasks)
        for t, block in zip(order, done):
            rows[t] = block
        per_molecule = [[] for _ in valid_work]
        for (m, _), block in zip(blocks, rows):
            per_molecule[m].append(block)
        computed_stats = []
        for parts in per_molecule:
            rmsd_mat = np.vstack(parts)
            computed_stats.append((calc_performance_stats(rmsd_mat, COV_THRESHOLDS), rmsd_mat))

        # Phase 3: per-molecule stats
        per_mol_stats = [None] * n_total
        rmsd_matrices = [None] * n_total
        for idx in range(n_total):
            if work_items[idx] is None:
                per_mol_stats[idx] = _failure_stats()

        for (orig_idx, _), (stats, rmsd_mat) in zip(valid_work, computed_stats):
            per_mol_stats[orig_idx] = stats
            rmsd_matrices[orig_idx] = rmsd_mat

        n_mols = n_total

        if n_mols == 0:
            return {
                'n_molecules': 0,
                'n_failures': n_failures,
                'n_total': n_total,
                'failure_rate': 1.0,
                'duration_seconds': duration,
            }

        # Unzip per-molecule stats
        coverage_recall, amr_recall, coverage_precision, amr_precision = zip(
            *per_mol_stats
        )

        amr_r = np.array(amr_recall)
        amr_p = np.array(amr_precision)

        metrics = {}

        for i, thresh in enumerate(COV_THRESHOLDS):
            cov_r_vals = [float(cr[i]) for cr in coverage_recall]
            cov_p_vals = [float(cp[i]) for cp in coverage_precision]

            t_label = f"{thresh:.3f}".rstrip('0').rstrip('.')
            metrics[f'cov_recall_mean@{t_label}'] = float(np.mean(cov_r_vals))
            metrics[f'cov_recall_median@{t_label}'] = float(np.median(cov_r_vals))
            metrics[f'cov_precision_mean@{t_label}'] = float(np.mean(cov_p_vals))
            metrics[f'cov_precision_median@{t_label}'] = float(np.median(cov_p_vals))

        # AMR means skip failures. Medians count them as inf.
        amr_r_for_mean = np.where(np.isinf(amr_r), np.nan, amr_r)
        amr_p_for_mean = np.where(np.isinf(amr_p), np.nan, amr_p)
        metrics['amr_recall_mean'] = float(np.nanmean(amr_r_for_mean))
        metrics['amr_recall_median'] = float(np.median(amr_r))
        metrics['amr_precision_mean'] = float(np.nanmean(amr_p_for_mean))
        metrics['amr_precision_median'] = float(np.median(amr_p))

        # Per-example stats
        invalid_rates = [info['example_invalid_rate'] for info in per_mol_info]
        n_batches_list = [info['n_batches'] for info in per_mol_info]
        reached_list = [info['reached_target'] for info in per_mol_info]

        metrics['mean_example_invalid_rate'] = float(np.mean(invalid_rates))
        metrics['median_example_invalid_rate'] = float(np.median(invalid_rates))
        metrics['mean_n_batches'] = float(np.mean(n_batches_list))
        metrics['frac_reached_2k'] = float(np.mean(reached_list))

        # Summary
        metrics['n_molecules'] = n_mols
        metrics['n_failures'] = n_failures
        metrics['n_total'] = n_total
        metrics['failure_rate'] = n_failures / n_total if n_total > 0 else 0.0
        metrics['duration_seconds'] = duration

        # Same print format as Torsional Diffusion
        print(f"\n{n_mols} conformer sets, {n_failures} failures "
              f"({metrics['failure_rate']:.1%}), "
              f"{metrics['frac_reached_2k']:.1%} reached 2K target")
        print(f"Mean invalid rate per example: {metrics['mean_example_invalid_rate']:.2%}")
        print(f"Mean batches consumed: {metrics['mean_n_batches']:.1f}")

        for t in REPORT_THRESHOLDS:
            idx = int(np.argmin(np.abs(COV_THRESHOLDS - t)))
            t_label = f"{COV_THRESHOLDS[idx]:.3f}".rstrip('0').rstrip('.')

            cov_r_mean = metrics[f'cov_recall_mean@{t_label}']
            cov_r_med = metrics[f'cov_recall_median@{t_label}']
            cov_p_mean = metrics[f'cov_precision_mean@{t_label}']
            cov_p_med = metrics[f'cov_precision_median@{t_label}']

            print(f"threshold {COV_THRESHOLDS[idx]:.3f}")
            print(f"  Recall Coverage:    Mean = {cov_r_mean * 100:.2f}, "
                  f"Median = {cov_r_med * 100:.2f}")
            print(f"  Recall AMR:         Mean = {metrics['amr_recall_mean']:.4f}, "
                  f"Median = {metrics['amr_recall_median']:.4f}")
            print(f"  Precision Coverage: Mean = {cov_p_mean * 100:.2f}, "
                  f"Median = {cov_p_med * 100:.2f}")
            print(f"  Precision AMR:      Mean = {metrics['amr_precision_mean']:.4f}, "
                  f"Median = {metrics['amr_precision_median']:.4f}")

        # Enrich per-example info with RMSD stats at key thresholds
        for idx in range(n_total):
            cov_r, amr_r_val, cov_p, amr_p_val = per_mol_stats[idx]
            per_mol_info[idx]['amr_recall'] = float(amr_r_val)
            per_mol_info[idx]['amr_precision'] = float(amr_p_val)
            for t in REPORT_THRESHOLDS:
                t_idx = int(np.argmin(np.abs(COV_THRESHOLDS - t)))
                t_label = f"{t:.3f}".rstrip('0').rstrip('.')
                per_mol_info[idx][f'cov_recall@{t_label}'] = float(cov_r[t_idx])
                per_mol_info[idx][f'cov_precision@{t_label}'] = float(cov_p[t_idx])

        # Store per-example info in metrics for debugging
        metrics['per_example'] = per_mol_info

        # Saved as rmsd_matrices.pt by artifacts(), for analysis at other thresholds.
        self._rmsd_matrices = rmsd_matrices

        return metrics
