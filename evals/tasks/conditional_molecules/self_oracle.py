# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-of-5 evaluation on QM9-EDM conditional generation.

For each property (alpha, gap, homo, lumo, mu, Cv), averaged over prompts:

  EGNN oracle (sample["egnn_pred"]):
    - mae@1            : |egnn_pred[0] - target|
    - mae@5_oracle     : min_i |egnn_pred[i] - target|

  Self oracle, where the model predicts the property of its own samples:
    - self_mae@1                : |self_pred[0] - target|
    - self_mae@5_oracle         : min_i |self_pred[i] - target|
    - self_selected_egnn_mae@5  : |egnn_pred[i] - target| for the sample
                                  i = argmin_i |self_pred[i] - target|

Predictions are read from
  evals/results/qm9_edm_cond_<prop>/<run_suffix>_qm9_edm_cond_<prop>_n1000/predictions.pt
and generated if missing.

The self-oracle prompt is the trained property-prediction format:
  "Predict the <display_name> of this molecule in <units>." + <XYZ>
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nahual.tokens import XYZ, parse_seq, tokenize_seq, tokenizer
from evals.metrics.geometry import first_xyz, full_text
from evals.backends.checkpoint import CHAT_AFTER, CHAT_BEFORE, load_checkpoint_model
from evals.backends.batched import generate_predictions
from evals.datasets import load_dataset_split
from evals.artifacts import load_predictions, save_predictions
from evals.catalog import get_task

PROPS = ["alpha", "gap", "homo", "lumo", "mu", "Cv"]

PROP_SELF_PROMPT = {
    "alpha": "Predict the polarizability of this molecule in Bohr^3.",
    "gap":   "Predict the HOMO-LUMO gap of this molecule in meV.",
    "homo":  "Predict the HOMO energy of this molecule in meV.",
    "lumo":  "Predict the LUMO energy of this molecule in meV.",
    "mu":    "Predict the dipole moment of this molecule in D.",
    "Cv":    "Predict the heat capacity of this molecule in cal/(mol*K).",
}

PROP_UNIT = {
    "alpha": "Bohr^3",
    "gap":   "meV",
    "homo":  "meV",
    "lumo":  "meV",
    "mu":    "D",
    "Cv":    "cal/(mol*K)",
}

PROP_PARSE = {
    "alpha": re.compile(r'(-?[\d\.]+)\s*Bohr\^3'),
    "gap":   re.compile(r'(-?[\d\.]+)\s*meV'),
    "homo":  re.compile(r'(-?[\d\.]+)\s*meV'),
    "lumo":  re.compile(r'(-?[\d\.]+)\s*meV'),
    "mu":    re.compile(r'(-?[\d\.]+)\s*D\b'),
    "Cv":    re.compile(r'(-?[\d\.]+)\s*cal/\(mol\*K\)'),
}


# ---------------------------------------------------------------------------
# Generation: build property-prediction prompt + run model in batches
# ---------------------------------------------------------------------------

def _build_self_prompt(prompt_text: str, xyz: XYZ):
    """Tokenize "<text>" + XYZ with the chat template."""
    seq = [prompt_text, xyz]
    sa, sx = tokenize_seq(seq)
    sa = np.concatenate([CHAT_BEFORE, sa, CHAT_AFTER], axis=0)
    sx = np.concatenate([
        np.zeros((len(CHAT_BEFORE), 3), dtype=np.float32),
        sx,
        np.zeros((len(CHAT_AFTER), 3), dtype=np.float32),
    ], axis=0)
    return sa, sx


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x


def _parse_value(text: str, pattern: re.Pattern):
    if not isinstance(text, str):
        return None
    m = pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


@torch.no_grad()
def run_self_oracle(model, prop_key, xyzs, batch_size=64, max_new_tokens=24, num_steps=2,
                    device="cuda"):
    """Predict the property of each xyz with the model. Returns a list of float or None."""
    prompt_text = PROP_SELF_PROMPT[prop_key]
    pattern = PROP_PARSE[prop_key]
    pad_id = tokenizer.pad_token_id or 0

    # None xyzs are skipped and stay None.
    pre = []
    for xyz in xyzs:
        if xyz is None:
            pre.append(None)
        else:
            pre.append(_build_self_prompt(prompt_text, xyz))

    self_preds = [None] * len(xyzs)
    pending_idxs = [i for i, p in enumerate(pre) if p is not None]

    for batch_start in range(0, len(pending_idxs), batch_size):
        idx_batch = pending_idxs[batch_start:batch_start + batch_size]
        seqs_a = [pre[i][0] for i in idx_batch]
        seqs_x = [pre[i][1] for i in idx_batch]
        lens = [len(s) for s in seqs_a]
        max_len = max(lens)
        bsz = len(idx_batch)

        batch_tokens = np.full((bsz, max_len), pad_id, dtype=np.int64)
        batch_coords = np.zeros((bsz, max_len, 3), dtype=np.float32)
        pad_mask = np.zeros((bsz, max_len), dtype=bool)
        for b in range(bsz):
            l = lens[b]
            off = max_len - l
            batch_tokens[b, off:] = seqs_a[b]
            batch_coords[b, off:] = seqs_x[b]
            pad_mask[b, off:] = 1

        bt = torch.tensor(batch_tokens).to(device)
        bc = torch.tensor(batch_coords).to(device)
        pm = torch.tensor(pad_mask).to(device)

        out_tokens, out_coords = model.generate(
            prefix=(bt, bc),
            max_len=max_new_tokens + max_len,
            num_steps=num_steps,
            keep_prefix=False,
            pad_mask=pm,
            temperature=0.0,
            pbar=False,
        )

        for b, abs_idx in enumerate(idx_batch):
            tok_seq = _to_numpy(out_tokens[b])
            crd_seq = _to_numpy(out_coords[b])
            seq = parse_seq(tok_seq, crd_seq)
            text = full_text(seq) if seq is not None else ""
            self_preds[abs_idx] = _parse_value(text, pattern)

    return self_preds


# ---------------------------------------------------------------------------
# Predictions: locate, optionally generate
# ---------------------------------------------------------------------------

DIRECT_RUN_NAME = None

def predictions_dir(prop, run_suffix, pred_suffix="_n1000"):
    if DIRECT_RUN_NAME:
        return REPO_ROOT / "evals" / "results" / f"qm9_edm_cond_{prop}" / DIRECT_RUN_NAME
    return REPO_ROOT / "evals" / "results" / f"qm9_edm_cond_{prop}" / \
        f"{run_suffix}_qm9_edm_cond_{prop}{pred_suffix}"


def ensure_predictions(prop, run_suffix, model, num_examples=1000, pred_suffix="_n1000"):
    """Load predictions, or generate them if missing."""
    pred_path = predictions_dir(prop, run_suffix, pred_suffix)
    pt_path = pred_path / "predictions.pt"
    if pt_path.exists():
        print(f"  [{prop}] using existing predictions: {pt_path}")
        return load_predictions(str(pred_path)), str(pred_path), False

    print(f"  [{prop}] predictions not found at {pt_path}, generating...")
    pred_path.mkdir(parents=True, exist_ok=True)
    task = get_task(f"qm9_edm_cond_{prop}")
    config = task.config
    config.num_examples = num_examples

    dataset = load_dataset_split(config.dataset)

    # Minimal args object the generate path expects.
    class _Args:
        pass
    args = _Args()
    args.device = "cuda"
    args.seed = 0
    args.offset = 0
    args.num_examples = num_examples
    args.indices = None
    args.save_path = str(pred_path)
    args.temperature = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    predictions = generate_predictions(model, dataset, config, args)
    return predictions, str(pred_path), True


# ---------------------------------------------------------------------------
# Aggregation: compute metrics
# ---------------------------------------------------------------------------

def compute_metrics(prop, predictions):
    """EGNN-oracle and self-oracle MAE metrics."""
    unit = PROP_UNIT[prop]
    egnn_at1, egnn_oracle = [], []
    self_at1, self_oracle = [], []
    self_selected_egnn = []
    n_total = len(predictions)
    n_target_missing = 0
    n_self_pred_failed = 0
    n_egnn_failed = 0

    for res in predictions:
        target = res.get("target_value")
        if target is None:
            n_target_missing += 1
            continue
        samples = res.get("samples", [])

        egnn_preds = [s.get("egnn_pred") for s in samples]
        self_preds = [s.get("self_pred") for s in samples]

        # EGNN oracle
        if egnn_preds and egnn_preds[0] is not None:
            egnn_at1.append(abs(egnn_preds[0] - target))
        valid_egnn = [p for p in egnn_preds if p is not None]
        if valid_egnn:
            egnn_oracle.append(min(abs(p - target) for p in valid_egnn))
        n_egnn_failed += sum(1 for p in egnn_preds if p is None)

        # Self oracle
        if self_preds and self_preds[0] is not None:
            self_at1.append(abs(self_preds[0] - target))
        valid_self = [(i, p) for i, p in enumerate(self_preds) if p is not None]
        if valid_self:
            best_i, _ = min(valid_self, key=lambda kv: abs(kv[1] - target))
            self_oracle.append(abs(self_preds[best_i] - target))
            if egnn_preds[best_i] is not None:
                self_selected_egnn.append(abs(egnn_preds[best_i] - target))
        n_self_pred_failed += sum(1 for p in self_preds if p is None)

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    metrics = {
        "n_total": n_total,
        "n_target_missing": n_target_missing,
        "n_egnn_failed": n_egnn_failed,
        "n_self_pred_failed": n_self_pred_failed,
        "unit": unit,
        f"egnn_mae_at1_{unit}": _mean(egnn_at1),
        f"egnn_mae_at5_oracle_{unit}": _mean(egnn_oracle),
        f"self_mae_at1_{unit}": _mean(self_at1),
        f"self_mae_at5_oracle_{unit}": _mean(self_oracle),
        f"self_selected_egnn_mae_at5_{unit}": _mean(self_selected_egnn),
        "n_egnn_at1": len(egnn_at1),
        "n_egnn_oracle": len(egnn_oracle),
        "n_self_at1": len(self_at1),
        "n_self_oracle": len(self_oracle),
        "n_self_selected_egnn": len(self_selected_egnn),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score_samples(samples_dir, props, output_dir):
    """Self-reward metrics from saved samples, which hold egnn_pred and self_pred; no model needed."""
    summary = {"samples_dir": str(samples_dir), "props": {}}
    for prop in props:
        pred_dir = Path(samples_dir) / f"qm9_edm_cond_{prop}"
        metrics = compute_metrics(prop, load_predictions(str(pred_dir)))
        (output_dir / f"{prop}.json").write_text(
            json.dumps({"property": prop, "predictions_dir": str(pred_dir), "metrics": metrics}, indent=4) + "\n")
        summary["props"][prop] = metrics
        print(prop, json.dumps(metrics), flush=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=4) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", "--samples-dir", default=None,
                        help="Score saved samples only: a folder with qm9_edm_cond_<prop>/predictions.pt "
                             "(for example samples/nahual/seed0 from the Hub). Writes to --output_dir "
                             "(default <samples_dir>/qm9_edm_self_oracle).")
    parser.add_argument("--pred-run-name", default=None)
    parser.add_argument("--checkpoint", help="Path to model .pt checkpoint")
    parser.add_argument(
        "--run_suffix",
        help="Run-name prefix used to locate predictions.pt (and to generate if missing).",
    )
    parser.add_argument("--num_examples", type=int, default=1000)
    parser.add_argument("--props", default=",".join(PROPS),
                        help="Comma-separated subset of properties to evaluate.")
    parser.add_argument("--output_dir",
                        default=None,
                        help="Output dir for per-property metrics and summary. "
                             "Default: evals/results/qm9_edm_self_oracle/<run_suffix>/.")
    parser.add_argument("--pred_suffix", default="_n1000",
                        help="Suffix on the predictions dir (e.g. '_full' for full-set runs).")
    parser.add_argument("--self_batch_size", type=int, default=64)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard_id", type=int, default=None,
                        help="Process only samples [shard_id::num_shards] and save them to "
                             "{prop}_shard{N}.pt. Metrics are computed by --merge_only.")
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--merge_only", action="store_true",
                        help="Merge shard files from output_dir and compute metrics, "
                             "without loading the model.")
    args = parser.parse_args()
    global DIRECT_RUN_NAME
    DIRECT_RUN_NAME = args.pred_run_name

    props = [p for p in args.props.split(",") if p]
    for p in props:
        if p not in PROPS:
            raise SystemExit(f"Unknown property: {p}. Expected one of {PROPS}.")
    if args.samples_dir:
        output_dir = Path(args.output_dir or Path(args.samples_dir) / "qm9_edm_self_oracle")
        output_dir.mkdir(parents=True, exist_ok=True)
        return score_samples(args.samples_dir, props, output_dir)
    if not args.checkpoint or not args.run_suffix:
        parser.error("--checkpoint and --run_suffix are required unless --samples_dir is given")

    if args.output_dir is None:
        output_dir = REPO_ROOT / "evals" / "results" / "qm9_edm_self_oracle" / args.run_suffix
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_shard_worker = args.shard_id is not None
    if is_shard_worker:
        if args.num_shards is None or args.num_shards <= 0:
            raise SystemExit("--shard_id requires --num_shards > 0")
        if not (0 <= args.shard_id < args.num_shards):
            raise SystemExit(f"--shard_id must be in [0, {args.num_shards})")

    if args.merge_only:
        gen_model = None
    else:
        print(f"Loading model from {args.checkpoint}...")
        gen_model = load_checkpoint_model(args.checkpoint, device=args.device, ema=not args.no_ema)

    summary = {
        "checkpoint": args.checkpoint,
        "run_suffix": args.run_suffix,
        "num_examples": args.num_examples,
        "no_ema": args.no_ema,
        "props": {},
    }

    overall_t0 = time.time()
    for prop in props:
        prop_t0 = time.time()
        print(f"\n=== {prop} ===")
        if args.merge_only:
            pred_path = predictions_dir(prop, args.run_suffix, args.pred_suffix)
            predictions = load_predictions(str(pred_path))
            pred_dir = str(pred_path)
            regenerated = False
        else:
            predictions, pred_dir, regenerated = ensure_predictions(
                prop, args.run_suffix, gen_model, num_examples=args.num_examples,
                pred_suffix=args.pred_suffix,
            )

        # Run the EGNN if egnn_pred or target_value is missing.
        needs_egnn = any(
            "egnn_pred" not in s for res in predictions for s in res.get("samples", [])
        )
        # Also re-run if target_value isn't populated.
        needs_target = any("target_value" not in res for res in predictions)
        if needs_egnn or needs_target:
            print(f"  [{prop}] running EGNN post-process to populate egnn_pred / target_value...")
            task = get_task(f"qm9_edm_cond_{prop}")
            dataset = load_dataset_split(task.config.dataset)
            task.post_process(predictions, dataset, pred_dir)

        # Gather first XYZ per sample, then run self-oracle predictions.
        flat_xyzs = []
        flat_locs = []  # (res_idx, sample_idx) per flat slot
        for r_idx, res in enumerate(predictions):
            for s_idx, sample in enumerate(res.get("samples", [])):
                xyz = first_xyz(sample.get("prediction"))
                flat_xyzs.append(xyz)
                flat_locs.append((r_idx, s_idx))

        if is_shard_worker:
            shard_xyzs = flat_xyzs[args.shard_id::args.num_shards]
            shard_locs = flat_locs[args.shard_id::args.num_shards]
            n_nonempty = sum(1 for x in shard_xyzs if x is not None)
            print(
                f"  [{prop}] shard {args.shard_id}/{args.num_shards}: "
                f"{n_nonempty}/{len(shard_xyzs)} non-empty samples"
            )
            t0 = time.time()
            shard_preds = run_self_oracle(
                gen_model, prop, shard_xyzs,
                batch_size=args.self_batch_size, device=args.device,
            )
            print(f"  [{prop}] shard {args.shard_id} took {time.time() - t0:.1f}s")
            chunk_path = output_dir / f"{prop}_shard{args.shard_id}.pt"
            torch.save(
                {"locs": shard_locs, "preds": shard_preds, "shard_id": args.shard_id},
                chunk_path,
            )
            print(f"  [{prop}] wrote {chunk_path}")
            continue  # metrics are written by the merge step

        if args.merge_only:
            chunks = sorted(output_dir.glob(f"{prop}_shard*.pt"))
            if not chunks:
                raise SystemExit(f"No chunks for {prop} in {output_dir}")
            print(f"  [{prop}] merging {len(chunks)} chunks...")
            for cp in chunks:
                ch = torch.load(cp, weights_only=False)
                for (r_idx, s_idx), pred in zip(ch["locs"], ch["preds"]):
                    predictions[r_idx]["samples"][s_idx]["self_pred"] = pred
        else:
            print(
                f"  [{prop}] running self-oracle on "
                f"{sum(1 for x in flat_xyzs if x is not None)}/{len(flat_xyzs)} non-empty samples..."
            )
            t0 = time.time()
            flat_self_preds = run_self_oracle(
                gen_model, prop, flat_xyzs,
                batch_size=args.self_batch_size, device=args.device,
            )
            print(f"  [{prop}] self-oracle took {time.time() - t0:.1f}s")
            for slot_i, (r_idx, s_idx) in enumerate(flat_locs):
                predictions[r_idx]["samples"][s_idx]["self_pred"] = flat_self_preds[slot_i]

        save_predictions(pred_dir, predictions)

        metrics = compute_metrics(prop, predictions)
        metrics["duration_seconds"] = time.time() - prop_t0
        metrics["regenerated"] = regenerated

        out_path = output_dir / f"{prop}.json"
        out_path.write_text(json.dumps({
            "property": prop,
            "checkpoint": args.checkpoint,
            "run_suffix": args.run_suffix,
            "predictions_dir": pred_dir,
            "metrics": metrics,
        }, indent=2))
        print(f"  [{prop}] wrote {out_path}")

        # Pretty-print key numbers.
        unit = PROP_UNIT[prop]
        print(f"  [{prop}] EGNN  mae@1 = {metrics[f'egnn_mae_at1_{unit}']:.4f} {unit}")
        print(f"  [{prop}] EGNN  mae@5 = {metrics[f'egnn_mae_at5_oracle_{unit}']:.4f} {unit}")
        print(f"  [{prop}] self  mae@1 = {metrics[f'self_mae_at1_{unit}']:.4f} {unit}")
        print(f"  [{prop}] self  mae@5 = {metrics[f'self_mae_at5_oracle_{unit}']:.4f} {unit}")
        print(f"  [{prop}] self->EGNN mae@5 = {metrics[f'self_selected_egnn_mae_at5_{unit}']:.4f} {unit}")

        summary["props"][prop] = metrics

    if is_shard_worker:
        return
    summary["duration_seconds"] = time.time() - overall_t0
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")


if __name__ == "__main__":
    main()
