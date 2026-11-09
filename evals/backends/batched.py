# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched generation loop: tokenize, left-pad, model.generate(), parse."""

import json
import os
import signal
import time

import numpy as np
import torch
from tqdm import tqdm

from evals.artifacts import (
    is_generation_complete,
    load_generate_meta_dict,
    load_predictions,
    save_generate_meta,
    save_predictions,
)
from nahual.tokens import XYZ, tokenizer, parse_seq, tokenize_seq, make_seq
from evals.backends.checkpoint import CHAT_BEFORE, CHAT_AFTER


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return x



def _install_stop_handlers():
    stop_requested = {"value": False}
    previous = {}

    def _handler(signum, frame):
        if not stop_requested["value"]:
            print(f"\nReceived signal {signum}; checkpointing after current batch...", flush=True)
        stop_requested["value"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, _handler)

    if hasattr(signal, "SIGUSR1"):
        previous[signal.SIGUSR1] = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, _handler)

    return stop_requested, previous


def _restore_stop_handlers(previous):
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def generate_predictions(model, dataset, config, args, task=None):
    """Generate config.samples_per_prompt samples for each selected prompt.

    Prompts are args.indices, or the range given by args.offset and num_examples.
    If task has build_forced_response(prompt_segs, ground_truth_segs) returning a
    1-D int64 array, those tokens are teacher-forced and only coordinates are
    sampled. With args.save_path set, progress is saved after every batch and a
    rerun resumes.

    Returns one dict per prompt with prompt_index, prompt, ground_truth and
    samples, where each sample is {'prediction': list[str | XYZ]}.
    """
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

    samples_per_prompt = config.samples_per_prompt
    prompt_batch_size = config.prompt_batch_size
    max_new_tokens = config.max_new_tokens
    num_steps = config.num_steps
    device = getattr(args, 'device', 'cuda')
    save_path = getattr(args, 'save_path', None)

    if indices_arg:
        print(f"Evaluating on {len(all_indices)} explicit indices...")
    else:
        print(f"Evaluating on {len(all_indices)} examples (offset={getattr(args, 'offset', 0)})...")
    print(f"Prompt Batch Size: {prompt_batch_size}, Samples per Prompt: {samples_per_prompt}")

    results_by_index = {}
    resume_duration = 0.0
    expected_total = len(all_indices)

    if save_path and os.path.exists(os.path.join(save_path, "predictions.pt")):
        meta = load_generate_meta_dict(save_path)
        resume_duration = float(meta.get("duration_seconds", 0.0))
        try:
            saved_results = load_predictions(save_path)
            saved_by_index = {
                res["prompt_index"]: res
                for res in saved_results
                if res.get("prompt_index") in set(all_indices)
            }
            if saved_by_index:
                results_by_index.update(saved_by_index)
                print(f"Resuming shard from {len(saved_by_index)}/{expected_total} saved prompts...")
        except Exception as e:
            print(f"Warning: could not load partial predictions from {save_path}: {e}")

        if is_generation_complete(save_path, expected_num_predictions=expected_total):
            if all(idx in results_by_index for idx in all_indices):
                print(f"Found completed generation shard at {save_path}, skipping.")
                return [results_by_index[idx] for idx in all_indices]

    all_indices = [idx for idx in all_indices if idx not in results_by_index]
    # Sorting by prompt length reduces left padding. Output order is unchanged.
    prompt_lengths = {}
    if config.extra.get('batch_by_length', False):
        for idx in all_indices:
            instruction = dataset[idx]['instruction']
            if isinstance(instruction, str):
                instruction = json.loads(instruction)
            prompt_lengths[idx] = len(tokenize_seq(make_seq(instruction))[0])
        all_indices.sort(key=lambda idx: (prompt_lengths[idx], idx))
    if not all_indices:
        if save_path and results_by_index:
            saved_predictions = [results_by_index[idx] for idx in sorted(results_by_index)]
            save_predictions(save_path, saved_predictions)
            save_generate_meta(
                save_path,
                resume_duration,
                complete=True,
                num_predictions=len(saved_predictions),
                total_expected=expected_total,
            )
            return [results_by_index[idx] for idx in sorted(results_by_index)]
        return []

    stop_requested, previous_handlers = _install_stop_handlers()
    loop_start = time.time()

    def _checkpoint(complete):
        if save_path is None:
            return
        ordered_predictions = [results_by_index[idx] for idx in sorted(results_by_index)]
        elapsed = resume_duration + (time.time() - loop_start)
        peak_gpu_memory_gb = None
        if torch.cuda.is_available():
            peak_gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        save_predictions(save_path, ordered_predictions)
        save_generate_meta(
            save_path,
            elapsed,
            peak_gpu_memory_gb=peak_gpu_memory_gb,
            complete=complete,
            num_predictions=len(ordered_predictions),
            total_expected=expected_total,
        )

    try:
        for i in tqdm(range(0, len(all_indices), prompt_batch_size)):
            batch_indices = all_indices[i : i + prompt_batch_size]

            prompts = []
            ground_truths = []

            for idx in batch_indices:
                row = dataset[idx]
                instruction = row["instruction"]
                response = row["response"]
                if isinstance(instruction, str):
                    instruction = json.loads(instruction)
                if isinstance(response, str):
                    response = json.loads(response)

                prompt_segs = make_seq(instruction)
                response_segs = make_seq(response)
                prompts.append(prompt_segs)
                ground_truths.append(response_segs)

            # Optional teacher forcing of response tokens. Coordinates are still sampled.
            forced_per_prompt = None
            if task is not None and hasattr(task, "build_forced_response"):
                forced_per_prompt = [
                    np.asarray(task.build_forced_response(p, gt), dtype=np.int64)
                    for p, gt in zip(prompts, ground_truths)
                ]

            # Tokenize and batch
            seqs_a, seqs_x, lens = [], [], []
            forced_seqs, forced_lens_list = [], []
            for p_idx_local, p in enumerate(prompts):
                sa, sx = tokenize_seq(p)
                sa = np.concatenate([CHAT_BEFORE, sa, CHAT_AFTER], axis=0)
                sx = np.concatenate([
                    np.zeros((len(CHAT_BEFORE), 3), dtype=np.float32),
                    sx,
                    np.zeros((len(CHAT_AFTER), 3), dtype=np.float32),
                ], axis=0)
                for _ in range(samples_per_prompt):
                    seqs_a.append(sa)
                    seqs_x.append(sx)
                    lens.append(len(sa))
                    if forced_per_prompt is not None:
                        forced_seqs.append(forced_per_prompt[p_idx_local])
                        forced_lens_list.append(len(forced_per_prompt[p_idx_local]))

            # Left-pad batch
            max_len = max(lens)
            total_bsz = len(seqs_a)
            pad_id = tokenizer.pad_token_id or 0

            batch_tokens = np.full((total_bsz, max_len), pad_id, dtype=np.int64)
            batch_coords = np.zeros((total_bsz, max_len, 3), dtype=np.float32)
            pad_mask = np.zeros((total_bsz, max_len), dtype=bool)

            for b in range(total_bsz):
                l = lens[b]
                off = max_len - l
                batch_tokens[b, off:] = seqs_a[b]
                batch_coords[b, off:] = seqs_x[b]
                pad_mask[b, off:] = 1

            batch_tokens_t = torch.tensor(batch_tokens).to(device)
            batch_coords_t = torch.tensor(batch_coords).to(device)
            pad_mask_t = torch.tensor(pad_mask).to(device)

            gen_kwargs = {}
            temp = getattr(args, "temperature", None)
            if temp is None:
                temp = getattr(config, "default_temperature", None)
            if temp is not None:
                gen_kwargs["temperature"] = float(temp)

            if forced_seqs:
                max_force_len = max(forced_lens_list)
                forced_arr = np.zeros((total_bsz, max_force_len), dtype=np.int64)
                for b, fseq in enumerate(forced_seqs):
                    forced_arr[b, :len(fseq)] = fseq
                gen_kwargs["forced_tokens"] = torch.tensor(forced_arr, device=device)
                gen_kwargs["forced_lens"] = torch.tensor(forced_lens_list, dtype=torch.long, device=device)

            with torch.no_grad():
                out = model.generate(
                    prefix=(batch_tokens_t, batch_coords_t),
                    max_len=max_new_tokens + max_len,
                    num_steps=num_steps,
                    keep_prefix=False,
                    pad_mask=pad_mask_t,
                    pbar=False,
                    **gen_kwargs,
                )

            batch_out_tokens, batch_out_coords = out
            # One host transfer per batch, not one per sample.
            batch_out_tokens = _to_numpy(batch_out_tokens)
            batch_out_coords = _to_numpy(batch_out_coords)

            # Parse outputs into segment sequences.
            current_idx = 0
            for p_idx, gt in enumerate(ground_truths):
                prompt_result = {
                    "prompt_index": batch_indices[p_idx],
                    "prompt": prompts[p_idx],
                    "ground_truth": gt,
                    "samples": [],
                }

                for _ in range(samples_per_prompt):
                    pred_toks = batch_out_tokens[current_idx]
                    pred_crd = batch_out_coords[current_idx]
                    # Truncate to the forced length. Rows that finish early keep
                    # emitting tokens until the longest row is done.
                    if forced_lens_list:
                        flen = forced_lens_list[current_idx]
                        pred_toks = pred_toks[:flen]
                        pred_crd = pred_crd[:flen]
                    current_idx += 1

                    pred_seq = parse_seq(pred_toks, pred_crd)
                    if pred_seq is None:
                        pred_seq = []

                    prompt_result["samples"].append({
                        "prediction": pred_seq,
                    })

                results_by_index[prompt_result["prompt_index"]] = prompt_result

            _checkpoint(complete=False)
            if stop_requested["value"]:
                raise RuntimeError("Generation interrupted after checkpoint")
    finally:
        _restore_stop_handlers(previous_handlers)

    ordered_results = [results_by_index[idx] for idx in sorted(results_by_index)]
    _checkpoint(complete=True)
    return ordered_results
