# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard library imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datetime
import argparse
import contextlib
import copy
import glob
import signal
import sys
import faulthandler
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path

# Third-party imports
import torch
torch.set_float32_matmul_precision('medium')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast

from nahual.config import Config
from nahual.model import NextTokenDiffusion, DiffusionMLP
from nahual.data import ChunkBuilder, SeqDataset, collate_fn
from nahual.paths import qwen3_dirs

# -------------------- Helpers --------------------

def is_rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank_zero(*args, **kwargs):
    if is_rank_zero():
        print(*args, **kwargs)


class ProgressWatchdog:
    """Per-rank stall detector.

    If the main loop stays in one phase too long, prints the rank, phase and step
    and dumps every thread's stack. The NCCL watchdog only reports the waiting
    ranks, not the one that stalled.
    """

    # Warn before the 10-minute NCCL timeout in setup_ddp(). Step 0 gets longer
    # because torch.compile runs there.
    def __init__(self, rank, warn_after_s=240.0, first_step_grace_s=900.0, poll_s=15.0):
        self.rank = rank
        self.warn_after_s = warn_after_s
        self.first_step_grace_s = first_step_grace_s
        self.poll_s = poll_s
        self._phase = "init"
        self._step = -1
        self._since = time.monotonic()
        self._warned_for = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="progress-watchdog", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def set_phase(self, phase, step=None):
        self._phase = phase
        if step is not None:
            self._step = step
        self._since = time.monotonic()
        self._warned_for = None

    def _threshold(self):
        # The first forward triggers compilation of the whole model; a few
        # minutes there is normal, not a stall.
        if self._step <= 0 and self._phase in ("forward", "init"):
            return self.first_step_grace_s
        return self.warn_after_s

    def _run(self):
        while not self._stop.wait(self.poll_s):
            stuck_for = time.monotonic() - self._since
            threshold = self._threshold()
            if stuck_for < threshold:
                continue
            # Re-warn once per threshold interval while the rank stays stuck.
            key = (self._phase, self._step, int(stuck_for // threshold))
            if key == self._warned_for:
                continue
            self._warned_for = key
            print(
                f"[STALL] rank={self.rank} phase={self._phase} step={self._step} "
                f"stuck_for={stuck_for:.0f}s -- dumping all thread stacks",
                flush=True,
            )
            faulthandler.dump_traceback()
            sys.stdout.flush()
            sys.stderr.flush()

def setup_ddp():
    """Initialize distributed training from the environment variables set by torchrun."""
    # Short timeout so a dead rank fails the job fast. A restart resumes from the last checkpoint.
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=10))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_model(config):
    classes = {"next_token": NextTokenDiffusion, "diffmlp": DiffusionMLP}
    if config.model_impl not in classes:
        raise ValueError(f"Unsupported model: {config.model_impl}")
    return classes[config.model_impl](config)


def iter_leaf_datasets(dataset):
    if hasattr(dataset, "datasets"):
        for child in dataset.datasets:
            yield from iter_leaf_datasets(child)
        return
    if hasattr(dataset, "dataset"):
        yield from iter_leaf_datasets(dataset.dataset)
        return
    yield dataset


def validate_response_loss_metadata(dataset, split_name: str):
    missing = []
    for leaf in iter_leaf_datasets(dataset):
        if isinstance(leaf, SeqDataset) and not getattr(leaf, "has_prompt_sizes", False):
            missing.append(leaf.name)
    if missing:
        missing_str = ", ".join(sorted(set(missing)))
        raise FileNotFoundError(
            f"response_loss_only=True requires prompt_sizes.npy for every sequence dataset in {split_name}. "
            f"Missing metadata for: {missing_str}"
        )


def build_optimizer(model, config: Config):
    wd_params = []
    no_wd_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_wd_params.append(param)
        else:
            wd_params.append(param)

    param_groups = []
    if wd_params:
        param_groups.append({"params": wd_params, "weight_decay": config.wd})
    if no_wd_params:
        param_groups.append({"params": no_wd_params, "weight_decay": 0.0})

    return torch.optim.AdamW(
        param_groups,
        betas=(config.beta1, config.beta2),
        lr=config.adam_lr,
        eps=1e-15,
        fused=config.grad_clip == 0.0,
    )


def build_scheduler(optimizer, config: Config):
    def schedule_multiplier(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        if config.decay_steps > 0:
            decay_start = config.max_steps - config.decay_steps
            if step >= decay_start:
                return max(0.0, 1.0 - (step - decay_start) / config.decay_steps)
        return 1.0

    lr_lambdas = [schedule_multiplier for _ in optimizer.param_groups]
    return LambdaLR(optimizer, lr_lambdas)


def compute_grad_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.detach().data.norm(2).item() ** 2
    return total_norm ** 0.5


def resolve_ema_dtype(value: str) -> torch.dtype:
    key = value.strip().lower()
    if key in {"fp32", "float32", "f32"}:
        return torch.float32
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported ema_dtype={value!r}; expected fp32 or bf16")


class LocalEMA:
    @torch.no_grad()
    def __init__(self, net, decay: float, device: torch.device, dtype: torch.dtype):
        self.net = net
        self.decay = decay
        self.ema = copy.deepcopy(net).to(device=device, dtype=dtype)
        self.ema.eval()
        for param in self.ema.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _copy_buffers(self):
        for src, dst in zip(self.net.buffers(), self.ema.buffers()):
            if dst.dtype.is_floating_point:
                dst.copy_(src.detach().to(device=dst.device, dtype=dst.dtype))
            else:
                dst.copy_(src.detach().to(device=dst.device))

    @torch.no_grad()
    def reset(self):
        for src, dst in zip(self.net.parameters(), self.ema.parameters()):
            dst.copy_(src.detach().to(device=dst.device, dtype=dst.dtype))
        self._copy_buffers()

    @torch.no_grad()
    def update(self):
        for src, dst in zip(self.net.parameters(), self.ema.parameters()):
            dst.lerp_(src.detach().to(device=dst.device, dtype=dst.dtype), 1.0 - self.decay)
        self._copy_buffers()

    def state_dict(self):
        self._copy_buffers()
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)


# -------------------- Checkpointing --------------------

def get_most_recent_checkpoint(checkpoint_dir):
    ckpts = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


def save_checkpoint(checkpoint_dir, model, ema, optimizer, scheduler, step, total_tokens, config):
    if not is_rank_zero():
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Write to a temp file, then rename atomically.
    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    tmp_path = path + ".tmp"
    torch.save({
        "step": step,
        "total_tokens": total_tokens,
        "model_state_dict": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(config),
    }, tmp_path)
    os.rename(tmp_path, path)
    # Update "last" symlink
    last_link = os.path.join(checkpoint_dir, "last.pt")
    tmp_link = last_link + ".tmp"
    if os.path.islink(tmp_link) or os.path.exists(tmp_link):
        os.remove(tmp_link)
    os.symlink(os.path.basename(path), tmp_link)
    os.rename(tmp_link, last_link)
    print_rank_zero(f"Saved checkpoint: {path}")


def load_checkpoint(path, model, ema, optimizer, scheduler, device):
    print_rank_zero(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model_state_dict"])
    if "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt["step"]
    total_tokens = int(ckpt.get("total_tokens", 0))
    print_rank_zero(f"Resumed from step {step} (total_tokens={total_tokens})")
    return step, total_tokens


# -------------------- Validation --------------------

@torch.no_grad()
def validate(model, val_loader, config, device, step, wandb_run=None):
    """Run validation loop on all ranks."""
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    total_diff_loss = 0.0
    total_ce_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        if n_batches >= config.limit_val_batches:
            break
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        with autocast("cuda", dtype=torch.bfloat16):
            diff_loss, ce_loss = raw_model(*batch)
        total_diff_loss += diff_loss.item()
        total_ce_loss += ce_loss.item()
        n_batches += 1

    if n_batches > 0:
        avg_diff = total_diff_loss / n_batches
        avg_ce = total_ce_loss / n_batches
    else:
        avg_diff = avg_ce = 0.0

    # All-reduce across ranks
    if dist.is_initialized():
        metrics = torch.tensor([avg_diff, avg_ce, float(n_batches)], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        avg_diff = metrics[0].item() / world_size
        avg_ce = metrics[1].item() / world_size

    val_total = config.diffusion_loss_weight * avg_diff + avg_ce
    print_rank_zero(
        f"[Step {step}] Val: diff_loss={avg_diff:.4f}  ce_loss={avg_ce:.4f}  "
        f"total={val_total:.4f} (diff×{config.diffusion_loss_weight:g})"
    )

    if wandb_run and is_rank_zero():
        wandb_run.log({
            "val/diffusion_loss": avg_diff,
            "val/cross_entropy_loss": avg_ce,
            "val/total_loss": val_total,
            "step": step,
        }, step=step)

    raw_model.train()

# -------------------- Training --------------------

def train(config: Config):
    if config.async_eval:
        raise ValueError("async_eval is not supported; evaluate saved checkpoints with `python -m evals run`")
    # --- DDP setup ---
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    print_rank_zero(f"World size: {world_size} (nodes={config.num_nodes}, devices={config.devices})")
    if config.grad_accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {config.grad_accum_steps}")
    print_rank_zero(
        f"Global batch: {world_size} GPUs x {config.grad_accum_steps} micro-batches x "
        f"{config.chunk_size} tokens = {world_size * config.grad_accum_steps * config.chunk_size:,} tokens/step"
    )
    print_rank_zero(config)

    from nahual.mixture import load_dataconfig
    datamix = load_dataconfig(config.dataconfig, config.require_data_audit, config.rotate, config.max_len)

    if config.response_loss_only:
        validate_response_loss_metadata(datamix.dset, "train")
        validate_response_loss_metadata(datamix.val_dset, "val")


    # --- Seeding ---
    # Seed before building the model. DDP broadcasts rank 0's initial weights, and
    # the rank offset decorrelates per-rank noise.
    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed_all(config.seed + rank)

    # --- Model ---
    model = build_model(config)

    if config.state_dict_path:
        print_rank_zero(f"Loading state_dict from {config.state_dict_path}")
        sd = torch.load(config.state_dict_path, map_location="cpu")
        # Training checkpoint (as released): start from its EMA weights.
        if "model_state_dict" in sd:
            sd = sd.get("ema_state_dict") or sd["model_state_dict"]
        model.load_state_dict(sd)
        del sd
        torch.cuda.empty_cache()

    if config.freeze:
        if hasattr(model, 'freeze_llm_weights'):
            model.freeze_llm_weights()
        else:
            model.embed_tokens.weight.requires_grad = False
            model.lm_head.weight.requires_grad = False
            for param in model.layers.parameters():
                param.requires_grad = False
            for param in model.norm.parameters():
                param.requires_grad = False

    if config.qwen3_size != "" and not config.state_dict_path and not config.resume_path:
        print_rank_zero(f"Loading Qwen3 weights from {qwen3_dirs[config.qwen3_size]}")
        model.load_qwen3_weights(Path(qwen3_dirs[config.qwen3_size]))

    # Move model to GPU and wrap with DDP
    model = model.to(device)
    ema = LocalEMA(model, decay=config.ema_decay, device=device, dtype=resolve_ema_dtype(config.ema_dtype))
    ema.reset()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Print param counts
    if is_rank_zero():
        raw = model.module
        # The denoiser is `layers2` (NextTokenDiffusion) or `diff_mlp` (DiffusionMLP).
        diff_module = getattr(raw, "layers2", None)
        if diff_module is None:
            diff_module = getattr(raw, "diff_mlp", None)
        diff_params = sum(p.numel() for p in diff_module.parameters()) if diff_module is not None else 0
        total_params = sum(p.numel() for p in raw.parameters())
        trunk_params = total_params - diff_params
        lm_head_params = sum(p.numel() for p in raw.lm_head.parameters())
        print(f"Total parameters: {total_params / 1e6:.1f}M")
        print(f"Trunk parameters: {trunk_params / 1e6:.1f}M")
        print(f"Diffusion parameters: {diff_params / 1e6:.1f}M")
        print(f"LM head parameters: {lm_head_params / 1e6:.1f}M")

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model.module, config)
    scheduler = build_scheduler(optimizer, config)

    # --- Data ---
    train_dset = ChunkBuilder(datamix.dset, chunk_size=config.chunk_size, max_len=config.max_len)
    val_dset = ChunkBuilder(datamix.val_dset, chunk_size=config.chunk_size, max_len=config.max_len)
    selected_collate_fn = collate_fn

    train_sampler = DistributedSampler(train_dset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed)
    val_sampler = DistributedSampler(val_dset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader_kwargs = {
        "batch_size": 1,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "collate_fn": selected_collate_fn,
    }
    if config.num_workers > 0:
        train_loader_kwargs["timeout"] = config.dataloader_timeout
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = config.prefetch_factor

    val_loader_kwargs = {
        "batch_size": 1,
        "num_workers": config.val_num_workers,
        "pin_memory": True,
        "collate_fn": selected_collate_fn,
    }
    if config.val_num_workers > 0:
        val_loader_kwargs["timeout"] = config.dataloader_timeout
        val_loader_kwargs["persistent_workers"] = False
        val_loader_kwargs["prefetch_factor"] = min(config.prefetch_factor, 2)

    train_loader = DataLoader(
        train_dset,
        sampler=train_sampler,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dset,
        sampler=val_sampler,
        **val_loader_kwargs,
    )

    # --- Checkpoint ---
    project = "nahual"
    checkpoint_dir = f"{config.output_root}/{config.name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # W&B
    wandb_run = None
    if config.wandb and is_rank_zero():
        import wandb
        wandb_run = wandb.init(
            dir="logs",
            project=project,
            entity=config.entity or None,
            name=config.name,
            config=asdict(config),
            mode="offline" if config.offline else "online",
        )

    # Resume
    global_step = 0
    total_tokens = 0
    # The run's own newest checkpoint wins, so restarting a continued run does not go back to resume_path.
    resume_path = get_most_recent_checkpoint(checkpoint_dir) or config.resume_path
    if resume_path and os.path.exists(resume_path):
        global_step, total_tokens = load_checkpoint(resume_path, model, ema, optimizer, scheduler, device)
    # Broadcast step and total_tokens so all ranks agree
    state_tensor = torch.tensor([global_step, total_tokens], device=device, dtype=torch.long)
    dist.broadcast(state_tensor, src=0)
    global_step = int(state_tensor[0].item())
    total_tokens = int(state_tensor[1].item())

    # --- Signal handling for preemption ---
    should_stop = False
    preemption_checkpoint_saved = False

    def checkpoint_on_exit():
        nonlocal preemption_checkpoint_saved
        if preemption_checkpoint_saved or not config.save_ckpt:
            return

        try:
            if dist.is_initialized():
                dist.barrier()
            save_checkpoint(checkpoint_dir, model, ema, optimizer, scheduler, global_step, total_tokens, config)
            if dist.is_initialized():
                dist.barrier()
            if is_rank_zero():
                (Path(checkpoint_dir).parent / "PREEMPTED").write_text(str(global_step) + "\n")
            preemption_checkpoint_saved = True
        except Exception as exc:
            print_rank_zero(f"Checkpoint-on-exit failed: {exc}")

    def _signal_handler(signum, frame):
        nonlocal should_stop
        print_rank_zero(f"Received signal {signum}, saving checkpoint and exiting...")
        should_stop = True

    signal.signal(signal.SIGUSR1, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Optionally stop before a wall-clock deadline so the last checkpoint is saved
    # cleanly. Ranks agree on the stop step through an all-reduce.
    job_end = float(os.environ.get("NAHUAL_STOP_AT", "0"))
    stop_check_interval = 10

    # --- Training loop ---
    model.train()
    print_rank_zero(f"Starting training from step {global_step}")

    watchdog = ProgressWatchdog(rank=dist.get_rank() if dist.is_initialized() else 0).start()

    train_iter = iter(train_loader)
    epoch = 0
    interval_data_time = 0.0
    interval_step_time = 0.0
    interval_tokens = 0.0

    while global_step < config.max_steps:
        if global_step % stop_check_interval == 0:
            deadline = job_end > 0 and time.time() > job_end - config.stop_margin_seconds
            stop_flag = torch.tensor(int(should_stop or deadline), device=device)
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if stop_flag.item():
                print_rank_zero(f"Stopping at step {global_step} (signal={should_stop}, deadline={deadline})")
                checkpoint_on_exit()
                break

        accum = config.grad_accum_steps
        iteration_start = time.perf_counter()
        data_time = 0.0
        optimizer.zero_grad(set_to_none=True)
        loss_sums = None
        for micro_step in range(accum):
            data_start = time.perf_counter()
            watchdog.set_phase("data", global_step)
            # Get next batch, cycling through epochs
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_sampler.set_epoch(epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            data_time += time.perf_counter() - data_start

            # Forward. Only the last micro-batch synchronizes gradients across ranks.
            watchdog.set_phase("forward", global_step)
            sync = contextlib.nullcontext() if micro_step == accum - 1 else model.no_sync()
            with sync:
                with autocast("cuda", dtype=torch.bfloat16):
                    diffusion_loss, cross_entropy_loss = model(*batch)
                    total_loss = (
                        config.diffusion_loss_weight * diffusion_loss + cross_entropy_loss
                    )

                # Backward
                watchdog.set_phase("backward", global_step)
                (total_loss / accum if accum > 1 else total_loss).backward()
            interval_tokens += batch[1].numel()
            if accum > 1:
                losses = torch.stack([total_loss.detach(), diffusion_loss.detach(), cross_entropy_loss.detach()])
                loss_sums = losses if loss_sums is None else loss_sums + losses
        if accum > 1:
            # Logged losses are micro-batch means, matching the accumulated gradient.
            total_loss, diffusion_loss, cross_entropy_loss = (loss_sums / accum).unbind()

        # Grad clip
        watchdog.set_phase("gradnorm", global_step)
        grad_norm = None
        if config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip).item()
        else:
            grad_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None]).item()

        if not torch.isfinite(torch.tensor(grad_norm)):
            raise FloatingPointError(f"Non-finite gradients at step {global_step}")

        # Step
        watchdog.set_phase("optim", global_step)
        optimizer.step()
        scheduler.step()

        next_step = global_step + 1
        ema.update()

        global_step = next_step
        step_time = time.perf_counter() - iteration_start - data_time
        interval_data_time += data_time
        interval_step_time += step_time

        # Logging
        if global_step % config.log_interval == 0:
            watchdog.set_phase("log", global_step)
            lr = scheduler.get_last_lr()[0]
            loss_val = total_loss.item()
            diff_val = diffusion_loss.item()
            ce_val = cross_entropy_loss.item()
            perf_sums = torch.tensor(
                [interval_data_time, interval_step_time, interval_tokens],
                device=device,
                dtype=torch.float64,
            )
            perf_max = torch.tensor(
                [interval_data_time, interval_step_time],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(perf_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(perf_max, op=dist.ReduceOp.MAX)

            total_tokens += int(perf_sums[2].item())

            avg_data_ms = 1000.0 * perf_sums[0].item() / (world_size * config.log_interval)
            avg_step_ms = 1000.0 * perf_sums[1].item() / (world_size * config.log_interval)
            max_step_ms = 1000.0 * perf_max[1].item() / config.log_interval
            global_tokens_per_s = perf_sums[2].item() / max(perf_max[1].item(), 1e-9)
            global_tokens_per_step = perf_sums[2].item() / config.log_interval
            data_fraction = perf_sums[0].item() / max(perf_sums[1].item(), 1e-9)

            if wandb_run:
                wandb_run.log({
                    "train/diffusion_loss": diff_val,
                    "train/cross_entropy_loss": ce_val,
                    "train/total_loss": loss_val,
                    "train/grad_norm": grad_norm,
                    "train/lr": lr,
                    "perf/avg_data_ms": avg_data_ms,
                    "perf/avg_step_ms": avg_step_ms,
                    "perf/max_step_ms": max_step_ms,
                    "perf/global_tokens_per_s": global_tokens_per_s,
                    "perf/global_tokens_per_step": global_tokens_per_step,
                    "perf/data_fraction": data_fraction,
                    "train/num_tokens": total_tokens,
                    "step": global_step,
                }, step=global_step)

            if global_step % config.print_interval == 0:
                print_rank_zero(
                    f"[Step {global_step}/{config.max_steps}] "
                    f"loss={loss_val:.4f} (diff={diff_val:.4f}×{config.diffusion_loss_weight:g} ce={ce_val:.4f}) "
                    f"grad_norm={grad_norm:.4f} lr={lr:.2e} "
                    f"avg_data_ms={avg_data_ms:.1f} avg_step_ms={avg_step_ms:.1f} "
                    f"max_step_ms={max_step_ms:.1f} tok/s={global_tokens_per_s:,.0f} "
                    f"num_tokens={total_tokens:,}"
                )

            interval_data_time = 0.0
            interval_step_time = 0.0
            interval_tokens = 0.0

        # Validation
        if config.val_check_interval > 0 and global_step % config.val_check_interval == 0:
            watchdog.set_phase("validate", global_step)
            validate(model, val_loader, config, device, global_step, wandb_run)
            model.train()

        # Checkpoint
        if config.save_ckpt and (global_step % config.checkpoint_interval == 0 or global_step == config.first_checkpoint_step):
            watchdog.set_phase("checkpoint", global_step)
            save_checkpoint(checkpoint_dir, model, ema, optimizer, scheduler, global_step, total_tokens, config)
            dist.barrier()

    # Final checkpoint
    if config.save_ckpt:
        if not preemption_checkpoint_saved and global_step % config.checkpoint_interval != 0:
            save_checkpoint(checkpoint_dir, model, ema, optimizer, scheduler, global_step, total_tokens, config)
        if dist.is_initialized():
            dist.barrier()

    if wandb_run:
        wandb_run.finish()

    watchdog.stop()
    if is_rank_zero() and global_step >= config.max_steps:
        (Path(checkpoint_dir).parent / "TRAINING_COMPLETE").write_text(str(global_step) + "\n")
    print_rank_zero(f"Training finished at step {global_step}")
    cleanup_ddp()


# -------------------- Main --------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args_from_config() -> Config:
    import json
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args()
    values = json.loads(Path(known.config).read_text()) if known.config else {}
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown training configuration fields: {sorted(unknown)}")
    defaults = Config.from_dict(values)
    parser = argparse.ArgumentParser(parents=[pre])
    for field in fields(Config):
        val = getattr(defaults, field.name)
        parser.add_argument(f"--{field.name}", type=str2bool if isinstance(val, bool) else type(val), default=val)
    args = vars(parser.parse_args())
    args.pop("config")
    return Config(**args)


def main():
    train(parse_args_from_config())


if __name__ == "__main__":
    main()
