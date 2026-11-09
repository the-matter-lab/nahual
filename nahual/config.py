# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, fields

@dataclass
class Config:
    # these defaults are Qwen3-0.6B

    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    model_type: str = "qwen3"
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_scaling: float = 1.0
    rope_theta: float = 1000000
    sliding_window: bool = False
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.51.0"
    use_cache: bool = True
    use_sliding_window: bool = False
    lm_vocab_size: int = 151669
    # GEN/END, 118 atom types, and 7 lattice tokens.
    atom_vocab_size: int = 2 + 118 + 7
    vocab_size: int = 151936

    qwen3_size: str = "0.6B"
    model_impl: str = "next_token"
    num_coord_transformer_layers: int = 24

    sigma_data: float = 1.5
    P_mean: float = -1.2
    P_std: float = 1.2
    diff_w: int = 2048
    diff_d: int = 12
    diff_mlp_expand: int = 1
    diff_mult: int = 4
    coord_fourier: int = 512
    coord_bandwidth: float = 20
    # "flex": FlexAttention with a document block mask.
    # "varlen": torch.nn.attention.varlen.varlen_attn. Needs torch >= 2.13, else falls back to "flex".
    attention_backend: str = "varlen"

    # General settings
    devices: int = 1
    num_nodes: int = 1
    seed: int = 0
    name: str = "try"
    dataconfig: str = "configs/datamix.json"
    resume_path: str = ""
    state_dict_path: str = ""
    freeze: bool = False

    # Data settings
    num_workers: int = 20
    val_num_workers: int = 2
    prefetch_factor: int = 4
    dataloader_timeout: int = 300
    chunk_size: int = 24480
    max_len: int = 4096
    # Packed chunks per GPU accumulated per optimizer step.
    # Global batch = GPUs x grad_accum_steps x chunk_size tokens. The paper used
    # 32 GPUs x 1 x 24,480. 4 GPUs x 8 gives the same global batch.
    grad_accum_steps: int = 1

    # Training settings
    diff_steps: int = 18
    adam_lr: float = 4e-4
    beta1: float = 0.9
    beta2: float = 0.95
    wd: float = 1e-5
    warmup_steps: int = 5_000
    grad_clip: float = 0.0
    diffusion_loss_weight: float = 1.0
    ema_decay: float = 0.999
    ema_dtype: str = "fp32"
    max_steps: int = 115_000
    decay_steps: int = 5_000
    val_check_interval: int = 0
    limit_val_batches: int = 50

    # Augmentation settings
    rotate: bool = True
    response_loss_only: bool = True

    # Checkpoint settings
    checkpoint_interval: int = 5_000
    first_checkpoint_step: int = 100
    # Seconds before the deadline in NAHUAL_STOP_AT (Unix time), if set, at which
    # training saves a checkpoint and exits.
    stop_margin_seconds: int = 600
    save_ckpt: bool = True

    # Weights & Biases logging (optional extra: pip install -e '.[logging]').
    # Off by default; an empty entity uses your W&B default entity.
    wandb: bool = False
    offline: bool = False
    entity: str = ""

    # Logging
    log_interval: int = 10
    print_interval: int = 100

    async_eval: bool = False
    # Optional fixed-subset request template, prepared on CPU before training.
    eval_template: str = ""
    compile: bool = True
    require_data_audit: bool = True
    output_root: str = "logs/nahual"

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})
