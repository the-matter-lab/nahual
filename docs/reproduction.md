# Reproducing the paper runs

## The four final runs

| Hub checkpoint | Config | Seed |
|---|---|---|
| `checkpoints/nahual/seed0` | `configs/next_token_seed0.json` | 0 |
| `checkpoints/nahual/seed1` | `configs/next_token_seed1.json` | 1 |
| `checkpoints/diffmlp/seed0` | `configs/diffmlp_seed0.json` | 0 |
| `checkpoints/diffmlp/seed1` | `configs/diffmlp_seed1.json` | 1 |

The next-token configs are the configurations saved in those runs' final
checkpoints. Current runs use varlen attention (`torch.nn.attention.varlen`), though
earlier runs used FlexAttention. The DiffMLP configs are derived from the next-token
configs and keep every training setting. Only operational switches differ from the
runs: Weights & Biases logging and asynchronous evaluation during training are off.
They do not affect training.

All four runs used 32 GB200 GPUs, 24,480 tokens per GPU per step, 115,000 steps (90.1B
tokens), AdamW at 4e-4 with betas (0.9, 0.95) and weight decay 1e-5, 5,000 warmup
steps, a constant rate through step 110,000, and 5,000 linear-decay steps. The loss
covers response tokens only, and FP32 EMA uses decay 0.999. The two architectures
differ only in the denoiser. DiffMLP uses a per-token 12-block, 2,048-wide AdaLN MLP
instead of the coordinate transformer. The comparison matches token exposure, not
parameter count or FLOPs. The same seed does not imply identical
architecture-dependent random draws.

With fewer GPUs, `--grad_accum_steps` keeps the global batch
(GPUs × `grad_accum_steps` × `chunk_size` tokens). See the README.

The restart policy restores the model, EMA, optimizer, scheduler, step and token
count. It does not restore the exact data-loader position or every RNG state. The
paper runs were trained in several time-limited segments under this policy, so they
are reproducible recipes, not bitwise-identical continuations.

## Software versions

Versions used for the paper's training and evaluation:

| Software | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.13.0a0 (NGC 26.07 build), CUDA 13.3, Triton 3.7.1 |
| NumPy, SciPy | 2.1.0, 1.18.0 |
| transformers, datasets, huggingface_hub, safetensors | 5.15.1, 5.0.0, 1.24.0, 0.8.0 |
| RDKit | 2026.3.1 |
| pymatgen, SMACT, ASE | 2026.5.4, 4.0.0, 3.29.0 |
| xtb | 6.7.1 |
| CREST | 3.0.2 (QCG-10 references and the noncovalent-dimer baseline) |
| Qwen3-0.6B | revision `c1899de2` (`scripts/download_assets.py`) |

## Data

The `data/` folder of the Hugging Face repo contains data splits for training
(`configs/datamix.json`) and evaluation. Its dataset card lists each source
and its license. `python -m nahual.prepare download` checks every shard against the
SHA256 hashes in its `manifest.json`.

## Data handling in the code

- The vocabulary is Qwen3's original vocabulary plus 127 atom and lattice tokens.
- Examples longer than 4,096 tokens are excluded, not truncated.
- Prompt tokens and targets that cross a packed document boundary are masked out
  of the loss. An example whose response is empty still gives a finite loss.

## Validation

- The CPU unit tests check that both denoisers are causal, that empty responses
  give finite losses, and that the cached and full trunk forward passes agree.
  They also cover the evaluation code.
- With PyTorch 2.13, compiled varlen attention with GQA agrees with SDPA, and
  varlen agrees with FlexAttention on both full models, forward and backward.
- All six EGNN property oracles load with their QM9 normalization data.
- Preprocessing from scratch passes `python -m nahual.prepare audit`.
- All 27 registered evaluation tasks run on real examples through their metrics.
- On four GB200 GPUs with 24,480-token chunks, training ran at about 237,000
  tokens/s for next-token diffusion and 232,000 tokens/s for DiffMLP, using about
  150 GB of memory per GPU.
- Released checkpoints were trained for 115,000 steps.
