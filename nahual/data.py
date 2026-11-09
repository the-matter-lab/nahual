# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from nahual.tokens import has_coords
from nahual.paths import processed_dir

def random_rotate_translate(coords):
    # Rotate
    Q = Rotation.random().as_matrix()
    translation = np.random.randn(3).astype(coords.dtype)

    return coords @ Q.astype(coords.dtype) + translation

class SeqDataset(Dataset):
    def __init__(self, name, random_rotate=True, max_len=4096):
        self.name = name
        # Random rotation and translation of coords. Disable when coords are not 3D positions.
        self.random_rotate = random_rotate

        self.tokens = np.load(processed_dir / f"{name}/tokens.npy", mmap_mode="r")
        self.coords = np.load(processed_dir / f"{name}/coords.npy", mmap_mode="r")
        self.sizes = np.load(processed_dir / f"{name}/sizes.npy", mmap_mode="r")
        prompt_sizes_path = processed_dir / f"{name}/prompt_sizes.npy"
        self.prompt_sizes = np.load(prompt_sizes_path, mmap_mode="r") if prompt_sizes_path.exists() else None
        self.has_prompt_sizes = self.prompt_sizes is not None

        self.starts = np.zeros_like(self.sizes)
        self.starts[1:] = np.cumsum(self.sizes[:-1])
        self.indices = np.flatnonzero((self.sizes >= 2) & (self.sizes <= max_len))
        if len(self.indices) == 0:
            raise ValueError(f"No sequences of length 2..{max_len} in {name}")

    def __getitem__(self, i):
        i = self.indices[i]
        tokens, coords = self.get(i)
        if self.random_rotate:
            coords = random_rotate_translate(coords)
        mask = has_coords(tokens)[:, None]
        coords = np.where(mask, coords, 0)
        prompt_size = int(self.prompt_sizes[i]) if self.prompt_sizes is not None else 0
        return tokens, coords, prompt_size

    def get(self, i):
        start = self.starts[i]
        end = start + self.sizes[i]
        tokens = self.tokens[start:end]
        coords = self.coords[start:end].astype(np.float32)

        return tokens, coords

    def __len__(self):
        return len(self.indices)

class ChunkBuilder(Dataset):
    def __init__(self, dataset, chunk_size, shuffle=True, max_len=4096):
        # takes in a dataset whose __getitem__ returns a single sequence
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.max_len = max_len

        self.num_examples = len(dataset)

        # if shuffle, create a global permutation
        if shuffle:
            self.perm = np.random.default_rng(0).permutation(self.num_examples)
        else:
            self.perm = np.arange(self.num_examples)

    def __getitem__(self, i):

        target_length = self.chunk_size + 1  # one extra token for the shifted target

        tokens_list = []
        coords_list = []
        target_loss_mask_list = []
        document_id_list = []
        document_id = 0

        total_len = 0
        offset = 0  # how many sequences ahead of `i` we have consumed

        # Keep concatenating sequences until we have at least `target_length` tokens
        while total_len < target_length:
            # Compute the index in the *permuted* order and wrap around if needed
            permuted_idx = self.perm[(i + offset) % self.num_examples]

            item = self.dataset[permuted_idx]
            if len(item) == 3:
                tokens, coords, prompt_size = item
            else:
                tokens, coords = item
                prompt_size = 0

            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
                coords = coords[:self.max_len]
                prompt_size = min(prompt_size, len(tokens))

            tokens_list.append(tokens)
            coords_list.append(coords)
            response_mask = (np.arange(len(tokens), dtype=np.int32) >= prompt_size)
            target_loss_mask_list.append(response_mask)
            document_id_list.append(np.ones(len(tokens)) * document_id)
            document_id += 1

            total_len += len(tokens)
            offset += 1

            # Safety check to avoid an infinite loop in case of ill-configured datasets
            if offset > self.num_examples + 1:
                raise RuntimeError("Unable to construct chunk - individual sequences may be too short or corrupted.")

        # Concatenate and trim to the exact required length
        tokens = np.concatenate(tokens_list)[:target_length]
        coords = np.concatenate(coords_list, axis=0)[:target_length]
        response_mask = np.concatenate(target_loss_mask_list)[:target_length]
        all_document_ids = np.concatenate(document_id_list, axis=0)[:target_length]
        document_ids = all_document_ids[:-1]
        target_loss_mask = response_mask[1:target_length] & (all_document_ids[1:] == document_ids)

        idx = np.arange(len(tokens)-1, dtype=np.int32)

        # Inputs and next-token targets
        return (
            idx,
            tokens[:-1].astype(np.int32),
            coords[:-1],
            tokens[1:].astype(np.int64),
            coords[1:],
            document_ids.astype(np.int32),
            target_loss_mask.astype(np.bool_)
        )

    def len(self):
        return len(self.dataset)

    def __len__(self):
        return 10_000_000

class WeightedDataset(Dataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights
        self.scaled_weights = [weight * len(ds) for ds, weight in zip(datasets, weights)]

        w = np.array(self.scaled_weights, dtype=np.float32)
        self.probabilities = w / w.sum()

        # Nominal length. Items are sampled by weight, not by index.
        self.total_len = sum(len(d) for d in datasets)

    def _get_rng(self):
        rng = getattr(self, "_rng", None)
        if rng is None:
            seed = torch.initial_seed() % (2 ** 32)
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __len__(self):
        return self.total_len

    def __getitem__(self, i):
        rng = self._get_rng()
        dataset_idx = rng.choice(len(self.datasets), p=self.probabilities)

        dataset = self.datasets[dataset_idx]
        return dataset[rng.integers(len(dataset))]

def collate_fn(batch):
    idx, a1, x1, a2, x2, document_ids, target_loss_mask = batch[0]
    idx = torch.from_numpy(idx)
    a1 = torch.from_numpy(a1)
    x1 = torch.from_numpy(x1)
    a2 = torch.from_numpy(a2)
    x2 = torch.from_numpy(x2)
    document_ids = torch.from_numpy(document_ids)
    target_loss_mask = torch.from_numpy(target_loss_mask)
    return idx, a1, x1, a2, x2, document_ids, target_loss_mask
