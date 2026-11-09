# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the weighted training mixture and the validation splits named in a data config."""
import json
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import ConcatDataset
from nahual.data import SeqDataset, WeightedDataset
from nahual.paths import processed_dir

def load_dataconfig(path, require_audit=True, rotate=True, max_len=4096):
    spec = json.loads(Path(path).read_text())
    if not spec['train'] or not spec['validation']:
        raise ValueError('Training and validation splits must be explicit and nonempty')
    if set(spec['train']) & set(spec['validation']):
        raise ValueError('Training and validation splits overlap')
    for name in spec['train']:
        if any(x in name.split('_') for x in ('val', 'valid', 'validation', 'test')):
            raise ValueError(f'Held-out split in training: {name}')
    if require_audit:
        from nahual.prepare import verify_audit
        verify_audit(spec, processed_dir)
    train = WeightedDataset([SeqDataset(n, random_rotate=rotate, max_len=max_len) for n in spec['train']], list(spec['train'].values()))
    val = ConcatDataset([SeqDataset(n, random_rotate=False, max_len=max_len) for n in spec['validation']])
    return SimpleNamespace(dset=train, val_dset=val)
