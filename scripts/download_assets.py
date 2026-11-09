# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch the Qwen3-0.6B initialization and, optionally, the QM9 property evaluators."""
import argparse
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nahual.paths import qwen3_dirs,asset_dir
REPO=os.environ.get('NAHUAL_DATA_REPO') or 'the-matter-lab/nahual'
# QM9 files and the EGNN property oracles of Bao et al. (2023) used by the QM9 evaluations.
QM9_FILES=('gdb9.sdf','gdb9.sdf.csv','uncharacterized.txt')
PROPERTIES=('alpha','gap','homo','lumo','mu','Cv')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--qm9-evaluator',action='store_true');a=p.parse_args()
    from huggingface_hub import hf_hub_download, snapshot_download
    snapshot_download('Qwen/Qwen3-0.6B',revision='c1899de289a04d12100db370d81485cdf75e47ca',local_dir=qwen3_dirs['0.6B'])
    if not a.qm9_evaluator:return
    for prop in PROPERTIES:
        dest=asset_dir/'egnn_oracles'/prop;dest.mkdir(parents=True,exist_ok=True)
        for name in ('args.pickle','best_checkpoint.npy'):
            if not (dest/name).exists():
                shutil.copyfile(hf_hub_download(REPO,f'assets/egnn_oracles/{prop}/{name}'),dest/name)
    asset_dir.mkdir(parents=True,exist_ok=True)
    for name in QM9_FILES:
        if not (asset_dir/name).exists():
            shutil.copyfile(hf_hub_download(REPO,f'assets/qm9/{name}'),asset_dir/name)

if __name__=='__main__':main()
