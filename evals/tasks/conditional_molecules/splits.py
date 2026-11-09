# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import logging
import numpy as np
from typing import Dict

def get_qm9_splits(
    root_dir: str,
    edm_splits: bool,
) -> Dict[str, np.ndarray]:
    """Adapted from EDM's qm9/data/prepare/qm9.py (https://github.com/ehoogeboom/e3_diffusion_for_molecules;
    MIT, Copyright (c) 2022 Emiel Hoogeboom, Victor Garcia Satorras, Clément Vignac)."""

    def is_int(string: str) -> bool:
        try:
            int(string)
            return True
        except:
            return False

    logging.info("Dropping uncharacterized molecules.")
    gdb9_txt_excluded = os.path.join(root_dir, "uncharacterized.txt")
    if not (os.path.exists(gdb9_txt_excluded) and os.path.getsize(gdb9_txt_excluded) > 0):
        raise FileNotFoundError(
            f"uncharacterized.txt not found or empty at {gdb9_txt_excluded}. "
            "Run python scripts/download_assets.py --qm9-evaluator."
        )

    # First, get list of excluded indices.
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0] for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert (
        len(excluded_idxs) == 3054
    ), "There should be exactly 3054 excluded atoms. Found {}".format(
        len(excluded_idxs)
    )

    # Now, create a list of included indices.
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array(sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now, generate random permutations to assign molecules to training/valation/test sets.
    Nmols = Ngdb9 - Nexcluded
    assert Nmols == len(
        included_idxs
    ), "Number of included molecules should be equal to Ngdb9 - Nexcluded. Found {} {}".format(
        Nmols, len(included_idxs)
    )

    Ntrain = 100000
    Ntest = int(0.1 * Nmols)
    Nval = Nmols - (Ntrain + Ntest)

    # Generate random permutation.
    rng = np.random.RandomState(0)
    if edm_splits:
        data_permutation = rng.permutation(Nmols)
    else:
        data_permutation = np.arange(Nmols)

    train, val, test, extra = np.split(
        data_permutation, [Ntrain, Ntrain + Nval, Ntrain + Nval + Ntest]
    )

    assert len(extra) == 0, "Split was inexact {} {} {} {}".format(
        len(train), len(val), len(test), len(extra)
    )

    train = included_idxs[train]
    val = included_idxs[val]
    test = included_idxs[test]

    splits = {"train": train, "val": val, "test": test}

    return splits
