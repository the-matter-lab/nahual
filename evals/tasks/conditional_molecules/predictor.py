# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EGNN property oracles for QM9 evaluation, trained by Bao et al. (2023).

The EGNN code is EDM's (see egnn/). The six evaluator checkpoints are in
NAHUAL_ASSET_DIR/egnn_oracles/<property>/ (python scripts/download_assets.py --qm9-evaluator).
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

from evals.tasks.conditional_molecules.egnn.models_property import EGNN

# QM9 atom types H, C, N, O, F to one-hot indices
ATOMIC_NUM_TO_IDX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
NUM_ATOM_TYPES = 5
MAX_N_NODES = 29

# Hartree to meV conversion
HARTREE_TO_MEV = 27211.386

# Stored in Hartree in raw QM9 and in meV in prompts
HARTREE_PROPERTIES = {"gap", "homo", "lumo"}

# CSV column names in gdb9.sdf.csv
PROPERTY_CSV_COLS = {
    "alpha": "alpha",
    "gap": "gap",
    "homo": "homo",
    "lumo": "lumo",
    "mu": "mu",
    "Cv": "cv",
}

# The oracles were trained on thermochemical labels minus a per-atom baseline
# (atomref.txt in the training code of Bao et al.). For Cv it is 2.981
# cal/(mol*K) per atom. Subtract it from val labels before computing mean/MAD
# and add it back to EGNN predictions.
THERMO_ATOMREF_PER_ATOM = {"Cv": 2.981}


def _thermo_baseline(property_name, atoms):
    """Per-molecule atomref baseline subtracted from the oracle training labels. 0.0 for non-thermo properties."""
    per_atom = THERMO_ATOMREF_PER_ATOM.get(property_name)
    if per_atom is None:
        return 0.0
    return per_atom * len(atoms)


def load_egnn_predictor(property_name, device="cuda"):
    """Load the EGNN oracle for one of alpha, gap, homo, lumo, mu, Cv.

    Returns (model, mean, mad), where mean and mad are the normalization stats.
    """
    from nahual.paths import asset_dir
    model_dir = os.path.join(str(asset_dir), "egnn_oracles", property_name)
    args_path = os.path.join(model_dir, "args.pickle")
    ckpt_path = os.path.join(model_dir, "best_checkpoint.npy")

    for path in (args_path, ckpt_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; run python scripts/download_assets.py --qm9-evaluator")

    with open(args_path, "rb") as f:
        args = pickle.load(f)

    args.device = device
    args.model_name = "egnn"
    model = EGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=args.nf, device=args.device,
                 n_layers=args.n_layers, coords_weight=1.0, attention=args.attention, node_attr=args.node_attr)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Compute normalization stats from QM9 validation split
    mean, mad = compute_qm9_property_norms(property_name)
    mean = torch.tensor(mean, dtype=torch.float32, device=device)
    mad = torch.tensor(mad, dtype=torch.float32, device=device)

    return model, mean, mad


_CACHED_VAL_IDXS = None
_CACHED_CSV_DF = None
_CACHED_ATOM_COUNTS = None


def _get_qm9_val_data():
    """QM9 validation indices and property CSV, cached."""
    global _CACHED_VAL_IDXS, _CACHED_CSV_DF

    if _CACHED_VAL_IDXS is None:
        from nahual.paths import asset_dir
        raw_dir = str(asset_dir)
        csv_path = os.path.join(raw_dir, "gdb9.sdf.csv")

        from evals.tasks.conditional_molecules.splits import get_qm9_splits

        splits = get_qm9_splits(raw_dir, edm_splits=True)
        _CACHED_VAL_IDXS = splits["val"]
        _CACHED_CSV_DF = pd.read_csv(csv_path)

    return _CACHED_VAL_IDXS, _CACHED_CSV_DF


def _get_qm9_atom_counts():
    """Per-molecule heavy+H atom count, indexed by GDB9 row. Cached."""
    global _CACHED_ATOM_COUNTS
    if _CACHED_ATOM_COUNTS is None:
        from rdkit import Chem
        from nahual.paths import asset_dir
        raw_dir = str(asset_dir)
        sdf_path = os.path.join(raw_dir, "gdb9.sdf")
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        counts = []
        for mol in supplier:
            counts.append(mol.GetNumAtoms() if mol is not None else 0)
        _CACHED_ATOM_COUNTS = np.array(counts, dtype=np.int32)
    return _CACHED_ATOM_COUNTS


def compute_qm9_property_norms(property_name):
    """Mean and MAD of a property on the QM9 validation split (EDM splits), as used to train the oracles.

    Before the stats, gap/homo/lumo are converted from Hartree to eV
    (see qm9/dataset.py qm9_to_eV) and Cv has its per-atom atomref baseline
    removed (see qm9/data/prepare/qm9.py add_thermo_targets).
    """
    HARTREE_TO_EV = 27.2114

    val_idxs, df = _get_qm9_val_data()
    csv_col = PROPERTY_CSV_COLS[property_name]
    values = df[csv_col].values[val_idxs].astype(np.float64)

    if property_name in HARTREE_PROPERTIES:
        values = values * HARTREE_TO_EV

    if property_name in THERMO_ATOMREF_PER_ATOM:
        n_atoms_val = _get_qm9_atom_counts()[val_idxs].astype(np.float64)
        values = values - THERMO_ATOMREF_PER_ATOM[property_name] * n_atoms_val

    mean = float(np.mean(values))
    mad = float(np.mean(np.abs(values - mean)))

    return mean, mad


def xyz_to_egnn_tensors(atoms_list, coords_list, device="cuda"):
    """Convert lists of atomic numbers and (n_atoms, 3) coords to EGNN input tensors."""
    batch_size = len(atoms_list)
    n_nodes = MAX_N_NODES

    # One-hot encoding with zero padding. Elements outside H, C, N, O, F get an all-zero encoding.
    one_hot = np.zeros((batch_size, n_nodes, NUM_ATOM_TYPES), dtype=np.float32)
    positions = np.zeros((batch_size, n_nodes, 3), dtype=np.float32)
    atom_mask = np.zeros((batch_size, n_nodes, 1), dtype=np.float32)

    for i, (atoms, coords) in enumerate(zip(atoms_list, coords_list)):
        n = len(atoms)
        if not 0 < n <= n_nodes or np.shape(coords) != (n, 3):
            raise ValueError(f'EGNN expects 1..{n_nodes} atoms with (N, 3) coordinates')
        for j, a in enumerate(atoms):
            idx = ATOMIC_NUM_TO_IDX.get(int(a))
            if idx is not None:
                one_hot[i, j, idx] = 1.0
        positions[i, :n] = coords
        atom_mask[i, :n] = 1.0
    one_hot = torch.from_numpy(one_hot).to(device)
    positions = torch.from_numpy(positions).to(device)
    atom_mask = torch.from_numpy(atom_mask).to(device)

    # Fully connected edges
    nodes = torch.arange(n_nodes, device=device)
    offsets = torch.arange(batch_size, device=device)[:, None] * n_nodes
    edges = [(nodes.repeat_interleave(n_nodes)[None, :] + offsets).reshape(-1),
             (nodes.repeat(n_nodes)[None, :] + offsets).reshape(-1)]

    # Keep edges between real atoms and drop self-edges, as in the oracles' training code.
    # Self-edges shift the predictions.
    atom_mask_flat = atom_mask.view(batch_size * n_nodes, 1)
    edge_mask = atom_mask_flat[edges[0]] * atom_mask_flat[edges[1]]
    diag = torch.eye(n_nodes, dtype=torch.bool, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    edge_mask = edge_mask.view(batch_size, n_nodes, n_nodes) * (~diag)
    edge_mask = edge_mask.reshape(batch_size * n_nodes * n_nodes, 1).to(atom_mask.dtype)

    return {
        "nodes": one_hot.view(batch_size * n_nodes, NUM_ATOM_TYPES),
        "atom_positions": positions.view(batch_size * n_nodes, 3),
        "atom_mask": atom_mask.view(batch_size * n_nodes, 1),
        "edge_mask": edge_mask,
        "edges": edges,
        "n_nodes": n_nodes,
    }


@torch.no_grad()
def predict_property_batch(model, atoms_list, coords_list, mean, mad, device="cuda"):
    """Run the EGNN on one batch. Returns predictions in the oracles' units, which are eV for gap/homo/lumo."""
    tensors = xyz_to_egnn_tensors(atoms_list, coords_list, device=device)

    pred_normalized = model(
        h0=tensors["nodes"],
        x=tensors["atom_positions"],
        edges=tensors["edges"],
        edge_attr=None,
        node_mask=tensors["atom_mask"],
        edge_mask=tensors["edge_mask"],
        n_nodes=tensors["n_nodes"],
    )

    # Denormalize: raw_value = mad * pred + mean
    pred_raw = mad * pred_normalized + mean
    return pred_raw.cpu().numpy()


def predict_property_raw(model, atoms_list, coords_list, mean, mad, device="cuda",
                         property_name=None):
    """Predict in the oracles' units, in batches of 128.

    For Cv, the per-molecule atomref baseline is added back.
    """
    batch_size = 128
    all_preds = []
    for i in range(0, len(atoms_list), batch_size):
        batch_atoms = atoms_list[i : i + batch_size]
        batch_coords = coords_list[i : i + batch_size]
        preds = predict_property_batch(model, batch_atoms, batch_coords, mean, mad, device)
        all_preds.append(preds)
    preds = np.concatenate(all_preds)

    if property_name in THERMO_ATOMREF_PER_ATOM:
        baseline = np.array(
            [_thermo_baseline(property_name, atoms) for atoms in atoms_list],
            dtype=preds.dtype,
        )
        preds = preds + baseline

    return preds


def raw_to_prompt_units(values, property_name):
    """Convert the oracles' units to prompt units: eV to meV for gap/homo/lumo, identity otherwise."""
    if property_name in HARTREE_PROPERTIES:
        return values * 1000.0  # eV -> meV
    return values


def predict_property_prompt_units(model, atoms_list, coords_list, mean, mad, property_name, device="cuda"):
    """Predict property and convert to prompt units (meV, Bohr^3, D, cal/(mol*K))."""
    raw = predict_property_raw(model, atoms_list, coords_list, mean, mad, device,
                               property_name=property_name)
    return raw_to_prompt_units(raw, property_name)


def score_molecules(model, xyzs, mean, mad, property_name, device='cuda', batch_size=128):
    """Predict the property of each XYZ in prompt units, in input order.

    Malformed samples get None. Model errors are raised.
    """
    if batch_size < 1:
        raise ValueError('EGNN batch size must be positive')
    output = [None] * len(xyzs)
    valid = [i for i, xyz in enumerate(xyzs) if xyz is not None
             and 0 < len(xyz.atoms) <= MAX_N_NODES
             and np.shape(xyz.coords) == (len(xyz.atoms), 3)
             and np.isfinite(xyz.coords).all()]
    for start in range(0, len(valid), batch_size):
        indices = valid[start:start + batch_size]
        values = predict_property_prompt_units(
            model, [xyzs[i].atoms for i in indices], [xyzs[i].coords for i in indices],
            mean, mad, property_name, device=device)
        for i, value in zip(indices, values):
            output[i] = float(value) if np.isfinite(value) else None
    return output
