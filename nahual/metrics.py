# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Molecular validity, structure alignment and bond-graph helpers.

check_validity_smiles follows symphony (https://github.com/atomicarchitects/symphony,
analyses/metrics.py; MIT, Copyright (c) 2024 The Atomic Architects). The covalent radii
and bonding rule of get_bond_graph are those of 3Dmol.js (BSD-3-Clause, Copyright (c)
2014, University of Pittsburgh and contributors).
"""
from networkx.algorithms.isomorphism import GraphMatcher
import networkx as nx
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
import scipy.optimize
from scipy.spatial.distance import pdist
import torch


def check_validity_smiles(M):
    """Checks whether a molecule is valid using xyz2mol."""
    mol = Chem.MolFromXYZBlock(M.xyzfile)

    # We should only have one conformer.
    if mol.GetNumConformers() != 1:
        return False, None

    try:
        RDLogger.DisableLog("rdApp.*")
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        smi = Chem.MolToSmiles(mol)
        RDLogger.EnableLog("rdApp.*")
    except:
        return False, None

    if mol.GetNumBonds() == 0:
        return False, None

    return True, smi


### IMPORTANT: validity depends on rdkit version
### when calculating validity on the 100k training set of QM9:
### rdkit==2023.03.3 gives 99.99%
### later versions rdkit>2023.03.3 give 94.78%, tested up to rdkit=2024.09.4
def compute_valid_unique(Ms, return_invalid=False):
    assert type(Ms) is list

    num_samples = len(Ms)
    num_valid = 0
    all_smiles = []
    invalid_idxs = []
    for i, M in enumerate(Ms):
        valid, smiles = check_validity_smiles(M)
        if valid:
            num_valid += 1
            all_smiles.append(smiles)
        elif return_invalid:
            invalid_idxs.append(i)

    validity = num_valid / num_samples
    uniqueness = len(set(all_smiles)) / num_valid if num_valid > 0 else 0

    if return_invalid:
        return validity, uniqueness, invalid_idxs
    return validity, uniqueness


def compute_rmsd(atoms_pred, atoms_true, coords_pred, coords_true):
    """RMSD under the best assignment of atoms of the same element, without alignment."""
    # assumes numpy arrays

    if atoms_pred.shape[0] != atoms_true.shape[0]:
        return False, np.inf

    costs = np.square(
        coords_pred[:, np.newaxis, :] - coords_true[np.newaxis, :, :]
    ).sum(axis=-1)
    costs = np.where(
        atoms_pred[:, np.newaxis] == atoms_true[np.newaxis, :], costs, np.inf
    )

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(costs)
    rmsd = np.sqrt(np.mean(costs[row_ind, col_ind]))

    return True, rmsd


def find_rigid_alignment_ordered(A, B):
    """Kabsch registration with known correspondences: R, t with R @ a + t ~ b.

    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    A (source) and B (target) are (N, D) arrays.
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.dot(B_c)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    # Rotation matrix
    R = V.dot(U.T)
    # Ensure R is a proper rotation
    if np.linalg.det(R) < 0:  # reflection
        V[:, -1] *= -1  # flip the sign of the last column of V
        R = V.dot(U.T)
    # Translation vector
    t = b_mean - R.dot(a_mean)
    return R, t


def get_rmsd_np(A, B):
    return np.sqrt((((A - B) ** 2).sum(axis=1)).mean())


def align_ordered_and_rmsd(A, B):
    """RMSD after Kabsch alignment. A and B must list atoms in the same order."""
    if isinstance(A, torch.Tensor):
        A = A.numpy()
    if isinstance(B, torch.Tensor):
        B = B.numpy()
    if not A.shape == B.shape:
        print(
            f"Error: A and B must have the same shape, but got {A.shape} and {B.shape}"
        )
        return None
    R, t = find_rigid_alignment_ordered(A, B)
    A_aligned = (R.dot(A.T)).T + t
    return get_rmsd_np(A_aligned, B)


def kabsch(P, Q):
    """P aligned onto Q by the Kabsch algorithm (same atom order)."""
    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Center the point clouds
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q

    # Compute covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Remove a reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_Q - R @ centroid_P

    return P @ R.T + t


import numpy as np

# Periodic table index = atomic number
_Z_TO_ELEM = [
    None, "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn"
]

# Covalent radii (Å) as in 3Dmol.js bondTable
_ELEM_RADIUS = {
    "H":0.37,"He":0.32,
    "Li":1.34,"Be":0.90,"B":0.82,"C":0.77,"N":0.75,"O":0.73,"F":0.71,"Ne":0.69,
    "Na":1.54,"Mg":1.30,"Al":1.18,"Si":1.11,"P":1.06,"S":1.02,"Cl":0.99,"Ar":0.97,
    "K":1.96,"Ca":1.74,"Sc":1.44,"Ti":1.56,"V":1.25,"Mn":1.39,"Fe":1.25,"Co":1.26,"Ni":1.21,"Cu":1.38,"Zn":1.31,"Ga":1.26,"Ge":1.22,"Se":1.16,"Br":1.14,"Kr":1.10,
    "Rb":2.11,"Sr":1.92,"Y":1.62,"Zr":1.48,"Nb":1.37,"Mo":1.45,"Tc":1.56,"Ru":1.26,"Rh":1.35,"Pd":1.31,"Ag":1.53,"Cd":1.48,"In":1.44,"Sn":1.41,"Sb":1.38,"Te":1.35,"I":1.33,"Xe":1.30,
    "Cs":2.25,"Ba":1.98,"Lu":1.60,"Hf":1.50,"Ta":1.38,"W":1.46,"Re":1.59,"Os":1.44,"Ir":1.37,"Pt":1.28,"Au":1.44,"Hg":1.49,"Tl":1.48,"Pb":1.47,"Bi":1.46,"Rn":1.45
}
_DEFAULT_RADIUS = 1.6  # matches bondLength fallback
_CATION_Z = {11, 12, 19, 20, 25, 38}  # Na, Mg, K, Ca, Mn, Sr (unboundCations option)
_FUDGE = 0.25
_MIN_DIST2 = 0.5  # same as JS guard

def _radius_from_Z(z: int) -> float:
    if 0 < z < len(_Z_TO_ELEM):
        elem = _Z_TO_ELEM[z]
        if elem and elem in _ELEM_RADIUS:
            return _ELEM_RADIUS[elem]
    return _DEFAULT_RADIUS

def get_bond_graph(atoms: np.ndarray, coords: np.ndarray, *, unbound_cations: bool = False):
    """Bonded pairs (i, j), i < j, from atomic numbers and coords in Å, using 3Dmol.js covalent radii."""
    atoms = np.asarray(atoms, dtype=int)
    coords = np.asarray(coords, dtype=float)
    n = len(atoms)
    edges = []
    for i in range(n):
        zi = atoms[i]
        ri = _radius_from_Z(zi)
        ci = coords[i]
        for j in range(i + 1, n):
            zj = atoms[j]
            # optional unbound cation skip
            if unbound_cations and (zi in _CATION_Z or zj in _CATION_Z):
                continue
            rj = _radius_from_Z(zj)
            maxdist = ri + rj + _FUDGE
            maxdist2 = maxdist * maxdist
            d = ci - coords[j]
            dist2 = float(np.dot(d, d))
            if dist2 < _MIN_DIST2 or dist2 > maxdist2:
                continue
            edges.append((i, j))
    return edges

# Example:
# edges = get_bond_graph(np.array([6,6,1,1]), np.array([[0,0,0],[1.4,0,0],[0.7,0.9,0],[0.7,-0.9,0]]))
# print(edges)

import networkx as nx


from networkx.algorithms.isomorphism import GraphMatcher
def build_graph(edge_list, atoms):
    """Graph with nodes labeled by atomic number and (i, j) edges."""
    G = nx.Graph()

    # Add nodes with atom labels
    for i, atom in enumerate(atoms):
        G.add_node(i, atom=int(atom))

    # Add edges
    G.add_edges_from(edge_list)

    return G

def are_isomorphic(G1, G2):
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        attr='atom',
        default=None
    )

    matcher = GraphMatcher(G1, G2, node_match=node_match)
    return matcher.is_isomorphic()

def get_water_indices(atoms, coords):
    """Index sets of the water molecules: bonded components of exactly one O and two H."""
    edges = get_bond_graph(atoms, coords)
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    G.add_edges_from(edges)

    water_molecule_indices = []
    for component in nx.connected_components(G):
        if len(component) == 3:
            comp_atoms = [atoms[i] for i in component]
            comp_atoms = list(comp_atoms)
            if comp_atoms.count(8) == 1 and comp_atoms.count(1) == 2:
                water_molecule_indices.append(set(component))

    return water_molecule_indices

def extract_solute(atoms, coords):
    """Atoms and coords with the water molecules removed."""
    water_indices_list = get_water_indices(atoms, coords)
    all_water_indices = set().union(*water_indices_list) if water_indices_list else set()

    solute_indices = [i for i in range(len(atoms)) if i not in all_water_indices]

    solute_atoms = atoms[solute_indices]
    solute_coords = coords[solute_indices]

    return solute_atoms, solute_coords

def count_waters(atoms, coords):
    """Number of water molecules."""
    return len(get_water_indices(atoms, coords))

def check_overlaps(coords, threshold=0.8):
    """True if any two atoms are closer than threshold (Å)."""
    if len(coords) < 2:
        return False
    dists = pdist(coords)
    if (dists < threshold).any():
        return True
    return False
