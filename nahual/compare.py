# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import scipy.optimize
import itertools
from nahual.tokens import XYZ
from scipy.spatial.distance import cdist

def _build_transforms_cache():
    transforms = []
    # 8 sign flips
    flips = list(itertools.product([-1, 1], repeat=3))

    for f in flips:
        mat = np.zeros((3, 3))
        for i in range(3):
            mat[i, i] = f[i]
        transforms.append(mat)

    return np.stack(transforms, axis=0)

TRANSFORMS = _build_transforms_cache()

def rigid_transform_3D(A, B, allow_reflection=False):
    """Kabsch alignment of (N, 3) arrays: R (3, 3) and t (1, 3) with A @ R + t ~ B.

    det(R) = 1 unless allow_reflection is True.
    """
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0, keepdims=True)
    centroid_B = np.mean(B, axis=0, keepdims=True)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm
    U, S, Vt = np.linalg.svd(H)

    R = U @ Vt

    # Check determinant
    if not allow_reflection:
        d = np.linalg.det(R)
        if d < 0:
            # Flip the direction of the smallest singular value.
            V = Vt.T
            V[:, 2] *= -1
            R = U @ V.T

    t = centroid_B - centroid_A @ R

    return R, t


def compare_xyz(true: XYZ, pred: XYZ, chirality_matters: bool = True, refine: bool = True):
    """RMSD of pred against true after alignment and atom matching.

    Aligns by PCA axes, matches same-element atoms by optimal assignment, then
    optionally refines with ICP. Reflections are allowed only if chirality_matters
    is False. Returns (rmse, aligned_pred) with aligned_pred in true's atom order,
    or (inf, pred) if the atoms cannot be matched.
    """
    if len(true.atoms) != len(pred.atoms):
        return float('inf'), pred

    # 1. Prepare Coordinates
    coords_true = true.coords
    coords_pred = pred.coords
    atom_nums_true = true.atoms
    atom_nums_pred = pred.atoms

    # 2. Centering
    mean_true = coords_true.mean(axis=0, keepdims=True)
    mean_pred = coords_pred.mean(axis=0, keepdims=True)

    X_true = coords_true - mean_true
    X_pred = coords_pred - mean_pred

    # PCA frames
    U_true, S_true, Vt_true = np.linalg.svd(X_true, full_matrices=True)
    V_true = Vt_true.T
    Y_true = X_true @ V_true

    U_pred, S_pred, Vt_pred = np.linalg.svd(X_pred, full_matrices=True)
    V_pred = Vt_pred.T
    Y_pred = X_pred @ V_pred

    # Search the 8 axis sign flips T with Y_pred @ T ~ Y_true.
    # The rotation from X_pred to X_true is R_eff = V_pred @ T @ V_true.T.

    transforms = TRANSFORMS

    # (8, 3, 3)
    R_eff = np.matmul(np.matmul(V_pred, transforms), V_true.T)

    determinants = np.linalg.det(R_eff)

    if chirality_matters:
        valid_indices = np.where(determinants > 0)[0]
    else:
        valid_indices = np.arange(len(transforms))

    if len(valid_indices) == 0:
        valid_indices = np.arange(len(transforms))

    # Calculate costs for valid transforms
    candidate_transforms = transforms[valid_indices]
    Y_pred_cands = np.matmul(Y_pred, candidate_transforms) # (K, N, 3)

    # Cost matrix: K x N x N
    diff = Y_pred_cands[:, :, np.newaxis] - Y_true[np.newaxis, :, :] # (K, N, N, 3)
    costs = np.sum(diff ** 2, axis=-1) # (K, N, N)

    # Mask incompatible atom types
    types_match = (atom_nums_pred[:, np.newaxis] == atom_nums_true[np.newaxis, :]) # (N, N)
    large_val = 1e9
    costs = np.where(types_match[np.newaxis, :, :], costs, large_val)

    best_rmse = float('inf')
    best_perm = None

    for k in range(len(valid_indices)):
        cost_matrix = costs[k]

        if np.any(np.min(cost_matrix, axis=1) >= large_val):
             continue

        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

        current_sq_error = cost_matrix[row_ind, col_ind].sum()
        current_rmse = np.sqrt(current_sq_error / len(true.atoms))

        if current_rmse < best_rmse:
            best_rmse = current_rmse
            # perm[j] is the index in pred that matches true[j]
            best_perm = row_ind[np.argsort(col_ind)]

    if best_perm is None:
        return float('inf'), pred

    # ICP refinement
    current_perm = best_perm
    allow_ref = not chirality_matters

    if refine:
        for _ in range(20):
             permuted_pred_coords = coords_pred[current_perm]
             R, t = rigid_transform_3D(permuted_pred_coords, coords_true, allow_reflection=allow_ref)

             # Current aligned coordinates
             aligned_coords = coords_pred @ R + t

             # Solve assignment with new coordinates
             dists = cdist(aligned_coords, coords_true)
             cost_matrix = np.where(types_match, dists ** 2, large_val)

             if np.any(np.min(cost_matrix, axis=1) >= large_val):
                  break

             row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
             new_perm = row_ind[np.argsort(col_ind)]

             if np.array_equal(new_perm, current_perm):
                  break
             current_perm = new_perm

    # Final Rigid Transform based on best permutation
    permuted_pred_coords = coords_pred[current_perm]
    R, t = rigid_transform_3D(permuted_pred_coords, coords_true, allow_reflection=allow_ref)

    final_coords = permuted_pred_coords @ R + t
    diff = final_coords - coords_true
    final_rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))

    # Construct result XYZ
    final_atoms = pred.atoms[current_perm]
    aligned_xyz = XYZ(atoms=final_atoms, coords=final_coords)

    return final_rmse, aligned_xyz
