# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone snippet replicating DiffCSP evaluation pieces (https://github.com/jiaor17/DiffCSP,
scripts/eval_utils.py; MIT, Copyright (c) 2023 Rui Jiao):
- structure_validity
- comp_validity (smact_validity)
- StructureMatcher match check and RMS distance
"""

import itertools
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.structure_matcher import StructureMatcher

import smact
from smact.screening import pauling_test


# ----------------------------------------------------------------------------- #
# Utility: atomic number -> symbol lookup (mimics diffcsp.common.data_utils.chemical_symbols)
# ----------------------------------------------------------------------------- #
chemical_symbols = {z: Element.from_Z(z).symbol for z in range(1, 119)}


# ----------------------------------------------------------------------------- #
# Composition validity (from scripts/eval_utils.py)
# ----------------------------------------------------------------------------- #
def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    """
    Args:
        comp: tuple/list of atomic numbers (e.g., (14, 8) for Si, O)
        count: tuple/list of integer counts (already reduced by gcd)
    Returns:
        bool
    """
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]

    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:  # guardrail, same as repo
        return False

    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


# ----------------------------------------------------------------------------- #
# Structure validity (from scripts/eval_utils.py)
# ----------------------------------------------------------------------------- #
def structure_validity(crystal: Structure, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


# ----------------------------------------------------------------------------- #
# Match + RMS (from scripts/compute_metrics.py RecEval.get_match_rate_and_rms)
# ----------------------------------------------------------------------------- #
def match_and_rms(gt_structure: Structure, pred_structure: Structure, matcher: StructureMatcher = None):
    """
    Returns (is_match, rms_distance_in_angstrom) using StructureMatcher.get_rms_dist.
    If no match or an exception occurs, returns (False, None).
    """
    matcher = matcher or StructureMatcher()
    try:
        rms = matcher.get_rms_dist(pred_structure, gt_structure)
        rms = None if rms is None else rms[0]
    except Exception:
        rms = None
    match = rms is not None
    return match, rms


# ----------------------------------------------------------------------------- #
# Minimal Crystal wrapper (mirrors the relevant parts of scripts/compute_metrics.py)
# ----------------------------------------------------------------------------- #
class Crystal:
    def __init__(self, structure: Structure):
        self.structure = structure
        self.constructed = True
        self.invalid_reason = None
        self._compute_composition()
        self._compute_validity()

    def _compute_composition(self):
        # atom_types as atomic numbers
        atom_types = [s.specie.Z for s in self.structure.sites]
        counts = {}
        for z in atom_types:
            counts[z] = counts.get(z, 0) + 1
        elems, cnts = zip(*sorted(counts.items()))
        cnts = np.array(cnts)
        cnts = cnts // np.gcd.reduce(cnts)  # reduce by gcd
        self.elems = tuple(elems)
        self.comps = tuple(cnts.tolist())

    def _compute_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        self.struct_valid = structure_validity(self.structure) if self.constructed else False
        self.valid = self.comp_valid and self.struct_valid


# ----------------------------------------------------------------------------- #
# Demonstration
# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Build a simple cubic Si cell (ground truth)
    a = 5.43
    lattice = Lattice.cubic(a)
    gt = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

    # Create a slightly perturbed predicted structure
    lattice_pred = Lattice.cubic(a * 1.01)
    pred = Structure(lattice_pred, ["Si", "Si"], [[0, 0, 0], [0.26, 0.25, 0.24]])

    gt_crys = Crystal(gt)
    pred_crys = Crystal(pred)

    print("GT comp_valid:", gt_crys.comp_valid, "struct_valid:", gt_crys.struct_valid)
    print("Pred comp_valid:", pred_crys.comp_valid, "struct_valid:", pred_crys.struct_valid)

    match, rms = match_and_rms(gt, pred)
    print("StructureMatcher match:", match)
    print("RMS distance (Å):", rms)
