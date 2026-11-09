# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rdkit.Chem as Chem  # type: ignore[import]
import torch
from ase import Atoms
from pymatgen.core import Structure, Lattice
from transformers import AutoTokenizer

from nahual.paths import qwen3_dirs

PTABLE = Chem.GetPeriodicTable()  # type: ignore[attr-defined]

STOP = eos_token_id = 151645
QWEN3_VOCAB_SIZE = 151669
GEN = QWEN3_VOCAB_SIZE
END = QWEN3_VOCAB_SIZE + 1
COORD_TOKEN_END = QWEN3_VOCAB_SIZE + 2 + 118 + 7
VOCAB_SIZE = COORD_TOKEN_END

new_tokens = ["<xyz>", "</xyz>"]
new_tokens.extend([f"<atom_{i}>" for i in range(1, 119)])
new_tokens.extend(["<lattice_o>", "<lattice_a>", "<lattice_b>", "<lattice_c>", "<lattice_a_open>", "<lattice_b_open>", "<lattice_c_open>"])

LATTICE_O = QWEN3_VOCAB_SIZE + 2 + 118
LATTICE_A = QWEN3_VOCAB_SIZE + 2 + 118 + 1
LATTICE_B = QWEN3_VOCAB_SIZE + 2 + 118 + 2
LATTICE_C = QWEN3_VOCAB_SIZE + 2 + 118 + 3


def has_coords(token):
    return (token >= QWEN3_VOCAB_SIZE + 2) & (token < COORD_TOKEN_END)


def coord_component_mask(tokens, n_components=3):
    tokens = torch.as_tensor(tokens) if not torch.is_tensor(tokens) else tokens
    return has_coords(tokens).unsqueeze(-1).expand(*tokens.shape, n_components)


class _LazyTokenizer:
    def __init__(self):
        self._value = None

    def __getattr__(self, name):
        if self._value is None:
            tok = AutoTokenizer.from_pretrained(qwen3_dirs["0.6B"], local_files_only=True)
            tok.add_tokens(new_tokens, special_tokens=True)
            if tok.convert_tokens_to_ids("<xyz>") != GEN:
                raise ValueError("Tokenizer vocabulary does not match Qwen3")
            self._value = tok
        return getattr(self._value, name)

tokenizer = _LazyTokenizer()

def atomic_number_to_token(atomic_number):
    return QWEN3_VOCAB_SIZE + 2 + atomic_number - 1

def token_to_atomic_number(token):
    return token - QWEN3_VOCAB_SIZE - 2 + 1

def split_sequence(atoms, coords):
    """
    Splits the sequence of atoms and coordinates into text and 3D parts.

    e.g. output of
    [
        {'type': 'text', 'tokens': array([ 8768,  4156,   773, 14975,  3120,  3382,  1007, 19581,  1005])},
        {'type': '3D',
         'atoms': array([8, 6, 7, 8, 6, 7, 1, 1]),
         'coords': array([[-1.9161061 , -0.50034785, -0.0682271 ],
                         [-0.64161247, -0.12256031, -0.02861518],
                         [-0.26242754,  1.1148667 ,  0.14836867],
                         [ 1.1245967 ,  1.0127702 ,  0.12090967],
                         [ 1.3995892 , -0.27198094, -0.07008167],
                         [ 0.36446226, -1.0383763 , -0.17331184],
                         [-2.4675689 ,  0.2841823 ,  0.07111529],
                         [ 2.444218  , -0.55172175, -0.1176736 ]], dtype=float32)},
        {'type': 'text', 'tokens': array([ 939, 1007, 2400])}
    ]
    """
    import torch
    if isinstance(atoms, torch.Tensor):
        atoms_list = atoms.tolist()
    else:
        atoms_list = list(atoms)

    parts = []
    i = 0
    N = len(atoms_list)
    while i < N:
        if atoms_list[i] == GEN:
            # Found a 3D block
            try:
                j = atoms_list.index(END, i + 1)
                parts.append({'type': '3D', 'atoms': atoms[i+1:j], 'coords': coords[i+1:j]})
                i = j + 1
            except ValueError:
                # No END token, treat the rest as a 3D block
                parts.append({'type': '3D', 'atoms': atoms[i+1:], 'coords': coords[i+1:]})
                i = N
        else:
            # Found a text block
            try:
                j = atoms_list.index(GEN, i)
            except ValueError:
                j = N # No more GEN tokens
            parts.append({'type': 'text', 'tokens': atoms[i:j]})
            i = j
    return parts

def xyzfile(atoms: np.ndarray, coords: np.ndarray, comment: str = "") -> str:
    """Format atomic numbers and coordinates as an XYZ file string.

    Stops at the first atom 0. Out-of-range atomic numbers are written as element 118.
    """
    num_atoms = 0

    xyz = ""
    for i in range(atoms.shape[0]):
        atom = atoms[i]
        if atom == 0:
            break
        if atom > 118 or atom < 1:
            atom = 118
        element = PTABLE.GetElementSymbol(int(atom))
        coord = coords[i]
        xyz += f"{element:<3} {coord[0]:14.8f} {coord[1]:14.8f} {coord[2]:14.8f}\n"
        num_atoms += 1

    return f"{num_atoms}\n{comment}\n{xyz}"

def force_atomic_number(atoms):
    atoms = np.where((atoms < 1) | (atoms > 125), 118, atoms)
    return atoms

def parse_seq(tokens, coords):
    import torch
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
        tokens_list = tokens.tolist()
    else:
        tokens_list = list(tokens)
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    seq = []
    i = 0
    N = len(tokens_list)
    while i < N:
        if tokens_list[i] == GEN:
            # Found a 3D block
            try:
                j = tokens_list.index(END, i + 1)
                atoms = token_to_atomic_number(tokens[i+1:j])
                atoms = force_atomic_number(atoms)
                xyz = XYZ(
                    atoms=atoms,
                    coords=coords[i+1:j],
                )
                seq.append(xyz)
                i = j + 1
            except ValueError:
                # No END token, treat the rest as a xyz block
                atoms = token_to_atomic_number(tokens[i+1:])
                atoms = force_atomic_number(atoms)
                xyz = XYZ(
                    atoms=atoms,
                    coords=coords[i+1:],
                )
                seq.append(xyz)
                i = N
        else:
            # Found a text block
            try:
                j = tokens_list.index(GEN, i)
            except ValueError:
                j = N # No more GEN tokens

            text = tokenizer.decode(tokens[i:j], skip_special_tokens=False)
            seq.append(text)
            i = j
    return seq

def parse_xyz(file, read_file=True):

    if read_file:
        with open(file, 'r') as f:
            lines = f.readlines()
    else:
        lines = file.split('\n')

    lines = [line.strip() for line in lines if line.strip()]

    if not lines:
        return XYZ(np.array([], dtype=np.int64), np.array([], dtype=np.float32).reshape(0, 3))

    data_lines = lines
    # Optional header: an atom count, then a comment line unless the next line holds coordinates.
    try:
        # If the first line is a single integer, it's the number of atoms
        if len(lines[0].split()) == 1:
            num_atoms = int(lines[0])
            # Assume a comment line follows, so data starts at line 2.
            # But if there's no comment line, data starts at line 1.
            # A simple check for an atom line on line 2 (index 1)
            if len(lines) > 1 and len(lines[1].split()) >= 4:
                try:
                    # check if it's a coordinate line
                    parts = lines[1].split()
                    float(parts[1])
                    float(parts[2])
                    float(parts[3])
                    # It is, so data starts at line 1
                    data_lines = lines[1:1+num_atoms]
                except (ValueError, IndexError):
                    # Not a coordinate line, assume comment, data starts at line 2
                    data_lines = lines[2:2+num_atoms]
            else: # Not enough lines for atom data or line 2 is not atom-like
                data_lines = lines[2:2+num_atoms]

    except (ValueError, IndexError):
        # No header
        pass

    atomic_numbers = []
    coordinates = []

    for line in data_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        element = parts[0]
        x, y, z = map(float, parts[1:4])
        # Get atomic number from the element symbol using RDKit periodic table
        atomic_number = PTABLE.GetAtomicNumber(element)
        atomic_numbers.append(atomic_number)
        coordinates.append([x, y, z])

    # Convert to numpy
    a = np.array(atomic_numbers, dtype=np.int64)
    c = np.array(coordinates, dtype=np.float32)

    return XYZ(a, c)

@dataclass
class XYZ:
    atoms: np.ndarray
    coords: np.ndarray

    def zero_center(self):
        mean = self.coords.mean(axis=0)
        self.coords -= mean

    def rotate(
        self,
        alpha: float = 0.0,
        beta: float = 0.0,
        gamma: float = 0.0,
        order: str = "XYZ",
        pivot: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        inplace: bool = True,
    ) -> "XYZ":
        """Rotate by Euler angles in radians about the x, y and z axes.

        order "XYZ" gives R = Rx @ Ry @ Rz and "ZYX" gives R = Rz @ Ry @ Rx.
        Coordinates become (X - pivot) @ R.T + pivot, with pivot defaulting to the
        centroid. Raises on lattice-bearing XYZ. Returns self if inplace, else a copy.
        """
        if self.has_lattice:
            raise ValueError("Rotation on lattice-bearing XYZ is not supported in this method.")

        if self.coords.size == 0:
            return self

        ordu = order.upper()
        if ordu not in {"ZYX", "XYZ"}:
            raise ValueError(f"Unsupported rotation order '{order}'. Use 'ZYX' or 'XYZ'.")

        # Build canonical rotation matrices
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)

        Rx = np.array([[1.0, 0.0, 0.0],
                       [0.0,  ca, -sa],
                       [0.0,  sa,  ca]], dtype=np.float32)
        Ry = np.array([[ cb, 0.0,  sb],
                       [0.0, 1.0, 0.0],
                       [-sb, 0.0,  cb]], dtype=np.float32)
        Rz = np.array([[ cg, -sg, 0.0],
                       [ sg,  cg, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)

        if ordu == "ZYX":
            R = Rz @ Ry @ Rx
        else:  # "XYZ"
            R = Rx @ Ry @ Rz

        # Determine pivot point
        if pivot is None:
            p = self.coords.mean(axis=0)
        else:
            p = np.asarray(pivot, dtype=self.coords.dtype)
            if p.shape != (3,):
                raise ValueError("pivot must be a length-3 vector")

        Xc = self.coords - p  # translate
        Yc = Xc @ R.T         # rotate for row-vectors
        Y = Yc + p            # translate back

        if inplace:
            self.coords = Y.astype(np.float32, copy=False)
            return self
        else:
            return XYZ(atoms=self.atoms.copy(), coords=Y.astype(np.float32, copy=False))

    def add_lattice(self, lattice: Lattice):
        pbc = lattice.pbc

        # Origin 119, then lattice vectors a, b, c as 120-122 (periodic) or 123-125 (open).
        new_atoms = [119]
        new_atoms.append(120 if pbc[0] else 123)  # Lattice a
        new_atoms.append(121 if pbc[1] else 124)  # Lattice b
        new_atoms.append(122 if pbc[2] else 125)  # Lattice c

        self.atoms = np.concatenate([new_atoms, self.atoms])
        self.coords = np.concatenate([np.zeros((1, 3)), lattice.matrix, self.coords], axis=0)
        self.zero_center()

    @property
    def has_lattice(self):
        return self.atoms[0] == 119

    @property
    def pbc(self):
        if not self.has_lattice:
            return None

        pbc = []
        pbc.append(True if self.atoms[1] == 120 else False)
        pbc.append(True if self.atoms[2] == 121 else False)
        pbc.append(True if self.atoms[3] == 122 else False)
        return pbc

    def to_pymatgen(self):
        """Convert a lattice-bearing XYZ to a pymatgen Structure, or None on failure."""
        assert self.has_lattice, "This XYZ object does not have a lattice."
        origin = self.coords[0]
        coords_shifted = self.coords - origin
        lattice_matrix = coords_shifted[1:4]

        pbc_list = self.pbc

        atoms = self.atoms[4:]
        coords = coords_shifted[4:]


        species = []
        for atom in atoms:
            if atom > 118 or atom < 1:
                atom = 118
            species.append(PTABLE.GetElementSymbol(int(atom)))

        try:
            # Ensure pbc is a tuple of bools as expected by Lattice
            pbc_tuple: tuple[bool, bool, bool] = tuple(pbc_list) if pbc_list is not None else (True, True, True)
            lattice = Lattice(lattice_matrix, pbc=pbc_tuple)
            structure = Structure(lattice, species, coords, coords_are_cartesian=True)
            return structure
        except Exception:
            return None

    def to_ase_atoms(self):
        if self.has_lattice:
            origin = self.coords[0]
            coords_shifted = self.coords - origin
            lattice_matrix = coords_shifted[1:4]

            pbc_list = self.pbc

            atoms = self.atoms[4:]
            coords = coords_shifted[4:]

            cell = lattice_matrix

        else:
            atoms = self.atoms
            coords = self.coords
            cell = None
            pbc_list = None

        return Atoms(
            positions=coords,
            numbers=atoms,
            cell=cell,
            pbc=pbc_list,
        )

    @property
    def xyzfile(self):
        assert not self.has_lattice, "This XYZ object has a lattice, use to_pymatgen() instead."
        return xyzfile(self.atoms, self.coords)

    @classmethod
    def from_xyz(cls, xyz_str, read_file=True):
        return parse_xyz(xyz_str, read_file=read_file)

    @classmethod
    def from_cif(cls, cif_str, niggli=True, raster=False):
        return cif_to_xyz(cif_str, niggli=niggli, raster=raster)

    @classmethod
    def from_ase_atoms(cls, ase_atoms):
        return ase_atoms_to_xyz(ase_atoms)


def wrap_crystal(xyz, fractional=False):
    """Wrap the atoms of a lattice-bearing XYZ into the unit cell.

    Returns (atoms, coords) without the lattice rows. Coords are fractional in
    [0, 1) if fractional is True, else Cartesian.
    """
    assert xyz.has_lattice, "XYZ must have a lattice."
    origin = xyz.coords[0]
    coords_shifted = xyz.coords - origin
    lattice_matrix = coords_shifted[1:4]  # (3, 3)
    atoms = xyz.atoms[4:]
    cart_coords = coords_shifted[4:]  # (N, 3)

    # cart = frac @ lattice_matrix  =>  frac = cart @ inv(lattice_matrix)
    inv_lat = np.linalg.inv(lattice_matrix)
    frac = cart_coords @ inv_lat
    frac_wrapped = frac % 1.0

    if fractional:
        return atoms, frac_wrapped
    return atoms, frac_wrapped @ lattice_matrix

def slice_xyz_by_range(xyz: XYZ, atom_range: range) -> XYZ:
    """Copy the atoms of a non-periodic XYZ at the indices in ``atom_range``.

    Indices are 1-based.
    """
    if xyz.has_lattice:
        raise ValueError(
            "slice_xyz_by_range does not support lattice-bearing XYZ; remove the lattice first."
        )
    n = int(xyz.atoms.shape[0])
    raw_indices = list(atom_range)
    # reduce all indices by 1
    raw_indices = [i - 1 for i in raw_indices]
    # some sanity checks
    if not raw_indices:
        raise IndexError("Empty atom range specified.")
    if any(i < 0 for i in raw_indices):
        raise IndexError("Negative indices are not allowed.")
    if any(i >= n for i in raw_indices):
        raise IndexError(f"Indices must be less than {n}.")

    out_atoms = xyz.atoms[raw_indices].copy()
    out_coords = xyz.coords[raw_indices].copy()
    return XYZ(atoms=out_atoms, coords=out_coords)

def row_to_xyz(row):
    """Convert an ``xyz`` data segment (a dict with atoms and coords) to an XYZ."""
    assert row["type"] == "xyz", "Row type must be 'xyz'"
    atoms = np.array(row['atoms'], dtype=np.int64)
    coords = np.array(row['coords'], dtype=np.float32)
    return XYZ(atoms=atoms, coords=coords)

def cif_to_xyz(cif_str, niggli=True, raster=False):
    struc = Structure.from_str(cif_str, fmt="cif")
    if niggli:
        struc = struc.get_reduced_structure()

    if raster:
        struc.sort(key=lambda s: tuple(reversed(s.frac_coords)))

    elements = [site.species_string for site in struc.sites]
    coords = [site.coords for site in struc.sites]

    atoms = np.array([PTABLE.GetAtomicNumber(el) for el in elements])
    coords = np.array(coords, dtype=np.float32)
    xyz = XYZ(atoms=atoms, coords=coords)
    xyz.add_lattice(struc.lattice)
    return xyz

def ase_atoms_to_xyz(ase_atoms):
    atoms = ase_atoms.get_atomic_numbers()
    coords = ase_atoms.get_positions()
    xyz = XYZ(atoms=atoms, coords=coords)

    pbc = ase_atoms.get_pbc()
    cell = ase_atoms.get_cell()
    if not (pbc == 0).all():
        lattice = Lattice(cell)
        lattice.pbc = pbc
        xyz.add_lattice(lattice)
    return xyz

CHAT_USER_PREFIX_IDS = np.array([151644, 872, 198], dtype=np.int64)

CHAT_ASSISTANT_PREFIX_IDS = np.array([151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271], dtype=np.int64)

def parse_segments(segments):
    seq = []
    for seg in segments:
        if seg["type"] == "text":
            seq.append(seg["text"])
        elif seg["type"] == "xyz":
            atoms = np.array(seg["atoms"])
            coords = np.array(seg["coords"])
            xyz = XYZ(atoms=atoms, coords=coords)
            seq.append(xyz)
        else:
            raise ValueError(f"Unknown segment type: {seg['type']}")
    return seq

def tokenize_seq(seq: list[str | XYZ]):
    gen = np.array([GEN], dtype=np.int64)
    end = np.array([END], dtype=np.int64)
    zero = np.zeros((1, 3), dtype=np.float32)
    seq_tokens = []
    seq_coords = []
    for item in seq:
        if isinstance(item, str):
            ids = tokenizer.encode(item, add_special_tokens=False)
            coords = np.zeros((len(ids), 3), dtype=np.float32)
        elif isinstance(item, XYZ):
            ids = np.concatenate([gen, atomic_number_to_token(item.atoms), end], axis=0)
            coords = np.concatenate([zero, item.coords, zero], axis=0)
        else:
            raise ValueError(f"Unknown item type: {type(item)}")
        seq_tokens.append(ids)
        seq_coords.append(coords)

    seq_tokens = np.concatenate(seq_tokens, axis=0)
    seq_coords = np.concatenate(seq_coords, axis=0)
    return seq_tokens, seq_coords

def apply_chat_template(prompt, bsz=1):
    seq_tokens, seq_coords = tokenize_seq(prompt)
    before = np.array([151644, 872, 198])
    after = np.array([151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271])
    seq_tokens = np.concatenate([before, seq_tokens, after], axis=0)
    seq_coords = np.concatenate([np.zeros((before.shape[0], 3), dtype=np.float32), seq_coords, np.zeros((after.shape[0], 3), dtype=np.float32)], axis=0)

    prompt_a = seq_tokens[None, :].repeat(bsz, axis=0)
    prompt_x = seq_coords[None, :].repeat(bsz, axis=0)

    return prompt_a, prompt_x

def make_seq(list_of_dict):
    seq = []
    for d in list_of_dict:
        if d["type"] == "text":
            seq.append(d["text"])
        elif d["type"] == "xyz":
            seq.append(XYZ(atoms=np.array(d["atoms"]), coords=np.array(d["coords"]).reshape(-1, 3)))
    return seq
