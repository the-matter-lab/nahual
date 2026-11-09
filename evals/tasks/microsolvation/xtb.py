# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np
from rdkit.Chem import GetPeriodicTable

PTABLE = GetPeriodicTable()

def write_xyz(atoms, coords, path):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for atom, coord in zip(atoms, coords):
            if isinstance(atom, (int, np.integer, np.int64, np.int32, np.int8)):
                symbol = PTABLE.GetElementSymbol(int(atom))
            else:
                symbol = atom
            f.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

def generate_xtb_scripts(xyz_dirs, script_path="run_xtb.sh"):
    """Write a script that runs xTB on cluster.xyz and solute.xyz in each of xyz_dirs.

    Jobs run in parallel with xargs. Each uses an xTB namespace so restart files do not collide.
    """
    cmd_file = script_path + ".cmds"
    with open(cmd_file, "w") as f:
        for d in xyz_dirs:
            abs_d = os.path.abspath(d)
            cmd = f"cd {abs_d} && " \
                  f"export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && " \
                  f"( [ -f cluster.out ] || xtb cluster.xyz --alpb water --namespace cluster > cluster.out 2>&1 ) && " \
                  f"( [ -f solute.out ] || xtb solute.xyz --alpb water --namespace solute > solute.out 2>&1 )"
            f.write(cmd + "\n")

    lines = [
        "#!/bin/bash",
        "# Assumes xtb is already available in PATH (via venv or system install)",
        "",
        "# Number of parallel jobs",
        "NPROCS=${1:-${NAHUAL_METRIC_WORKERS:-16}}",
        "",
        f"echo \"Running $(wc -l < {cmd_file}) jobs in parallel with $NPROCS cores...\"",
        f"cat {cmd_file} | xargs -L 1 -P $NPROCS -I {{}} bash -c \"{{}}\"",
        "echo \"Done.\""
    ]

    with open(script_path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(script_path, 0o755)


def parse_xtb_energy(out_path):
    if not os.path.exists(out_path):
        return None
    try:
        with open(out_path, "r") as f:
            for line in f:
                if "| TOTAL ENERGY" in line:
                    return float(line.split()[3])
    except Exception as e:
        print(f"Error parsing {out_path}: {e}")
    return None

def calculate_eform(e_cluster, e_solute, n_solvent, e_solvent=-5.085036885157):
    """Eform = Ecluster - Esolute - n * Esolvent, in Hartree."""
    if e_cluster is None or e_solute is None:
        return None
    return e_cluster - e_solute - n_solvent * e_solvent
