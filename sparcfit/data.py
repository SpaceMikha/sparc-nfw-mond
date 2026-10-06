"""Reading the SPARC master table and the per-galaxy mass models."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Column order of the data block of SPARC_Lelli2016c.mrt. The rows are read as
# whitespace-separated fields; the byte ranges in the file header are not exact.
_FIELDS = ("name", "T", "D", "e_D", "f_D", "inc", "e_inc", "L36", "e_L36", "Reff",
           "SBeff", "Rdisk", "SBdisk", "MHI", "RHI", "Vflat", "e_Vflat", "Q", "ref")
_INTEGER = ("T", "f_D", "Q")


@dataclass
class Galaxy:
    name: str
    D: float        # Mpc
    e_D: float      # Mpc
    f_D: int        # distance method (1 Hubble flow, 2 TRGB, 3 Cepheids, 4 UMa, 5 SN)
    inc: float      # deg
    e_inc: float    # deg
    L36: float      # 1e9 Lsun
    SBeff: float    # Lsun / pc^2
    MHI: float      # 1e9 Msun
    Vflat: float    # km/s, 0 if the curve never flattens
    Q: int
    R: np.ndarray       # kpc
    Vobs: np.ndarray    # km/s
    eV: np.ndarray      # km/s
    Vgas: np.ndarray    # km/s (signed)
    Vdisk: np.ndarray   # km/s for M/L = 1
    Vbul: np.ndarray    # km/s for M/L = 1

    @property
    def N(self):
        return self.R.size

    @property
    def has_bulge(self):
        return bool(np.any(self.Vbul > 0))


def read_master(path=None):
    path = Path(path) if path else DATA_DIR / "SPARC_Lelli2016c.mrt"
    lines = path.read_text().splitlines()
    # The data block follows the last dashed separator line.
    start = max(i for i, line in enumerate(lines) if line.startswith("-----")) + 1
    table = []
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = line.split()
        row = dict(zip(_FIELDS, parts))
        for key in _FIELDS[1:-1]:
            row[key] = int(row[key]) if key in _INTEGER else float(row[key])
        table.append(row)
    return table


def load_sample(data_dir=None):
    """Return the 175 SPARC galaxies as a list of Galaxy objects."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    sample = []
    for row in read_master(data_dir / "SPARC_Lelli2016c.mrt"):
        rot = np.loadtxt(data_dir / "rotmod" / f"{row['name']}_rotmod.dat", ndmin=2)
        sample.append(Galaxy(
            name=row["name"], D=row["D"], e_D=row["e_D"], f_D=row["f_D"],
            inc=row["inc"], e_inc=row["e_inc"], L36=row["L36"], SBeff=row["SBeff"],
            MHI=row["MHI"], Vflat=row["Vflat"], Q=row["Q"],
            R=rot[:, 0], Vobs=rot[:, 1], eV=rot[:, 2],
            Vgas=rot[:, 3], Vdisk=rot[:, 4], Vbul=rot[:, 5],
        ))
    return sample
