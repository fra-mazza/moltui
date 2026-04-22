from __future__ import annotations

from pathlib import Path

import numpy as np

from moltui.gto import eval_gto, parse_molden
from moltui.molden import load_molden_data


def _write_molden(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "sample.molden"
    path.write_text(contents)
    return path


def test_eval_gto_cartesian_d_shell(tmp_path: Path) -> None:
    """Cartesian d-shells (default without [5D]) evaluate without index errors."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
d 1 1.0
1.0 1.0

[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 0.10
2 0.20
3 0.30
4 0.40
5 0.50
6 0.60
""",
    )

    basis = parse_molden(path)
    ao = eval_gto(basis.shells, np.array([[0.1, 0.2, 0.3]]), basis.spherical)

    assert ao.shape == (1, 6)


def test_parse_molden_spherical_flags(tmp_path: Path) -> None:
    """Extended Molden flags should update spherical/cartesian behavior."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[ATOMS] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
s 1 1.0
1.0 1.0

[5D10F]
[9G]
[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 1.0
""",
    )

    basis = parse_molden(path)

    assert basis.spherical[2] is True
    assert basis.spherical[3] is False
    assert basis.spherical[4] is True


def test_parse_molden_reads_vibrational_sections(tmp_path: Path) -> None:
    """Vibrational sections should be parsed alongside MO data."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] Angs
H 1 1 0.0 0.0 0.0
[GTO]
1 0
s 1 1.0
1.0 1.0

[FREQ]
1000.0
[FR-COORD]
H 0.0 0.0 0.0
[FR-NORM-COORD]
vibration 1
0.1 0.0 0.0

[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 1.0
""",
    )

    basis = parse_molden(path)

    assert basis.mo_coefficients.shape == (1, 1)
    assert basis.atom_coords_bohr.shape == (1, 3)
    assert basis.frequencies is not None
    assert basis.normal_modes is not None
    assert np.isclose(basis.frequencies[0], 1000.0)
    assert basis.normal_modes.shape == (1, 1, 3)


def test_eval_gto_cartesian_g_shell(tmp_path: Path) -> None:
    """Cartesian g-shells (default without [9G]) should evaluate to 15 AOs."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
g 1 1.0
1.0 1.0

[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 0.10
2 0.20
3 0.30
4 0.40
5 0.50
6 0.60
7 0.70
8 0.80
9 0.90
10 1.00
11 1.10
12 1.20
13 1.30
14 1.40
15 1.50
""",
    )

    basis = parse_molden(path)
    ao = eval_gto(basis.shells, np.array([[0.1, 0.2, 0.3]]), basis.spherical)

    assert ao.shape == (1, 15)


def test_eval_gto_spherical_g_shell(tmp_path: Path) -> None:
    """Spherical g-shells ([9G]) should evaluate to 9 AOs."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] AU
H 1 1 0.0 0.0 0.0
[GTO]
1 0
g 1 1.0
1.0 1.0

[9G]
[MO]
Sym= A1
Ene= -0.1
Spin= Alpha
Occup= 2.0
1 0.10
2 0.20
3 0.30
4 0.40
5 0.50
6 0.60
7 0.70
8 0.80
9 0.90
""",
    )

    basis = parse_molden(path)
    ao = eval_gto(basis.shells, np.array([[0.1, 0.2, 0.3]]), basis.spherical)

    assert ao.shape == (1, 9)


def test_parse_molden_openmolcas_gto_style(tmp_path: Path) -> None:
    """OpenMolcas emits single-token atom-index lines and no scale factor in [GTO]."""
    path = _write_molden(
        tmp_path,
        """[Molden Format]
[Atoms] (AU)
C   1   6   0.000000000   0.000000000   0.000000000
H   2   1   1.188987000   1.188987000   1.188987000
[5D]
[7F]
[GTO] (AU)
   1
   s   2
  6.665000000E+03  6.920001793E-04
  1.000000000E+03  5.329001381E-03
   p   1
  9.439000000E+00  3.810902127E-02

   2
   s   1
  1.596000000E-01  1.000000000E+00

[MO]
Sym= A1
Ene= -11.2154
Spin= Alpha
Occup= 2.0
1  0.99
2  0.01
3  0.00
4  0.00
Sym= A1
Ene= -0.9350
Spin= Alpha
Occup= 2.0
1  0.00
2  0.60
3  0.40
4  0.00
""",
    )

    basis = parse_molden(path)

    # Two atoms, three shells (s+p on C, s on H)
    assert len(basis.atom_symbols) == 2
    assert basis.atom_symbols[0] == "C"
    assert basis.atom_symbols[1] == "H"
    assert len(basis.shells) == 3  # C: s + p, H: s
    assert basis.shells[0].l == 0  # s on C
    assert basis.shells[1].l == 1  # p on C
    assert basis.shells[2].l == 0  # s on H
    assert basis.mo_coefficients.shape == (4, 2)


def test_load_molden_data_accepts_normal_modes_only_file(tmp_path: Path) -> None:
    """Normal-modes-only Molden files should load geometry and mode vectors."""
    path = _write_molden(
        tmp_path,
        """[MOLDEN FORMAT]
[N_FREQ]
9
[FREQ]
0.0
0.0
0.0
0.0
0.0
0.0
1626.44
3645.22
3751.11
[NATOM]
3
[FR-COORD]
O -0.57 -0.43 0.30
H -0.02 1.27 0.76
H 0.60 -0.85 -1.06
[FR-NORM-COORD]
vibration 1
0.0 0.0 0.0
0.0 0.0 0.0
0.0 0.0 0.0
[INT]
0.0
""",
    )

    data = load_molden_data(path)
    assert len(data.molecule.atoms) == 3
    assert data.n_mos == 0
    assert data.normal_modes is not None
    assert data.normal_modes.shape[0] == 1
    assert data.mode_frequencies is not None
    assert np.isclose(data.mode_frequencies[-1], 0.0)
