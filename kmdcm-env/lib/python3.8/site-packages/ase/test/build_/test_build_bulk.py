"""Tests for `bulk`"""
import pytest

from ase.build import bulk
from ase.data import chemical_symbols, reference_states

lat_map = {
    'fcc': 'FCC',
    'bcc': 'BCC',
    'hcp': 'HEX',
    'bct': 'BCT',
    'diamond': 'FCC',
    # 'sc': 'CUB',
    # 'orthorhombic': 'ORC',
    'rhombohedral': 'RHL',
}


def test_build_bulk():
    """Test reference states"""
    lat_counts: dict = {}

    for symbol in chemical_symbols:
        atomic_number = chemical_symbols.index(symbol)
        ref = reference_states[atomic_number]

        if ref is None:
            continue

        structure = str(ref['symmetry'])
        if structure not in lat_map:
            continue

        if symbol in {'B', 'Se', 'Te'}:
            continue
        lat_counts.setdefault(structure, []).append(symbol)

        atoms = bulk(symbol)
        lattice = atoms.cell.get_bravais_lattice()
        print(
            atomic_number,
            atoms.symbols[0],
            structure,
            lattice,
            atoms.cell.lengths(),
        )
        par1 = lattice.tocell().niggli_reduce()[0].cellpar()
        par2 = atoms.cell.niggli_reduce()[0].cellpar()
        assert abs(par2 - par1).max() < 1e-10
        assert lat_map[structure] == lattice.name

        if lattice.name in ['RHL', 'BCT']:
            continue

        _check_orthorhombic(symbol)

        if lattice.name in ['HEX', 'TET', 'ORC']:
            continue

        _check_cubic(symbol)

        for key, val in lat_counts.items():
            print(key, len(val), ''.join(val))


def _check_orthorhombic(symbol: str):
    atoms = bulk(symbol, orthorhombic=True)
    lattice = atoms.cell.get_bravais_lattice()
    angles = lattice.cellpar()[3:]
    assert abs(angles - 90.0).max() < 1e-10


def _check_cubic(symbol: str):
    atoms = bulk(symbol, cubic=True)
    lattice = atoms.cell.get_bravais_lattice()
    assert lattice.name == 'CUB', lattice


@pytest.mark.parametrize('structure', ['sc'])
@pytest.mark.parametrize('symbol', chemical_symbols)
def test_crystal_structures(symbol: str, structure: str):
    """Test crystal structures"""
    bulk(symbol, structure, a=1.0)


@pytest.mark.parametrize('symbol', ['Mn', 'P'])
def test_complex_cubic(symbol: str):
    """Test elements with complex cubic reference states"""
    with pytest.raises(ValueError):
        bulk(symbol)
