import numpy as np
import pytest

from ase.build import bulk, fcc111


@pytest.mark.calculator('eam')
@pytest.mark.calculator_lite()
def test_eam_run(factory):
    with open(f'{factory.factory.potentials_path}/Pt_u3.eam') as fd:
        eam = factory.calc(potential=fd, form='eam', elements=['Pt'])
    slab = fcc111('Pt', size=(4, 4, 2), vacuum=10.0)
    slab.calc = eam

    assert abs(-164.277599313 - slab.get_potential_energy()) < 1E-8
    assert abs(6.36379627645 - np.linalg.norm(slab.get_forces())) < 1E-8


@pytest.mark.parametrize(
    'potential,element',
    (
        ('Pt_u3.eam', 'Pt'),
        ('NiAlH_jea.eam.alloy', 'Ni'),
        ('NiAlH_jea.eam.fs', 'Ni'),
        ('AlCu.adp', 'Al'),
    )
)
@pytest.mark.calculator('eam')
@pytest.mark.calculator_lite()
def test_read_potential(factory, potential: str, element: str):
    """Test if the potential can be read without errors."""
    potential = f'{factory.factory.potentials_path}/{potential}'
    calc = factory.calc(potential=potential, elements=[element])
    atoms = bulk(element)
    atoms.calc = calc
    atoms.get_potential_energy()
