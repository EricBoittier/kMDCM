from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import kB


def test_maxwellboltzmann():

    atoms = FaceCenteredCubic(size=(50, 50, 50), symbol="Cu", pbc=False)
    print("Number of atoms:", len(atoms))
    MaxwellBoltzmannDistribution(atoms, temperature_K=0.1 / kB)
    temp = atoms.get_kinetic_energy() / (1.5 * len(atoms))

    print("Temperature", temp, " (should be 0.1)")
    assert abs(temp - 0.1) < 1e-3
