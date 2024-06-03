import numpy as np

from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list


def fcut(r, r0, r1):
    """
    Piecewise quintic C^{2,1} regular polynomial for use as a smooth cutoff.
    Ported from JuLIP.jl, https://github.com/JuliaMolSim/JuLIP.jl

    Parameters
    ----------
    r0 - inner cutoff radius
    r1 - outder cutoff radius
    """""
    s = 1.0 - (r - r0) / (r1 - r0)
    return (s >= 1.0) + ((s > 0.0) & (s < 1.0)) * (
        6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3
    )


def fcut_d(r, r0, r1):
    """
    Derivative of fcut() function defined above
    """
    s = 1 - (r - r0) / (r1 - r0)
    return -(
        ((s > 0.0) & (s < 1.0))
        * (30 * s**4 - 60 * s**3 + 30 * s**2)
        / (r1 - r0)
    )


class MorsePotential(Calculator):
    """Morse potential.

    Default values chosen to be similar as Lennard-Jones.
    """

    implemented_properties = ['energy', 'forces']
    default_parameters = {'epsilon': 1.0,
                          'rho0': 6.0,
                          'r0': 1.0,
                          'rcut1': 1.9,
                          'rcut2': 2.7}
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        epsilon: float
          Absolute minimum depth, default 1.0
        r0: float
          Minimum distance, default 1.0
        rho0: float
          Exponential prefactor. The force constant in the potential minimum
          is k = 2 * epsilon * (rho0 / r0)**2, default 6.0
        """
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['positions', 'numbers', 'cell',
                                  'pbc', 'charges', 'magmoms']):
        Calculator.calculate(self, atoms, properties, system_changes)
        epsilon = self.parameters.epsilon
        rho0 = self.parameters.rho0
        r0 = self.parameters.r0
        rcut1 = self.parameters.rcut1 * r0
        rcut2 = self.parameters.rcut2 * r0

        forces = np.zeros((len(self.atoms), 3))
        preF = - 2 * epsilon * rho0 / r0

        i, j, d, D = neighbor_list('ijdD', atoms, rcut2)
        dhat = (D / d[:, None]).T

        expf = np.exp(rho0 * (1.0 - d / r0))
        fc = fcut(d, rcut1, rcut2)

        E = epsilon * expf * (expf - 2)
        dE = preF * expf * (expf - 1) * dhat
        energy = 0.5 * (E * fc).sum()

        F = (dE * fc + E * fcut_d(d, rcut1, rcut2) * dhat).T
        for dim in range(3):
            forces[:, dim] = np.bincount(i, weights=F[:, dim],
                                         minlength=len(atoms))

        self.results['energy'] = energy
        self.results['forces'] = forces
