import warnings
from typing import IO, Optional, Union

import numpy as np

from ase import Atoms
from ase.md.md import MolecularDynamics


class VelocityVerlet(MolecularDynamics):
    def __init__(
        self,
        atoms: Atoms,
        timestep: Optional[float] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        dt: Optional[float] = None,
        append_trajectory: bool = False,
    ):
        """Molecular Dynamics object.

        Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        timestep: float
            The time step in ASE time units.

        trajectory: Trajectory object or str  (optional)
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Default: None.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.  Default: None.

        loginterval: int (optional)
            Only write a log line for every *loginterval* time steps.
            Default: 1

        append_trajectory: boolean
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.

        dt: float (deprecated)
            Alias for timestep.
        """
        if dt is not None:
            warnings.warn(
                FutureWarning(
                    'dt variable is deprecated; please use timestep.'))
            timestep = dt
        if timestep is None:
            raise TypeError('Missing timestep argument')

        MolecularDynamics.__init__(self, atoms, timestep, trajectory, logfile,
                                   loginterval,
                                   append_trajectory=append_trajectory)

    def step(self, forces=None):

        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces(md=True)

        p = atoms.get_momenta()
        p += 0.5 * self.dt * forces
        masses = atoms.get_masses()[:, np.newaxis]
        r = atoms.get_positions()

        # if we have constraints then this will do the first part of the
        # RATTLE algorithm:
        atoms.set_positions(r + self.dt * p / masses)
        if atoms.constraints:
            p = (atoms.get_positions() - r) * masses / self.dt

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.
        atoms.set_momenta(p, apply_constraint=False)

        forces = atoms.get_forces(md=True)

        # Second part of RATTLE will be done here:
        atoms.set_momenta(atoms.get_momenta() + 0.5 * self.dt * forces)
        return forces
