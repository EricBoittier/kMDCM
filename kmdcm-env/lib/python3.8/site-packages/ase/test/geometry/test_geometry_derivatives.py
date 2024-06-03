import numpy as np
import pytest

from ase.geometry import (get_angles, get_angles_derivatives, get_dihedrals,
                          get_dihedrals_derivatives, get_distances_derivatives)


def get_numerical_derivatives(positions, mode, epsilon):
    if mode == 'distance':
        mode_n = 0  # get derivative for bond 0-1
    elif mode == 'angle':
        mode_n = 1  # get derivative for angle 0-1-2
    elif mode == 'dihedral':
        mode_n = 2  # get derivative for dihedral 0-1-2-3
    derivs = np.zeros((2 + mode_n, 3))
    for i in range(2 + mode_n):
        for j in range(3):
            pos = positions.copy()
            pos[i, j] -= epsilon
            if mode == 'distance':
                minus = np.linalg.norm(pos[1] - pos[0])
            elif mode == 'angle':
                minus = get_angles(
                    [pos[0] - pos[1]],
                    [pos[2] - pos[1]],
                )[0]  # size-1 array -> scalar
            elif mode == 'dihedral':
                minus = get_dihedrals(
                    [pos[1] - pos[0]],
                    [pos[2] - pos[1]],
                    [pos[3] - pos[2]],
                )[0]  # size-1 array -> scalar
            pos[i, j] += 2 * epsilon
            if mode == 'distance':
                plus = np.linalg.norm(pos[1] - pos[0])
            elif mode == 'angle':
                plus = get_angles(
                    [pos[0] - pos[1]],
                    [pos[2] - pos[1]],
                )[0]  # size-1 array -> scalar
            elif mode == 'dihedral':
                plus = get_dihedrals(
                    [pos[1] - pos[0]],
                    [pos[2] - pos[1]],
                    [pos[3] - pos[2]],
                )[0]  # size-1 array -> scalar
            derivs[i, j] = (plus - minus) / (2 * epsilon)
    return derivs


def test_atoms_angle():
    # example: positions for H2O2 molecule
    pos = np.asarray([[0.840, 0.881, 0.237],     # H
                      [0., 0.734, -0.237],       # O
                      [0., -0.734, -0.237],      # O
                      [-0.840, -0.881, 0.237]])  # H

    # analytical derivatives in Angstrom/Angstrom, i.e. degrees/Angstrom
    distances_derivs = get_distances_derivatives([pos[1] - pos[0]])[0]
    angles_derivs = get_angles_derivatives([pos[0] - pos[1]],
                                           [pos[2] - pos[1]])[0]
    dihedrals_derivs = get_dihedrals_derivatives([pos[1] - pos[0]],
                                                 [pos[2] - pos[1]],
                                                 [pos[3] - pos[2]])[0]

    # numerical approximations to derivatives using finite differences
    epsilon = 1e-5
    num_distances_derivs = get_numerical_derivatives(pos, mode='distance',
                                                     epsilon=epsilon)
    num_angles_derivs = get_numerical_derivatives(pos, mode='angle',
                                                  epsilon=epsilon)
    num_dihedrals_derivs = get_numerical_derivatives(pos, mode='dihedral',
                                                     epsilon=epsilon)

    # print(distances_derivs - num_distances_derivs)
    # print(angles_derivs - num_angles_derivs)
    # print(dihedrals_derivs - num_dihedrals_derivs)

    # finite differences versus analytical results
    assert num_distances_derivs == pytest.approx(distances_derivs, abs=1e-8)
    assert num_angles_derivs == pytest.approx(angles_derivs, abs=1e-8)
    assert num_dihedrals_derivs == pytest.approx(dihedrals_derivs, abs=1e-8)

    # derivatives of multiple internal coordinates at once
    assert (distances_derivs ==
            get_distances_derivatives([pos[1] - pos[0],
                                       pos[1] - pos[0]])[0]).all()
    assert (angles_derivs ==
            get_angles_derivatives([pos[0] - pos[1], pos[0] - pos[1]],
                                   [pos[2] - pos[1],
                                    pos[2] - pos[1]])[0]).all()
    assert (dihedrals_derivs ==
            get_dihedrals_derivatives([pos[1] - pos[0], pos[1] - pos[0]],
                                      [pos[2] - pos[1], pos[2] - pos[1]],
                                      [pos[3] - pos[2],
                                       pos[3] - pos[2]])[0]).all()
