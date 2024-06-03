import numpy as np
import os
import pickle
import pandas as pd

from ff_energy.pydcm.dcm import bohr_to_a, get_clcl
import cclib


def load_nc(path, n=3):
    #  load nuclear coordinates
    with open(path) as f:
        nc_lines = f.readlines()[6 : 6 + n]
    ncs = np.array([[float(y) * bohr_to_a for y in x.split()[2:]] for x in nc_lines])
    return ncs


def get_dist_matrix(atoms):
    # https://www.kaggle.com/code/rio114/coulomb-interaction-speed-up/notebook
    num_atoms = len(atoms)
    loc_tile = np.tile(atoms.T, (num_atoms, 1, 1))
    dist_mat = np.sqrt(((loc_tile - loc_tile.T) ** 2).sum(axis=1))
    return dist_mat


def scale_min_max(data, x):
    return (x - data.min()) / (data.max() - data.min())


def scale_max(data, x):
    return (x) / (data.max())


def inv_scale_min_max(x, dmin, dmax):
    return x * (dmax - dmin) + dmin


def scale_Z(data, x):
    return (x - data.mean()) / (data.std())


def inv_scale_Z(x, dmean, dstd):
    return x * dstd + dmean


def scale_sum(data, x):
    return x / sum(data)


# https://stackoverflow.com/a/31364297/412655
def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def get_data(cubes, pickles, natoms):
    """
    Returns the distance matrix, ids and local charge positions of the pickles
    """
    distM = []
    ids = []
    lcs = []
    for i, (cube, pickle_name) in enumerate(zip(cubes, pickles)):
        # print(cube, pickle_name)
        ncs = load_nc(cube, n=natoms)
        dm = get_dist_matrix(ncs)
        # reduce to only the upper triangle, no diagonals (zeros)
        iu1 = np.triu_indices(natoms)
        uptri = dm[iu1]
        uptri_dm = uptri[uptri != 0]
        pkl = pd.read_pickle(pickle_name)
        local = pkl[np.mod(np.arange(pkl.size) + 1, 4) != 0]
        distM.append(uptri_dm)
        ids.append(i)
        lcs.append(local)

    return (np.array(_) for _ in [distM, ids, lcs, cubes, pickles])


def get_cclib_data(filename):
    data = cclib.io.ccread(filename)
    return data
