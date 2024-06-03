import numpy as np
import pandas as pd
from kmdcm.pydcm.dcm import bohr_to_a, get_clcl


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
