import numpy as np
import sklearn
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import pickle
import uuid
from kmdcm.pydcm.dcm import get_clcl
from pathlib import Path

#  set seed for reproducibility
np.random.seed(0)


def graipher(pts, K, start=False) -> (np.ndarray, np.ndarray):
    """
    https://en.wikipedia.org/wiki/Farthest-first_traversal
    :param pts:
    :param K:
    :param start:
    :return: farthest_pts
            farthest_pts_ids
    """
    # error handling
    if K > len(pts):
        raise ValueError("K must be less than the number of points")
    if K < 1:
        raise ValueError("K must be greater than 0")
    if len(pts.shape) != 2:
        raise ValueError("pts must be a 2D array")
    # initialize the farthest points array
    farthest_pts = np.zeros((K, pts.shape[1]))
    farthest_pts_ids = []
    if start:
        farthest_pts[0] = start
    else:
        farthest_pts[0] = pts[np.random.randint(len(pts))]

    farthest_pts_ids.append(np.random.randint(len(pts)))

    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        farthest_pts_ids.append(np.argmax(distances))
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))

    return farthest_pts, farthest_pts_ids


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


class KernelFit:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.alpha = None
        self.kernel = None
        self.models = []
        self.scale_parms = []
        self.r2s = []
        self.test_results = []
        self.lcs = None
        self.test_ids = None
        self.train_ids = None
        self.train_results = []
        self.uuid = str(uuid.uuid4())
        self.lcs = None
        self.pkls = None
        self.prev_uuid = None

    def __int__(self):
        self.init()

    def init(self):
        self.models = []
        self.scale_parms = []
        self.r2s = []
        self.test_results = []
        self.train_results = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.alpha = None
        self.kernel = None
        self.test_ids = None
        self.train_ids = None
        self.lcs = None
        self.pkls = None
        self.fname = None
        self.manifest_path = None

    def set_data(self, distM, ids, lcs, cubes, pkls, fname=None):
        self.X = distM
        self.y = lcs
        self.ids = ids
        self.fname = fname
        self.cubes = cubes
        self.pkls = pkls

    def __repr__(self):
        return f"KernelFit: {self.uuid} {self.alpha} {self.kernel}"

    def __str__(self):
        return f"KernelFit: {self.uuid} {self.alpha} {self.kernel}"

    def write_manifest(self, path):
        string_ = f"{self.uuid} {self.alpha} {self.kernel} {self.fname}\nTest ids:\n"
        for test in self.test_ids:
            string_ += f"test {test}\n"
        string_ += "Train ids:\n"
        for train in self.train_ids:
            string_ += f"train {train}\n"

        self.manifest_path = path
        with open(path, "w") as f:
            f.write(string_)
        return string_

    def set_prev_uuid(self, prev_uuid):
        self.prev_uuid = prev_uuid

    def get_samples(self, N_SAMPLE_POINTS, N_factor, start):
        # sample N_SAMPLE_POINTS
        if N_SAMPLE_POINTS is None:
            N_SAMPLE_POINTS = len(self.X) // N_factor
            print("len(X)", len(self.X))
            print("N_SAMPLE_POINTS set to {}".format(N_SAMPLE_POINTS))

        points, ids = graipher(self.X, N_SAMPLE_POINTS, start=start)
        npoints = len(self.X)
        inx_vals = np.arange(npoints)
        self.train_ids = ids
        test_ids = np.delete(inx_vals, ids, axis=0)
        self.test_ids = test_ids
        self.X_train = [self.X[i] for i in ids]
        self.X_test = [self.X[i] for i in test_ids]

        return test_ids, inx_vals, npoints, ids

    def fit(
            self,
            alpha=1e-3,
            N_SAMPLE_POINTS=None,
            start=False,
            model_type=KernelRidge,
            kernel=RBF(length_scale=1.0),
            N_factor=10,
            l2=None,
            get_samples=True,
            provide_samples=None
    ):
        """

        :param alpha:
        :param N_SAMPLE_POINTS:
        :param start:
        :return:
        """
        self.alpha = alpha
        self.kernel = kernel
        self.N_factor = N_factor
        self.l2 = l2

        if get_samples:
            test_ids, inx_vals, npoints, ids = self.get_samples(N_SAMPLE_POINTS,
                                                                N_factor, start)
        else:
            test_ids, inx_vals, npoints, ids = provide_samples

        # a kernel for each axis of each charge
        for chgindx in range(self.y.shape[1]):
            lcs_ = np.array([np.array(_).flatten()[chgindx] for _ in self.y])
            y = lcs_
            y_train = np.array([y[i] for i in ids])
            y_test = np.array([y[i] for i in test_ids])

            model = model_type(
                alpha=alpha,
                kernel=kernel,
            ).fit(self.X_train, y_train)

            # evaluate the model
            train_predictions = model.predict(self.X_train)
            test_predictions = model.predict(self.X_test)

            r2_train = sklearn.metrics.r2_score(y_train, train_predictions)
            r2_test = sklearn.metrics.r2_score(y_test, test_predictions)
            #  save the model
            self.models.append(model)
            self.scale_parms.append((lcs_.min(), lcs_.max()))
            self.r2s.append([r2_test, r2_train])
            self.test_results.append((y_test, test_predictions))
            self.train_results.append((y_train, train_predictions))

    def move_clcls(self, m):
        clcl = m.mdcm_clcl
        charges = clcl.copy()
        files = []
        #  iterate over each structure
        for index, i in enumerate(self.X):
            local_pos = []
            #  iterate over each charge
            for j, model in enumerate(self.models):
                local_pos.append(model.predict([i]))
            # get the new clcl array
            new_clcl = get_clcl(local_pos, charges)
            Path(f"pkls/{self.uuid}").mkdir(parents=True, exist_ok=True)
            fn = f"pkls/{self.uuid}/{self.cubes[index].stem}.pkl"
            filehandler = open(fn, "wb")
            files.append(fn)
            pickle.dump(new_clcl, filehandler)
            filehandler.close()

        import re

        def clean_non_alpha(x):
            x = re.sub("[^0-9]", "", x)
            # print(x)
            return int(x)

        files.sort(key=lambda x: clean_non_alpha(str(Path(x).stem)))

        return files

    def predict(self, X):
        return np.array([model.predict(X) for model in self.models])
