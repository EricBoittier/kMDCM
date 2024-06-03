import numpy as np
import sklearn
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import uuid
from pathlib import Path
from sklearn.decomposition import PCA
from ff_energy.pydcm.dcm import get_clcl
from pathlib import Path

from sklearn.preprocessing import StandardScaler

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
        # self.X = None
        # self.y = None
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
            test_ids, inx_vals, npoints, ids = self.get_samples(N_SAMPLE_POINTS, N_factor, start)
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
            return  int(x)
        
        files.sort(key=lambda x: clean_non_alpha(str(Path(x).stem)))
            
        return files
    
    def move_new_clcls(self, m, location, filenames, X):
        print("...")
        pass

    def predict(self, X):
        return np.array([model.predict(X) for model in self.models])

    def pca(self):
        pca = PCA(n_components=2)
        pca.fit(self.X)
        return pca.transform(self.X)

    def plot_pca(self, rmses, title=None, name=None):
        pca = self.pca()
        markers = [5 if "nms" in str(_) else 2 for _ in self.pkls]
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        sc = ax.scatter(pca[:, 0], pca[:, 1], c=rmses, s=markers, cmap="viridis")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        import matplotlib as mpl

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="RMSE"
        )

        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        if name is not None:
            plt.savefig(f"pngs/{self.uuid}_{name}.png", bbox_inches="tight")

    def plot_fits(self, rmses, name=None):
        N = len(self.models) // 3
        fig, ax = plt.subplots(N, 2, figsize=(5, 15))
        test_rmses = [rmses[i] for i in self.test_ids]
        train_rmses = [rmses[i] for i in self.train_ids]
        print("n test", len(test_rmses), "n train", len(train_rmses))
        print(np.mean(test_rmses), np.mean(train_rmses))

        for i in range(N):
            for j in range(3):
                ij = i * 3 + j
                y_test, test_predictions = self.test_results[ij]
                y_train, train_predictions = self.train_results[ij]

                ax[i][0].scatter(y_test, test_predictions, c=test_rmses)

                ax[i][0].set_title("{} test r2: {:.2f}".format(i, self.r2s[ij][0]))

                ax[i][1].scatter(y_train, train_predictions, c=train_rmses)
                ax[i][1].set_title("{} train r2: {:.2f}".format(i, self.r2s[ij][1]))

                ax[i][0].set_xlabel("actual")
                ax[i][0].set_ylabel("predicted")
                ax[i][1].set_xlabel("actual")
                ax[i][1].set_ylabel("predicted")
                ax[i][0].set_xlim(-2, 2)
                ax[i][0].set_ylim(-2, 2)
                ax[i][1].set_xlim(-2, 2)
                ax[i][1].set_ylim(-2, 2)

        plt.tight_layout()
        if name is not None:
            plt.savefig(name, bbox_inches="tight")


"""
Plotting
"""


def plot3d(i, ax, test_results, test_angle):
    plt.set_cmap("CMRmap")
    # plt.set_cmap('jet')
    # fig, ax = plt.subplots(1,2,subplot_kw=dict(projection='3d'))
    ax[0].set_proj_type("ortho")
    ax[0].view_init(20, -120)

    X, Y, Z = test_results[i][0], test_results[i + 1][0], test_results[i + 2][0]
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    ax[0].scatter(X, Y, Z, c=np.array(test_angle))

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array(
        [X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]
    ).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
        X.max() + X.min()
    )
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
        Y.max() + Y.min()
    )
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
        Z.max() + Z.min()
    )

    ax[0].plot(X, Z, zs=Yb.max(), zdir="y", label="points in (x, z)", c="g", alpha=0.1)
    ax[0].plot(Y, Z, zs=Xb.max(), zdir="x", label="points in (x, z)", c="r", alpha=0.1)
    ax[0].plot(X, Y, zs=Zb.min(), zdir="z", label="points in (x, z)", c="b", alpha=0.1)
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax[0].plot([xb], [yb], [zb], "w")
    # plt.show()

    plt.set_cmap("CMRmap")
    ax[1].set_proj_type("ortho")

    X, Y, Z = test_results[i][1], test_results[i + 1][1], test_results[i + 2][1]
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    ax[1].plot(X, Z, zs=Yb.max(), zdir="y", label="points in (x, z)", c="g", alpha=0.1)
    ax[1].plot(Y, Z, zs=Xb.max(), zdir="x", label="points in (x, z)", c="r", alpha=0.1)
    ax[1].plot(X, Y, zs=Zb.min(), zdir="z", label="points in (x, z)", c="b", alpha=0.1)

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax[1].plot([xb], [yb], [zb], "w")

    ax[1].scatter(X, Y, Z, c=np.array(test_angle))
    ax[1].view_init(20, -120)


def plot3d(i, ax, test_results, test_angle, test_rmses):
    # plt.set_cmap('CMRmap')
    plt.set_cmap("viridis_r")
    fig, ax = plt.subplots(
        1, 3, figsize=(14, 3), sharey=False, gridspec_kw={"width_ratios": [2, 2, 3]}
    )
    plt.subplots_adjust(wspace=0.7)

    X, Y, Z = test_results[i][0], test_results[i + 1][0], test_results[i + 2][0]
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    dist = [np.linalg.norm([x, y, z]) * 0.56 for x, y, z in zip(X, Y, Z)]
    ax[0].scatter(np.array(test_angle), dist, color="k", s=6, alpha=0.5)
    data1 = pd.DataFrame({"angle": test_angle, "dist": dist})

    ax[0].scatter(data1["angle"], data1["dist"])

    X, Y, Z = test_results[i][1], test_results[i + 1][1], test_results[i + 2][1]
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    dist = [np.linalg.norm([x, y, z]) * 0.56 for x, y, z in zip(X, Y, Z)]
    ax[1].scatter(np.array(test_angle), dist, c=test_rmses, s=6, alpha=0.5)
    data2 = pd.DataFrame({"angle": test_angle, "dist": dist})
    # sns.lineplot(data=data2, x="angle", y="dist",ax=ax[1], color="gray")
    ax[1].scatter(data2["angle"], data2["dist"])

    ax[2].scatter(data1["dist"], data2["dist"], c=test_rmses, alpha=0.5)
    ax[2].set_xlim(data1["dist"].min(), data1["dist"].max())
    ax[2].set_ylim(data1["dist"].min(), data1["dist"].max())
    ax[2].plot(
        [data1["dist"].min(), data1["dist"].max()],
        [data1["dist"].min(), data1["dist"].max()],
        c="k",
    )
    ax[2].set_aspect(1)

    ax[0].set_ylim(data1["dist"].min(), data1["dist"].max())
    ax[1].set_ylim(data1["dist"].min(), data1["dist"].max())
    ax[0].set_ylabel("$r_{\mathrm{DC}}$ [$\mathrm{\AA}$]", fontsize=20)
    ax[0].set_xlabel("$\\theta _{\mathrm{HOC}}$ [$^{\circ}$]", fontsize=20)
    ax[1].set_xlabel("$\\theta _{\mathrm{HOC}}$ [$^{\circ}$]", fontsize=20)
    plt.savefig(f"{i}_charges3d.pdf", bbox_inches="tight")
