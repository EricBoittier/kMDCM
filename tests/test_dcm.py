import os
import unittest

import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from kmdcm.pydcm.dcm import (
    mdcm,
    mdcm_set_up,
    scan_fesp,
    scan_fdns,
    mdcm_cxyz,
    mdcm_clcl,
    local_pos,
    get_clcl,
    optimize_mdcm,
    eval_kernel,
)
from kmdcm.pydcm.dcm import FFE_PATH, espform, densform
from kmdcm.utils import dcm_utils as du
from kmdcm.pydcm.mdcm_dict import MDCM
from kmdcm.pydcm.kernel import KernelFit


def make_df_same_size(df_dict):
    """
    Pad all dataframes in a dictionary to the same size
    :param df_dict:
    :return:
    """
    sizes = {
        k: 1 if type(df_dict[k]) is not list else len(df_dict[k])
        for k in df_dict.keys()
    }
    max_size = max(sizes.values())
    for k in df_dict.keys():
        if sizes[k] < max_size:
            df_dict[k] = np.pad(df_dict[k], (0, max_size - sizes[k]), "constant")
    return df_dict


# path to  this file
PATH_TO_TESTDIR = Path(os.path.dirname(os.path.abspath(__file__)))


def ignore_warnings(test_func):
    def do_test(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            test_func(self, *args, **kwargs)

    return do_test


class kMDCM_Experiments(unittest.TestCase):
    """
    Class of tests and experiments for kMDCM
    """

    def get_mdcm(self, mdcm_dict=None):
        if mdcm_dict is not None and type(mdcm_dict) is dict:
            if "scan_fesp" in mdcm_dict.keys():
                scan_fesp = mdcm_dict["scan_fesp"]
            else:
                scan_fesp = None
            if "scan_fdns" in mdcm_dict.keys():
                scan_fdns = mdcm_dict["scan_fdns"]
            else:
                scan_fdns = None
            if "mdcm_cxyz" in mdcm_dict.keys():
                mdcm_cxyz = mdcm_dict["mdcm_cxyz"]
            else:
                mdcm_cxyz = None
            if "mdcm_clcl" in mdcm_dict.keys():
                mdcm_clcl = mdcm_dict["mdcm_clcl"]
            else:
                mdcm_clcl = None
            if "local_pos" in mdcm_dict.keys():
                local_pos = mdcm_dict["local_pos"]
            else:
                local_pos = None
        else:
            scan_fesp = None
            scan_fdns = None
            local_pos = None
            mdcm_cxyz = None
            mdcm_clcl = None

        if scan_fesp is None:
            scan_fesp = [espform]
        if scan_fdns is None:
            scan_fdns = [densform]

        if mdcm_cxyz is None:
            mdcm_cxyz = FFE_PATH / "ff_energy/pydcm/sources/" "dcm8.xyz"
        if mdcm_clcl is None:
            mdcm_clcl = FFE_PATH / "ff_energy/pydcm/sources/" "dcm.mdcm"

        return mdcm_set_up(
            scan_fesp,
            scan_fdns,
            local_pos=local_pos,
            mdcm_cxyz=mdcm_cxyz,
            mdcm_clcl=mdcm_clcl,
        )

    def test_dcm_fortran(self):
        m = mdcm_set_up(
            scan_fesp,
            scan_fdns,
            local_pos=local_pos,
            mdcm_cxyz=mdcm_cxyz,
            mdcm_clcl=mdcm_clcl,
        )
        print(m.get_rmse())
        optimize_mdcm(m, m.mdcm_clcl, "", "test")

    def test_load_data(
            self,
            l2,
            cube_path=None,
            pickle_path=None,
            natoms=5,
            uuid=None,
            fname=None,
    ):

        print("Cube path:", cube_path)

        print(f"Loading data, uuid:{uuid}")

        if cube_path is None:
            cube_path = f"{FFE_PATH}/cubes/dcm/"

        if pickle_path is None:
            PICKLES = list(Path(f"{FFE_PATH}/cubes/clcl/{l2}").glob("*clcl.obj"))
        else:
            PICKLES = list(Path(pickle_path).glob("*clcl.obj"))

        if uuid is not None:
            PICKLES = list(
                Path(f"{FFE_PATH}/ff_energy/pydcm/tests/pkls/{uuid}").glob("*pkl"))
            if len(PICKLES) == 0:
                P = Path(f"{FFE_PATH}/cubes/clcl/{fname}/{uuid}")
                print(P)
                PICKLES = list(P.glob("*obj"))

        scanpath = Path(cube_path)
        chosen_points = []
        chosen_files = []
        for c in scanpath.glob("*/*.p.cube"):
            ccode = c.name.split(".p.")[0]
            if ccode not in chosen_points:
                chosen_points.append(ccode)
                chosen_files.append(c)

        CUBES = [
            scanpath / f"{chosen_files[i].parents[0]}/{c}.p.cube"
            for i, c in enumerate(chosen_points)
        ]

        # here we have two lists of files, one for cubes and one for pickles
        # there will always be n cube files and m x n pickle files, where n is the
        # number of conformations and m is the number of times an optimization has
        # been run
        def sort_rmse(x):
            spl = x.stem.split("_")
            for i, _ in enumerate(spl):
                if _ == "rmse":
                    return float(spl[i + 1])
            return x.stem

        # make the cube and pickle lists the same, keeping the order based on
        # print(PICKLES)
        # the cube list
        pkls = []
        for _ in chosen_points:
            # print(_)
            # tmp_pkl = [x for x in PICKLES if "_".join(_.split("_")[:3])[1:] in x.name]
            tmp_pkl = [x for x in PICKLES if str(x.stem).split(".")[0] == _]
            # print(x.name)
            # print(tmp_pkl)

            tmp_pkl.sort(key=sort_rmse)
            pkls.append(tmp_pkl[0])

        PICKLES = pkls

        # print("CUBES:", CUBES[::100])
        # print("PICKLES:", PICKLES[::100])
        print("n cubes:", len(CUBES))
        print("n pickles:", len(PICKLES))
        #  they must be the same length
        assert len(CUBES) == len(PICKLES)
        # sort them
        import re
        def clean_non_alpha(x):
            x = re.sub("[^0-9]", "", x)
            # print(x)
            return int(x)

        CUBES.sort(key=lambda x: clean_non_alpha(str(x.stem)))
        PICKLES.sort(key=lambda x: clean_non_alpha(str(x.stem).split("_")[0]))

        for i in range(len(CUBES)):
            assert clean_non_alpha(str(CUBES[i].stem)) == clean_non_alpha(
                str(PICKLES[i].stem).split("_")[0])

        #  return the data
        return du.get_data(CUBES, PICKLES, natoms)

    def test_standard_rmse(
            self,
            files,
            cubes,
            pickles,
            cubes_pwd=None,
            fname="",
            mdcm_dict=None,
    ):
        """
        Test the standard RMSE
        :param files: list of CLCL filenames
        :param cubes: list of cube objects
        :param pickles: list of pickle objects
        :param cubes_pwd: path to the cubes
        :return:
        """

        if cubes_pwd is None:
            cube_paths = Path(f"{FFE_PATH}/cubes/dcm/")
        print("cubes", len(cubes))
        # print("cubes path", cube_paths)
        # ecube_files = list(cube_paths.glob("*/*esp.cube"))
        # dcube_files = list(cube_paths.glob("*/*dens.cube"))
        ecube_files = cubes
        dcube_files = cubes
        print("ecube", len(ecube_files))
        print("dcube", len(dcube_files))
        # print(len(cubes), len(pickles))
        rmses = eval_kernel(
            files,
            cubes,
            cubes,
            fname=fname,
            mdcm_clcl=mdcm_dict["mdcm_clcl"],
            mdcm_xyz=mdcm_dict["mdcm_cxyz"],
        )

        print("RMSEs:", rmses)
        rmse = sum(rmses) / len(rmses)
        print("RMSE:", rmse)
        pd.DataFrame({"rmses": rmses, "filename": files}).to_csv(
            f"{fname}_standard_.csv"
        )

    def test_mdcm_standard_csv(self):
        fname = "methanol_perm"
        mdcm_dict = MDCM(fname + ".json").asDict()
        print(mdcm_dict)
        print(" " * 20, "Eval Null", "*" * 20)
        if len(mdcm_dict["scan_fesp"]) > 0:
            self.test_standard_rmse(
                None,
                mdcm_dict["scan_fesp"],
                mdcm_dict["scan_fdns"],
                mdcm_dict=mdcm_dict,
                fname=fname,
            )

    def test_standard(self):
        """ """
        self.test_fit(alpha=1e-5, l2="1.0", n_factor=4, load_data=False)

    def test_water(self):
        waterpath = Path(f"{FFE_PATH}/ff_energy/pydcm/water.json")
        water = MDCM(str(waterpath))
        self.test_fit(
            alpha=1e-5,
            l2="1.0",
            n_factor=4,
            load_data=False,
            mdcm_dict=water,
            do_optimize=False,
            fname="water",
            natoms=3,
        )

    def test_methanol(self):
        path = Path(f"{FFE_PATH}/ff_energy/pydcm/methanol.json")
        m = MDCM(str(path))
        self.test_fit(
            alpha=1e-5,
            l2="1.0",
            n_factor=4,
            load_data=False,
            mdcm_dict=m,
            do_optimize=True,
            fname="methanol",
            natoms=6,
            do_null=True,
        )

    def test_methanol_perm(self):
        path = Path(f"{FFE_PATH}/ff_energy/pydcm/tests/methanol_perm.json")
        m = MDCM(str(path))
        self.test_fit(
            alpha=1e-5,
            l2="1.0",
            n_factor=4,
            load_data=False,
            mdcm_dict=m,
            do_optimize=True,
            fname="methanol_perm",
            natoms=6,
            do_null=True,
        )

    def experiments(self):
        alphas = [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1]
        l2s = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]
        n_factors = [2, 4, 6, 8, 10, 12]
        for alpha in alphas:
            for l2 in l2s:
                for n in n_factors:
                    print("alpha", alpha, "l2", l2, "N_factors", n_factors)
                    self.test_fit(alpha=alpha, l2=l2, n_factor=n)

    def test_N_repeats(self, n=1):
        for i in range(n):
            print("i", i)
            self.experiments()

    @ignore_warnings
    def test_fit(
            self,
            alpha=0.0,
            l2=0.0,
            do_null=False,
            n_factor=2,
            do_optimize=False,
            cubes_pwd=FFE_PATH / "cubes/dcm/",
            mdcm_dict=None,
            load_data=False,
            fname="test",
            natoms=3,
            uuid=None,
    ):
        """
        Test the kernel fit
        """

        if mdcm_dict is None:
            # path to cubes
            cube_paths = Path(cubes_pwd)
            ecube_files = list(cube_paths.glob("*/.cube"))
            dcube_files = list(cube_paths.glob("*/*dens.cube"))
        elif isinstance(mdcm_dict, str):
            mdcm_dict = MDCM(mdcm_dict).asDict()
        else:
            mdcm_dict = mdcm_dict.asDict()

        if isinstance(mdcm_dict, dict):
            ecube_files = mdcm_dict["scan_fesp"]
            dcube_files = mdcm_dict["scan_fdns"]
            mdcm_dict["scan_fesp"] = [mdcm_dict["scan_fesp"][0]]
            mdcm_dict["scan_fdns"] = [mdcm_dict["scan_fdns"][0]]

        print("n_cubes", len(ecube_files))
        print("l2", l2)
        print(ecube_files[::100])
        #  load mdcm object
        m = self.get_mdcm(mdcm_dict=mdcm_dict)
        print("mdcm_clcl")
        print(m.mdcm_clcl)

        #  kernel fit
        k = KernelFit()

        # dp optimization
        if do_optimize and uuid is None:
            print("*" * 80)
            print("Optimizing with l2=", l2)
            opt_rmses = eval_kernel(
                None,
                ecube_files,
                dcube_files,
                opt=True,
                l2=l2,
                verbose=True,
                fname=fname,
                uuid=k.uuid,
                mdcm_clcl=mdcm_dict["mdcm_clcl"],
                mdcm_xyz=mdcm_dict["mdcm_cxyz"],
            )
            print("Opt RMSEs:", opt_rmses)
            opt_rmse = sum(opt_rmses) / len(opt_rmses)
            print("Opt RMSE:", opt_rmse)

            # unload the data
            x, i, y, cubes, pickles = self.test_load_data(
                l2=str(l2),
                pickle_path=FFE_PATH / "cubes" / "clcl" / fname / f"{l2}",
                cube_path=FFE_PATH / "cubes" / fname,
                natoms=natoms,
            )

        if uuid is not None:
            # k = pd.read_pickle()
            # unload the data
            x, i, y, cubes, pickles = self.test_load_data(
                l2=str(l2),
                pickle_path=FFE_PATH / "cubes" / "clcl" / fname / f"{uuid}",
                cube_path=FFE_PATH / "cubes" / fname,
                natoms=natoms,
                uuid=uuid,
                fname=fname
            )

            # sort them
            import re
            def clean_non_alpha(x):
                x = re.sub("[^0-9]", "", x)
                # print(x)
                return int(x)

            # CUBES.sort(key=lambda x: clean_non_alpha(str(x.stem)))
            pickles = sorted(pickles, key=lambda x: clean_non_alpha(
                str(Path(x).stem).split("_")[0]))
            ecube_files = sorted(ecube_files, key=lambda x: clean_non_alpha(
                str(Path(x).stem).split("_")[0]))
            dcube_files = sorted(dcube_files, key=lambda x: clean_non_alpha(
                str(Path(x).stem).split("_")[0]))

            # for i in range(len(CUBES)):
            #     assert clean_non_alpha(str(CUBES[i].stem)) == clean_non_alpha(str(PICKLES[i].stem).split("_")[0])

            if do_optimize:
                print("doing opt::")
                #  optimized model
                opt_rmses = eval_kernel(
                    pickles,
                    ecube_files,
                    dcube_files,
                    load_pkl=True,
                    mdcm_clcl=mdcm_dict["mdcm_clcl"],
                    mdcm_xyz=mdcm_dict["mdcm_cxyz"],
                    uuid=uuid,
                    opt=True,
                    l2=l2,
                    fname=fname,
                )
                print("Opt RMSEs:", opt_rmses)
                opt_rmse = sum(opt_rmses) / len(opt_rmses)
                print("Opt RMSE:", opt_rmse)

            # unload the data
            x, i, y, cubes, pickles = self.test_load_data(
                l2=str(l2),
                pickle_path=FFE_PATH / "cubes" / "clcl" / fname / f"{uuid}",
                cube_path=FFE_PATH / "cubes" / fname,
                natoms=natoms,
                # uuid=uuid
            )

            # all arrays should be the same length
        assert len(x) == len(i) == len(y) == len(cubes) == len(pickles)

        k.set_data(x, i, y, cubes, pickles, fname=fname)
        k.fit(alpha=alpha, N_factor=n_factor, l2=l2)

        if uuid is not None:
            k.set_prev_uuid(uuid)

        # printing
        print("*" * 20, "Kernel Fit", "*" * 20)
        print("N X:", len(k.X))
        print("N:", len(k.ids))
        print("N test:", len(k.test_ids))
        print("N_train:", len(k.train_ids))
        print("r2s:", k.r2s)
        print("sum r2s test:", sum([_[0] for _ in k.r2s]))
        print("sum r2s train:", sum([_[1] for _ in k.r2s]))
        print("n models:", len(k.r2s))
        print("r2s:", k.r2s)

        #   Move the local charges
        print("Moving clcls")
        files = k.move_clcls(m)
        # files.sort()
        print("N files:", len(files), "\n")

        print("*" * 20, "Eval Results", "*" * 20)
        #  test the original model
        if do_null:
            print(" " * 20, "Eval Null", "*" * 20)
            self.test_standard_rmse(
                k, files, cubes, pickles, mdcm_dict=mdcm_dict, fname=fname
            )

        print("nfiles", len(files))

        do_k_opt = True if uuid else False

        print(files[:10])
        print(ecube_files[:10])
        print(ecube_files[:10])

        #  test the optimized model
        rmses = eval_kernel(
            files,
            ecube_files,
            dcube_files,
            load_pkl=True,
            mdcm_clcl=mdcm_dict["mdcm_clcl"],
            mdcm_xyz=mdcm_dict["mdcm_cxyz"],
            uuid=k.uuid,
            # opt=do_k_opt,
            l2=l2,
            fname=fname,
        )

        print("len(rmses):", len(rmses))

        #  Printing the rmses
        kern_rmse = self.print_rmse(rmses)
        print("RMSEs:", rmses)
        self.prepare_df(k, rmses, files, alpha=alpha, l2=l2, fname=fname)

        if do_optimize is True:
            self.prepare_df(
                k, opt_rmses, files, alpha=alpha, l2=l2, opt=True, fname=fname
            )

        print("*" * 20, "Eval Kernel", "*" * 20)
        # plot fits
        k.plot_fits(rmses)
        k.plot_pca(rmses, title=f"Kernel ({kern_rmse:.2f})",
                   name=f"kernel_{k.uuid}")

        #  plot optimized
        if do_optimize is True:
            print("opt rmses:", opt_rmses)
            print("n_opt:", len(opt_rmses))

        #  pickle kernel
        self.pickle_kernel(k)
        #  write manifest
        print("Writing manifest")
        k.write_manifest(PATH_TO_TESTDIR / f"manifest/{k.uuid}.json")
        return k

    def print_rmse(self, rmses):
        print("RMSEs:", rmses)
        rmse = sum(rmses) / len(rmses)
        print("RMSE:", rmse)
        return rmse

    def prepare_df(self, k, rmses, files, alpha=0.0, l2=0.0, opt=False, fname="test"):
        """ """
        class_name = ["test" if _ in k.test_ids else "train" for _ in k.ids]
        if opt:
            fn = f"csvs/{fname}_opt_{k.uuid}_{l2}.csv"
        else:
            fn = f"csvs/{fname}_kernel_{k.uuid}_{alpha}_{l2}.csv"

        df_dict = {
            "rmse": rmses,
            "pkl": files,
            "class": class_name,
            "alpha": alpha,
            "uuid": k.uuid,
            "l2": l2,
            "type": ["nms" if "nms" in str(_) else "scan" for _ in files],
        }

        pd.DataFrame(make_df_same_size(df_dict)).to_csv(fn)

    def pickle_kernel(self, k):
        p = PATH_TO_TESTDIR / f"models/kernel_{k.uuid}.pkl"
        print("Pickling kernel to", p)
        with open(p, "wb") as f:
            pickle.dump(k, f)

    def test_files(self):
        i = 4
        l2 = 0.0
        cube_paths = Path("/home/boittier/Documents/phd/ff_energy/cubes/dcm/scan")
        ecube_files = list(cube_paths.glob("*esp.cube"))
        dcube_files = list(cube_paths.glob("*dens.cube"))
        print(len(ecube_files), len(dcube_files))
        ecube_files.sort()
        dcube_files.sort()
        print(ecube_files[0], dcube_files[0])
        #  name of the esp and dens cube files
        e = str(ecube_files[i])
        d = str(dcube_files[i])
        #  set up the mdcm object
        m = mdcm_set_up(
            [e], [d], local_pos=local_pos, mdcm_cxyz=mdcm_cxyz, mdcm_clcl=mdcm_clcl
        )
        print("RMSE:", m.get_rmse())
        outname = ecube_files[i].name + f"_{l2}"
        optimize_mdcm(m, m.mdcm_clcl, "", outname, l2=l2)


if __name__ == "__main__":
    from argparse import ArgumentParser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--n_factor", type=int, default=1)
    parser.add_argument("--n_atoms", type=int, default=6)
    parser.add_argument("--do_opt", action='store_true', default=False)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--fname", type=str, default="test")
    parser.add_argument("--uuid", type=str, default=None)
    parser.add_argument("--json", type=str, required=True)

    parser.add_argument("unittest_args", nargs="*")

    args = parser.parse_args()
    print(args)

    k = kMDCM_Experiments()

    k.test_fit(
        alpha=args.alpha,
        n_factor=args.n_factor,
        natoms=args.n_atoms,
        l2=args.l2,
        fname=args.fname,
        mdcm_dict=args.json,
        do_optimize=args.do_opt,
        uuid=args.uuid,
    )
