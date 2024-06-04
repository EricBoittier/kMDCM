import os
import unittest

import pandas as pd
from pathlib import Path
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from kmdcm.pydcm.dcm import (
    mdcm_set_up,
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


import re
def clean_non_alpha(x):
    x = re.sub("[^0-9]", "", x)
    return int(x)

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

        return mdcm_set_up(
            scan_fesp,
            scan_fdns,
            local_pos=local_pos,
            mdcm_cxyz=mdcm_cxyz,
            mdcm_clcl=mdcm_clcl,
        )

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
        print("Pickle path:", pickle_path)
        print(f"Loading data, uuid:{uuid}")

        if cube_path is None:
            cube_path = f"{FFE_PATH}/cubes/dcm/"

        if pickle_path is None:
            PICKLES = list(Path(f"{FFE_PATH}/cubes/clcl/{l2}").glob("*clcl.obj"))
        else:
            PICKLES = list(Path(pickle_path).glob("*"))

        scanpath = Path(cube_path)
        chosen_points = []
        chosen_files = []
        for c in scanpath.glob("*.p.cube"):
            ccode = c.name.split(".p.")[0]
            if ccode not in chosen_points:
                chosen_points.append(ccode)
                chosen_files.append(c)

        CUBES = [
            scanpath / f"{chosen_files[i].parents[0]}/{c}.p.cube"
            for i, c in enumerate(chosen_points)
        ]

        # make the cube and pickle lists the same, keeping the order based on
        # the cube list
        pkls = []
        # print("PICKLES:", PICKLES)
        for _ in chosen_points:
            tmp_pkl = [x for x in PICKLES if str(Path(_).stem).split(".")[0] == str(x.stem).split(".")[0]]
            #print(tmp_pkl)
            pkls.append(tmp_pkl[0])

        PICKLES = pkls

        #  they must be the same length
        assert len(CUBES) == len(PICKLES)
        assert len(CUBES) > 0
        # sort them
        import re
        def clean_non_alpha(x):
            #print(x)
            x = re.sub("[^0-9]", "", x)
            #print(x)
            return int(x)

        CUBES.sort(key=lambda x: clean_non_alpha(str(x.stem).split(".")[0]))
        PICKLES.sort(key=lambda x: clean_non_alpha(str(x.stem).split(".")[0]))

        for i in range(len(CUBES)):
            #print(i, CUBES[i].stem, PICKLES[i].stem)
            assert clean_non_alpha(str(CUBES[i].stem).split(".")[0]) == clean_non_alpha(
                str(PICKLES[i].stem).split(".")[0]), f"{CUBES[i].stem} {PICKLES[i].stem}"

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

        rmses = eval_kernel(
            files,
            cubes,
            cubes,
            fname=fname,
            mdcm_clcl=mdcm_dict["mdcm_clcl"],
            mdcm_xyz=mdcm_dict["mdcm_cxyz"],
        )

        # print("RMSEs:", rmses)
        rmse = sum(rmses) / len(rmses)
        print("RMSE:", rmse)
        pd.DataFrame({"rmses": rmses, "filename": files}).to_csv(
            f"{fname}_standard_.csv"
        )

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
            print("mdcm_dict")
            ecube_files = mdcm_dict["scan_fesp"]
            dcube_files = mdcm_dict["scan_fdns"]
            mdcm_dict["scan_fesp"] = [mdcm_dict["scan_fesp"][0]]
            mdcm_dict["scan_fdns"] = [mdcm_dict["scan_fdns"][0]]

        #  load mdcm object
        m = self.get_mdcm(mdcm_dict=mdcm_dict)

        #  kernel fit
        k = KernelFit()

        # dp optimization
        if do_optimize and uuid is None:
            ecube_files = sorted(ecube_files, key=lambda x: clean_non_alpha(
                str(Path(x).stem).split(".")[0]))
            dcube_files = sorted(dcube_files, key=lambda x: clean_non_alpha(
                str(Path(x).stem).split(".")[0]))
            
            print("*" * 80)
            print("Optimizing with l2=", l2)
            opt_rmses = eval_kernel(
                None,
                ecube_files,
                dcube_files,
                opt=True,
                l2=l2,
                fname=fname,
                uuid=k.uuid,
                mdcm_clcl=mdcm_dict["mdcm_clcl"],
                mdcm_xyz=mdcm_dict["mdcm_cxyz"],
            )
            # print("Opt RMSEs:", opt_rmses)
            opt_rmse = sum(opt_rmses) / len(opt_rmses)
            print("Opt RMSE:", opt_rmse)

            # unload the data
            x, i, y, cubes, pickles = self.test_load_data(
                l2=str(l2),
                pickle_path=FFE_PATH / "cubes" / "clcl" / fname / f"{k.uuid}",
                cube_path=FFE_PATH / "cubes" / fname,
                natoms=natoms,
            )
            pickles = sorted(pickles, key=lambda x: clean_non_alpha(
                str(Path(x).stem).split(".")[0]))

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


            # CUBES.sort(key=lambda x: clean_non_alpha(str(x.stem)))
            pickles = sorted(pickles, key=lambda x: clean_non_alpha(
                str(Path(x).stem).split(".")[0]))
            ecube_files = sorted(ecube_files, key=lambda x: clean_non_alpha(
                str(Path(x).stem).split(".")[0]))
            dcube_files = sorted(dcube_files, key=lambda x: clean_non_alpha(
                str(Path(x).stem).split(".")[0]))

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
                # print("Opt RMSEs:", opt_rmses)
                opt_rmse = sum(opt_rmses) / len(opt_rmses)
                print("Opt RMSE:", opt_rmse)

            # unload the data
            x, i, y, cubes, pickles = self.test_load_data(
                l2=str(l2),
                pickle_path=FFE_PATH / "cubes" / "clcl" / fname / f"{uuid}",
                cube_path=FFE_PATH / "cubes" / fname,
                natoms=natoms,
            )

        uuid = k.uuid
        # all arrays should be the same length
        assert len(x) == len(i) == len(y) == len(cubes) == len(pickles)

        k.set_data(x, i, y, cubes, pickles, fname=fname)
        k.fit(alpha=alpha, N_factor=n_factor, l2=l2)

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

        #   Move the local charges
        print("Moving clcls")
        files = k.move_clcls(m)
        print("N files:", len(files), "\n")

        print("*" * 20, "Eval Results", "*" * 20)
        #  test the original model
        if do_null:
            print(" " * 20, "Eval Null", "*" * 20)
            self.test_standard_rmse(
                k, files, cubes, pickles, mdcm_dict=mdcm_dict, fname=fname
            )

        #  test the optimized model
        rmses = eval_kernel(
            files,
            ecube_files,
            dcube_files,
            load_pkl=True,
            mdcm_clcl=mdcm_dict["mdcm_clcl"],
            mdcm_xyz=mdcm_dict["mdcm_cxyz"],
            uuid=k.uuid,
            l2=l2,
            fname=fname,
        )

        #  Printing the rmses
        self.print_rmse(rmses)
        
        # print("RMSEs:", rmses)
        kern_df = self.prepare_df(k, rmses, files, alpha=alpha, l2=l2, fname=fname)
        
        print("test:", kern_df[kern_df["class"] == "test"]["rmse"].mean())
        print("train:", kern_df[kern_df["class"] == "train"]["rmse"].mean())

        if do_optimize is True:
            opt_df = self.prepare_df(
                k, opt_rmses, files, alpha=alpha, l2=l2, opt=True, fname=fname
            )
            print("(opt) test:", opt_df[opt_df["class"] == "test"]["rmse"].mean())
            print("(opt) train:", opt_df[opt_df["class"] == "train"]["rmse"].mean())

        print("*" * 20, "Eval Kernel", "*" * 20)
        #  pickle kernel
        self.pickle_kernel(k)
        #  write manifest
        print("Writing manifest")
        k.write_manifest(PATH_TO_TESTDIR / f"manifest/{k.uuid}.json")
        return k

    def print_rmse(self, rmses):
        # print("RMSEs:", rmses)
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

        df = pd.DataFrame(make_df_same_size(df_dict))
        df.to_csv(fn)
        return df

    def pickle_kernel(self, k):
        p = PATH_TO_TESTDIR / f"models/kernel_{k.uuid}.pkl"
        print("Pickling kernel to", p)
        with open(p, "wb") as f:
            pickle.dump(k, f)


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
