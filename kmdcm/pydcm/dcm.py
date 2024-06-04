from kmdcm.pydcm.dcm_fortran import dcm_fortran
import numpy as np

# Basics
import os
import subprocess
import time
from pathlib import Path, PosixPath
import pickle
import pandas as pd
import argparse

# Optimization
from scipy.optimize import minimize

bohr_to_a = 0.529177

DCM_PY_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "dcm.py"
HOME_PATH = Path(os.path.expanduser("~"))
FFE_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]
print("DCMPY:", DCM_PY_PATH)
print("HOME:", HOME_PATH)
print("FFE:", FFE_PATH)

mdcm = dcm_fortran

espform = FFE_PATH / "cubes/dcm/nms/test_nms_0_0.xyz_esp.cube"
densform = FFE_PATH / "cubes/dcm/nms/test_nms_0_0.xyz_dens.cube"

scan_fesp = [espform]
scan_fdns = [densform]

mdcm_cxyz = FFE_PATH / "ff_energy/pydcm/sources/dcm8.xyz"
mdcm_clcl = FFE_PATH / "ff_energy/pydcm/sources/dcm.mdcm"

local_pos = None


def mdcm_set_up(scan_fesp, scan_fdns, mdcm_cxyz=None, mdcm_clcl=None, local_pos=None):
    # convert PosixPath to string
    if isinstance(scan_fesp, PosixPath):
        scan_fesp = str(scan_fesp)
    if isinstance(scan_fdns, PosixPath):
        scan_fdns = str(scan_fdns)
    if isinstance(mdcm_cxyz, PosixPath):
        mdcm_cxyz = str(mdcm_cxyz)
    if isinstance(mdcm_clcl, PosixPath):
        mdcm_clcl = str(mdcm_clcl)

    # Load MDCM global and local files
    mdcm.dealloc_all()
    Nfiles = len(scan_fesp)
    Nchars = int(
        np.max(
            [
                len(str(filename))
                for filelist in [scan_fesp, scan_fdns]
                for filename in filelist
            ]
        )
    )

    esplist = np.empty((Nfiles, Nchars), dtype="U{:d}".format(Nchars), order="F")
    dnslist = np.empty((Nfiles, Nchars), dtype="U{:d}".format(Nchars), order="F")
    for ifle in range(Nfiles):
        esplist[ifle] = "{0:{1}s}".format(str(scan_fesp[ifle]), Nchars)
        dnslist[ifle] = "{0:{1}s}".format(str(scan_fdns[ifle]), Nchars)

    # Load cube files, read MDCM global and local files
    mdcm.load_cube_files(Nfiles, Nchars, esplist.T, dnslist.T)

    if mdcm_cxyz is not None:
        print(mdcm_cxyz)
        mdcm.load_cxyz_file(mdcm_cxyz)
        cxyz = mdcm.mdcm_cxyz
        print("cxyz:", cxyz)
        # mdcm.set_cxyz(cxyz)

    if mdcm_clcl is not None:
        print(mdcm_clcl)
        mdcm.load_clcl_file(mdcm_clcl)
        print(mdcm.mdcm_clcl)

    if local_pos is not None:
        print("Setting local positions")
        mdcm.set_clcl(local_pos)

    # Write MDCM global from local and Fitted ESP cube files
    mdcm.write_cxyz_files()
    mdcm.write_mdcm_cube_files()
    return mdcm


def get_clcl(local_pos, charges):
    NCHARGES = len(charges)
    _clcl_ = np.zeros(NCHARGES)
    for i in range(NCHARGES):
        if (i + 1) % 4 == 0:
            _clcl_[i] = charges[i]
        else:
            if i >= 70 * 4:
                _clcl_[i] = 0.0
            else:
                _clcl_[i] = local_pos[i - ((i) // 4)]
    return _clcl_


def set_bounds(local_pos, change=0.1):
    bounds = []
    for i, x in enumerate(local_pos):
        bounds.append((x - abs(x) * change, x + abs(x) * change))
    return tuple(bounds)


def optimize_mdcm(mdcm, clcl, outname, l2, fname, esp, eval_prev=False):
    # Get RMSE, averaged or weighted over ESP files,
    # or per ESP file each
    if fname is None:
        fname = "test"

    rmse = mdcm.get_rmse()
    print(uuid)
    print(rmse)
    print("clcl: ", clcl)
    if eval_prev == uuid[4:]:
        clcl = mdcm.mdcm_clcl

    #  save an array containing original charges
    charges = clcl.copy()
    local_pos = clcl[np.mod(np.arange(clcl.size) + 1, 4) != 0]
    local_ref = local_pos.copy()

    def mdcm_rmse(local_pos, local_ref=local_ref, l2=l2):
        """Minimization routine"""
        _clcl_ = get_clcl(local_pos, charges)

        mdcm.set_clcl(_clcl_)
        rmse = mdcm.get_rmse()
        if local_ref is not None:
            l2diff = l2 * np.sum((local_pos - local_ref) ** 2) / local_pos.shape[0]
            rmse += l2diff
        return rmse

    print("local_ref", local_ref)

    # Apply simple minimization without any feasibility check (!)
    # Leads to high amplitudes of MDCM charges and local positions
    res = minimize(
        mdcm_rmse,
        local_pos,
        method="L-BFGS-B",
        bounds=[(-0.55, 0.55)] * len(local_ref),
        options={
            "disp": None,
            "maxls": 20,
            "iprint": -1,
            "gtol": 1e-6,
            "eps": 1e-6,
            "maxiter": 15000,
            "ftol": 1e-6,
            "maxcor": 10,
            "maxfun": 15000,
        },
    )
    print(res)
    print(res.x)
    # Recompute final RMSE each
    rmse = mdcm.get_rmse()
    print(rmse)
    mdcm.write_cxyz_files()
    outfn = esp + "_" + fname + "_" + "opt" + "_" + uuid + ".xyz"
    # rename the file
    os.rename(esp + ".mdcm.xyz", outfn)
    #  get the local charges array after optimization
    clcl_out = get_clcl(res.x, charges)
    print(clcl_out)
    difference = np.sum((res.x - local_ref) ** 2) / local_pos.shape[0]
    print("charge RMSD:", difference)

    outname = f"{outname}_l2_{l2:.1e}_rmse_{rmse:.4f}_rmsd_{difference:.4f}"
    on = outname.split("/")[-1]

    if not eval_prev:
        obj_name = f"{FFE_PATH}/cubes/clcl/{fname}/{l2}/{on}_clcl.obj"
    else:
        obj_name = f"{FFE_PATH}/cubes/clcl/{fname}/{eval_prev}/{on}_clcl.obj"

    print(obj_name)

    # make parent directory if it does not exist
    if not os.path.exists(os.path.dirname(obj_name)):
        try:
            os.makedirs(os.path.dirname(obj_name))
        except Exception as e:
            print(e)

    #  save as pickle
    with open(obj_name, "wb") as filehandler:
        pickle.dump(clcl_out, filehandler)

    # Not necessary but who knows when it becomes important to deallocate all
    # global arrays
    # mdcm.dealloc_all()
    print("RMSE:", rmse)


def eval_kernel(
        clcls,
        esp_path,
        dens_path,
        mdcm_clcl=None,
        mdcm_xyz=None,
        load_pkl=False,
        opt=False,
        l2=None,
        fname=None,
        uuid=None,
):
    """
    Evaluate kernel for a set of ESP and DENS files
    """
    rmses = []
    commands = []
    N = len(esp_path)
    path__ = f"{FFE_PATH}/ff_energy/cubes/clcl/{fname}/{l2}"

    for i in range(N):
        ESP_PATH = esp_path[i]
        DENS_PATH = dens_path[i]
        job_command = (
            f"python {DCM_PY_PATH} -esp {ESP_PATH} -dens {DENS_PATH} "
            f" -mdcm_clcl {mdcm_clcl} -mdcm_xyz {mdcm_xyz} "
        )
        #  if loading the local charges from a pickle file
        if load_pkl:
            job_command += f" -l {clcls[i]} "

        if uuid is not None:
            if opt:
                job_command += f" -uuid opt-{uuid} "
            else:
                job_command += f" -uuid rmse-{uuid} "
        #  if optimizing
        if opt:
            _optname = Path(ESP_PATH).stem
            #  normal optimization
            if uuid is None:
                job_command += (
                    f" -opt True -l2 {l2} "
                    f"-o {FFE_PATH}/cubes/clcl/{fname}/{l2}/{_optname}"
                )
            #  opt. from prev solution.
            else:
                job_command += (
                    f" -opt True -l2 {l2} "
                    f" -eval_prev {uuid} "
                    f"-o {FFE_PATH}/cubes/clcl/{fname}/{uuid}/{_optname}"
                )

        job_command += f" -fname {fname}"
        Path(path__).mkdir(parents=True, exist_ok=True)

        job_command = f"sbatch --wrap='{job_command}' --partition=vshort"  # vshort
        if i == 0:
            print(job_command)

        commands.append(job_command)

    procs = []
    for command in commands:
        p = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        procs.append(p)
        time.sleep(0.005)

    SLEEPTIME = 60 * 0.5
    if opt:
        SLEEPTIME += 0.5 * 60

    print(f"SLEEPING: {SLEEPTIME}")
    time.sleep(SLEEPTIME)

    for p in procs:
        p.wait()
        result = p.stdout.readlines()
        rmse_out = int(result[-1].split()[-1])
        rmses.append(rmse_out)

    tmp_rmse = []
    # load RMSE from slurm output
    for fn in rmses:
        fn = f"slurm-{fn}.out"
        with open(fn) as f:
            lines = f.readlines()
            rmse = [_ for _ in lines if "RMSE:" in _][0].split()[-1]
            tmp_rmse.append(float(rmse))
        os.remove(fn)
    rmses = tmp_rmse
    return rmses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan and average for fMDCM")
    parser.add_argument("-n", "--nodes_to_avg", help="", required=False, type=int)
    parser.add_argument("-l", "--local_pos", help="", default=None, type=str)
    parser.add_argument(
        "-l2", "--l2", help="lambda coef. for l2 reg.", default=100.0, type=float
    )
    parser.add_argument("-o", "--outdir", help="", default=None, type=str)
    parser.add_argument("-opt", "--opt", help="", default=False, type=bool)
    parser.add_argument(
        "-esp", "--esp", help="format string for esp files", default=None, type=str
    )
    parser.add_argument(
        "-dens",
        "--dens",
        help="format string for density files",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-mdcm_clcl", "--mdcm_clcl", help="mdcm clcl file", default=None, type=str
    )
    parser.add_argument(
        "-mdcm_xyz", "--mdcm_xyz", help="mdcm xyz file", default=None, type=str
    )
    parser.add_argument("-fname", "--fname", help="name of the molecule")
    parser.add_argument(
        "-uuid", "--uuid", help="uuid of the molecule", default=None, type=str
    )
    parser.add_argument(
        "-eval_prev", "--eval_prev", help="uuid of last step", default=False, type=str
    )

    args = parser.parse_args()
    print(" ".join(f"{k}={v}\n" for k, v in vars(args).items()))

    if args.local_pos is not None:
        local = pd.read_pickle(args.local_pos)
    else:
        print("WARNING: No local positions specified")
        local = None

    if args.mdcm_clcl is not None:
        mdcm_clcl = args.mdcm_clcl
    else:
        print("WARNING: No MDCM clcl file specified")

    if args.mdcm_xyz is not None:
        mdcm_xyz = args.mdcm_xyz
    else:
        mdcm_xyz = None
        print("WARNING: No MDCM xyz file specified")

    if args.esp is not None:
        esp = args.esp
    else:
        raise ValueError("No ESP file specified")

    if args.dens is not None:
        dens = args.dens
    else:
        raise ValueError("No density file specified")

    if args.outdir is not None:
        outdir = args.outdir
    else:
        print("WARNING: No output directory specified")

    if args.nodes_to_avg is not None:
        i = args.nodes_to_avg
    else:
        i = None
        print("WARNING: No nodes to average specified")

    if args.fname is not None:
        fname = args.fname
    else:
        raise ValueError("No fname specified")

    if args.uuid is not None:
        uuid = args.uuid
    else:
        uuid = "uuid_not_specified"

    if args.nodes_to_avg is not None:
        ESPF = [esp.format(i)]
        DENSF = [dens.format(i)]
    else:
        ESPF = [esp]
        DENSF = [dens]

    mdcm = mdcm_set_up(
        ESPF, DENSF, mdcm_cxyz=mdcm_xyz, mdcm_clcl=mdcm_clcl, local_pos=local
    )

    clcl = mdcm.mdcm_clcl
    print(clcl)
    mdcm.write_cxyz_files()
    clcl = mdcm.mdcm_clcl
    print(clcl)

    if args.opt:
        print(f"Optimizing: {args.outdir}")
        outname = esp.format(i).split("/")[-1]
        optimize_mdcm(mdcm, clcl, args.outdir, args.l2, fname, esp,
                      eval_prev=args.eval_prev)
    else:
        rmse = mdcm.get_rmse()
        #  save the global charges to a file
        mdcm.write_cxyz_files()
        cxyz = mdcm.mdcm_cxyz
        outfn = esp + "_" + fname + "_" + uuid + ".xyz"
        # rename the file
        os.rename(esp + ".mdcm.xyz", outfn)
        print("Saved global charges to:", outfn)
        print("RMSE:", rmse)
