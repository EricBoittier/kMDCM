# Basics
import os
import subprocess
import numpy as np

# Utilities
# from .utils import *
from kmdcm.utils.kai_utils import save_config, load_content, is_array_like

# Atomic Simulation Environment
import ase.io

# Miscellaneous
import time
from itertools import product


class Scan:
    """
    The Scan class provides tools to perform a scan over a defined set of
    degrees of freedom for arbitrary molecules with QM programs like Gaussian,
    ...
    """

    def __init__(self, config=None):
        """
        Check input
        """

        # Default values of optional parameters
        self.default = {
            "system_file_type": "xyz",
            "system_total_charge": 0,
            "system_spin_multiplicity": 1,
            "scan_dofs": None,
            "scan_steps": None,
            "scan_constrained_opt": False,
            "scan_write_esp_cube": True,
            "scan_write_dens_cube": True,
            "scan_qm_program": "Gaussian",
            "scan_qm_method": "HF",
            "scan_qm_basis_set": "cc-pVDZ",
            "scan_parallel_tasks": 1,
            "scan_cpus_per_task": 1,
            "scan_memory_per_task": 1000,
            "scan_overwrite": False,
            "scan_time_check_tasks": 60,
        }

        # Keywords and definition
        self.keywords = {
            "system_label": (
                "type(str):\n"
                + "  Label tag of your system for file identification.\n"
                + "  Each written file includes this label."
            ),
            "system_file_coord": (
                "type(str):\n" + "  Path to a cartesian 'xyz' or z-matrix 'zmat' file."
            ),
            "system_file_type": (
                "type(str): ('xyz', 'zmat'), default 'xyz'\n"
                + "  Type identifier of your coordinate file as\n"
                + "  cartesian 'xyz' or z-matrix 'zmat' file."
            ),
            "system_total_charge": (
                "type(int), optional: default '{:s}'\n".format(
                    str(self.default["system_total_charge"])
                )
                + "  Total charge of your system.\n"
                + "  Only integer charges are possible."
            ),
            "system_spin_multiplicity": (
                "type(int), optional: default '{:s}'\n".format(
                    str(self.default["system_spin_multiplicity"])
                )
                + "  Spin multiplicity of your system.\n"
                + "  The multiplicity is related to your total spin S by:\n"
                + "    multiplicity = 2*S + 1"
            ),
            "scan_dofs": (
                "type(list), optional: default '{:s}'\n".format(
                    str(self.default["scan_dofs"])
                )
                + "  List defining the degrees of freedom to scan over.\n"
                + "  In case of 'xyz' coordination file format, the list\n"
                + "  must contain sublists of atom indices. The length of the\n"
                + "  sublist defines the type 'bond', 'angle', 'dihedral'.\n"
                + "  To match the parameter value, only the respectively last\n"
                + "  atom in the list is moved.\n"
                + "    Example:\n"
                + "      scan_dofs = [\n"
                + "          [0, 1],        # <- Bond between atom 0 and 1\n"
                + "          [1, 0, 2],     # <- Angle between atoms 1<-0->2\n"
                + "          [2, 0, 1, 3]   # <- Dihedral angle\n"
                + "      ]"
                + "\n"
                + "  In case of 'zmat' coordination file format, the list\n"
                + "  must contain sublists of the respective variable label\n"
                + "  in the zmat file and the type definition.\n"
                + "    Example:\n"
                + "      Z-matrix file:\n"
                + "        O\n"
                + "        O 1 dOO\n"
                + "        H 1 dHO1 2 aHOO1\n"
                + "        H 2 dHO2 1 aHOO2 3 dihHOOH\n"
                + "\n"
                + "        dOO     = 1.2\n"
                + "        dHO1    = 1.0\n"
                + "        dHO2    = 1.0\n"
                + "        aHOO1   = 60.0\n"
                + "        aHOO2   = 60.0\n"
                + "        dihHOOH = 180.0\n"
                + "      ----\n"
                + "      scan_dofs = [\n"
                + "          ['dOO', 'bond'],\n"
                + "          ['aHOO1', 'angle'],\n"
                + "          ['dihHOOH', 'dihdral'],\n"
                + "      ]"
                + "\n"
                + "  If scan_dofs is not defined or as None, only the system\n"
                + "  conformation in the coordinate file is evaluated."
            ),
            "scan_steps": (
                "type(list), optional: default '{:s}'\n".format(
                    str(self.default["scan_steps"])
                )
                + "  List defining the grid steps for the degrees of freedom\n"
                + "  defined in 'scan_dofs'. Bonds are given in Angstrom and\n"
                + "  angular parameter in degree.\n"
                + "    Example (see example coordinates in 'scan_dofs'):\n"
                + "      scan_steps = [\n"
                + "          [0.9, 1.0, 1.1],    # <- Bond distances\n"
                + "          [50, 60, 70, 80],   # <- Angles\n"
                + "          [0., 90., 180.]     # <- Dihedral angles\n"
                + "      ]"
                + "\n"
                + "  If scan_steps is not defined or as None, only the system\n"
                + "  conformation in the coordinate file is evaluated."
            ),
            "scan_constrained_opt": (
                "type(bool), optional: default '{:s}'\n".format(
                    str(self.default["scan_constrained_opt"])
                )
                + "  Whether if perform constrained optimization for each\n"
                + "  grid point in the scan or just single point."
            ),
            "scan_write_esp_cube": (
                "type(bool), optional: default '{:s}'\n".format(
                    str(self.default["scan_write_esp_cube"])
                )
                + "  Write a cube file with the electrostatic potential."
            ),
            "scan_write_dens_cube": (
                "type(bool), optional: default '{:s}'\n".format(
                    str(self.default["scan_write_dens_cube"])
                )
                + "  Write a cube file with the electron density."
            ),
            "scan_qm_program": (
                "type(str), optional: default '{:s}'\n".format(
                    str(self.default["scan_qm_program"])
                )
                + "  QM calculation program for evaluating potential,\n"
                + "  electron density and ESP."
            ),
            "scan_qm_method": (
                "type(str), optional: default '{:s}'\n".format(
                    str(self.default["scan_qm_method"])
                )
                + "  QM method, see documentation for available methods with\n"
                + "  the respective QM program."
            ),
            "scan_qm_basis_set": (
                "type(str), optional: default '{:s}'\n".format(
                    str(self.default["scan_qm_basis_set"])
                )
                + "  QM atom centred basis set, see documentation for\n"
                + "  available basis set in the respective QM program."
            ),
            "scan_parallel_tasks": (
                "type(int), optional: default '{:s}'\n".format(
                    str(self.default["scan_parallel_tasks"])
                )
                + "  Number of parallel QM calculations on your machine."
            ),
            "scan_cpus_per_task": (
                "type(int), optional: default '{:s}'\n".format(
                    str(self.default["scan_cpus_per_task"])
                )
                + "  Number of available CPUs per task."
            ),
            "scan_memory_per_task": (
                "type(int), optional: default '{:s}'\n".format(
                    str(self.default["scan_memory_per_task"])
                )
                + "  Amount of requested memory per single CPU in MB.\n"
                + "  In general, declared memory available in the QM programm\n"
                + "  is usually just 90% of the total memory declared in the\n"
                + "  bash script file."
            ),
            "scan_overwrite": (
                "type(bool), optional: default '{:s}'\n".format(
                    str(self.default["scan_overwrite"])
                )
                + "  Whether to recompute the system at grid points even if \n"
                + "  all output files are given and viable (successful run)."
            ),
            "scan_time_check_tasks": (
                "type(int), optional: default '{:s}'\n".format(
                    str(self.default["scan_time_check_tasks"])
                )
                + "  Time delay in seconds to check the status of the tasks\n"
                + "  and submit next ones."
            ),
        }

        if config is not None:
            self.initialize_scan(config)

        else:
            self.config = None

    def print_doc(self, keys=None):
        if keys is None:
            keys = self.keywords.keys()

        # Print documentation of each keyword in config dictionary
        for key in keys:
            print(key)
            print("-" * (len(key) + 2))
            print(self.keywords[key])
            if self.config is not None:
                if key in self.config.keys():
                    print("Your input: ", self.config[key])
            print()

    def initialize_scan(self, config):
        """
        Initialize system and scan parameters defined by a dictionary or
        file containing required variables.
        """

        # Check config input
        if isinstance(config, str):
            # Read config file
            config = load_content(config)

        if isinstance(config, dict):
            # Check dictionary keys
            unknown = False
            for key in config.keys():
                if key not in self.keywords.keys():
                    msg = "Keyword '{:s}' not recognized.\n".format(key)
                    print(msg)

                    unknown = True

            # If unknown key in config, print key list
            if unknown:
                print("Perhaps you have mixed up on of these:\n")

                for key, item in self.keywords.items():
                    print(key)
                    # print("-"*(len(key) + 2))
                    # print(item)
                    # if key in config.keys():
                    # print("Your input: ", config[key])
                    # print()

        # Save config dictionary
        self.config = config

        # System:
        # ---------

        # System label tag
        self.syst_ltag = config["system_label"]

        # Coordinate file
        self.syst_fcrd = config["system_file_coord"]

        # Coordinate file type ("xyz" or "zmat")
        if "system_file_type" in config.keys():
            self.syst_tcrd = config["system_file_type"]
        else:
            self.syst_tcrd = self.default["system_file_type"]

        # Charge and spin state (singlet: 1, doublet: 2, ...)
        if "system_total_charge" in config.keys():
            self.syst_chrg = config["system_total_charge"]
        else:
            self.syst_chrg = self.default["system_total_charge"]
        if "system_spin_multiplicity" in config.keys():
            self.syst_spin = config["system_spin_multiplicity"]
        else:
            self.syst_spin = self.default["system_spin_multiplicity"]

        # Scan:
        # -------

        # Scan along atom indices (start counting with 0)
        if "scan_dofs" in config.keys():
            # Check if scan_dofs is 2D list or None
            if is_array_like(config["scan_dofs"]):
                if is_array_like(config["scan_dofs"][0]):
                    self.scan_dofs = config["scan_dofs"]
                else:
                    self.scan_dofs = [config["scan_dofs"]]
            else:
                self.scan_dofs = [config["scan_dofs"]]
        else:
            self.scan_dofs = self.default["scan_dofs"]

        # Scan steps defined by nested lists
        if "scan_steps" in config.keys():
            # Check if scan_steps is 2D list or None
            if is_array_like(config["scan_steps"]):
                if is_array_like(config["scan_steps"][0]):
                    self.scan_stps = config["scan_steps"]
                else:
                    self.scan_stps = [config["scan_steps"]]
            else:
                self.scan_stps = [config["scan_steps"]]
        else:
            self.scan_stps = self.default["scan_steps"]

        # Constrained optimization at each grid point
        if "scan_constrained_opt" in config.keys():
            self.scan_copt = config["scan_constrained_opt"]
        else:
            self.scan_copt = self.default["scan_constrained_opt"]

        # Write electrostatic potential or electron density cube file
        if "scan_write_esp_cube" in config.keys():
            self.scan_espc = config["scan_write_esp_cube"]
        else:
            self.scan_espc = self.default["scan_write_esp_cube"]
        if "scan_write_dens_cube" in config.keys():
            self.scan_dnsc = config["scan_write_dens_cube"]
        else:
            self.scan_dnsc = self.default["scan_write_dens_cube"]

        # Define the quantum-electronic program (only Gaussian so far)
        if "scan_qm_program" in config.keys():
            self.scan_prgm = config["scan_qm_program"]
        else:
            self.scan_prgm = self.default["scan_qm_program"]

        # Define method and basis set
        if "scan_qm_method" in config.keys():
            self.scan_mthd = config["scan_qm_method"]
        else:
            self.scan_mthd = self.default["scan_qm_method"]
        if "scan_qm_basis_set" in config.keys():
            self.scan_bsst = config["scan_qm_basis_set"]
        else:
            self.scan_bsst = self.default["scan_qm_basis_set"]

        # Number of parallel tasks
        if "scan_parallel_tasks" in config.keys():
            self.scan_tsks = config["scan_parallel_tasks"]
        else:
            self.scan_tsks = self.default["scan_parallel_tasks"]

        # CPUs per task
        if "scan_cpus_per_task" in config.keys():
            self.scan_cpus = config["scan_cpus_per_task"]
        else:
            self.scan_cpus = self.default["scan_cpus_per_task"]

        # Memory per cpu in MB
        if "scan_memory_per_task" in config.keys():
            self.scan_memr = config["scan_memory_per_task"]
        else:
            self.scan_memr = self.default["scan_memory_per_task"]

        # Overwrite flag
        if "scan_overwrite" in config.keys():
            self.scan_ovrw = config["scan_overwrite"]
        else:
            self.scan_ovrw = self.default["scan_overwrite"]

        # Waiting time between task id checks
        if "scan_time_check_tasks" in config.keys():
            self.scan_tslp = config["scan_time_check_tasks"]
        else:
            self.scan_tslp = self.default["scan_time_check_tasks"]

        # Directories, template files and tags:
        # (Does not have to be adjusted)
        # ---------------------------------------

        # Directories

        # Main directory
        self.dirs_main = os.getcwd()

        # Module directory
        self.dirs_modl = "/" + os.path.join(*os.path.realpath(__file__).split("/")[:-1])

        # Template directory
        self.dirs_tmpl = "templates"

        # Data directory
        self.dirs_data = "data"

        # Working directory
        self.dirs_work = "input"

        # Template files

        # Scan running script template
        self.tmps_srun = "{:s}_template.sh"

        # Scan program input file template
        self.tmps_sinp = "{:s}_template.com"

        # Working files

        # Scan running script template
        self.work_srun = "{:s}_{:d}_{:s}.sh"

        # Step file of scan program input
        self.work_sinp = "{:s}_{:d}_{:s}.com"

        # Step file of scan program output
        self.work_sout = "{:s}_{:d}_{:s}.out"

        # Data files

        # Configuration file
        self.data_cnfg = "config_scan_{:s}.txt"

        # Gaussian checkpoint step file
        self.data_gchk = "{:s}_{:d}_{:s}.chk"
        self.data_gfck = "{:s}_{:d}_{:s}.fchk"

        # Density and ESP cube step file
        self.data_dnsc = "{:s}_{:d}_{:s}_dens.cube"
        self.data_espc = "{:s}_{:d}_{:s}_esp.cube"

        # Make directories
        self.dirs_work = os.path.join(self.dirs_main, self.dirs_work)
        if not os.path.exists(self.dirs_work):
            os.mkdir(self.dirs_work)
        self.dirs_data = os.path.join(self.dirs_main, self.dirs_data)
        if not os.path.exists(self.dirs_data):
            os.mkdir(self.dirs_data)

        # Read reference structure
        if self.syst_tcrd == "xyz":
            # Read system
            self.syst_refr = ase.io.read(self.syst_fcrd)

            # Number of atoms
            self.scan_Natm = len(self.syst_refr)

            # Get reference step parameter
            self.scan_pref = np.zeros(len(self.scan_dofs), dtype=float)
            for istp, step_indc in enumerate(self.scan_dofs):
                if len(step_indc) == 2:
                    self.scan_pref[istp] = self.syst_refr.get_distance(
                        step_indc[0], step_indc[1]
                    )
                elif len(step_indc) == 3:
                    self.scan_pref[istp] = self.syst_refr.get_angle(
                        step_indc[0], step_indc[1], step_indc[2]
                    )
                elif len(step_indc) == 4:
                    self.scan_pref[istp] = self.syst_refr.get_dihedral(
                        step_indc[0], step_indc[1], step_indc[2], step_indc[3]
                    )
                else:
                    print(step_indc)
                    raise IOError(
                        "Length of above  atom index list in 'scan_dofs' do not"
                        + " match requirements"
                    )

        elif self.syst_tcrd == "zmat":
            # Read z-matrix file
            with open(self.syst_fcrd, "r") as f:
                step_lzmt = f.readlines()

            # Determine matrix and parameter part

            # Skip blank lines
            ilne = 0
            while not len(step_lzmt[ilne][:-1]):
                ilne += 1
            step_iskp = ilne
            ilne += 1

            # Matrix part until first blank line
            scan_Ndmy = 0
            while len(step_lzmt[ilne][:-1]):
                if step_lzmt[ilne].split()[0] == "X":
                    scan_Ndmy += 1
                ilne += 1
            ilne += 1
            step_imtx = ilne

            # Number of atoms
            self.scan_Natm = step_imtx - step_iskp - scan_Ndmy - 1

            # TODO Z-matrix - Read structure
            # gc.write_xyz(*gc.readzmat(syst_fcrd))

        # Grid step list
        self.scan_grid = list(product(*self.scan_stps))

        # Potential list
        self.scan_epot = np.zeros(len(self.scan_grid), dtype=float)

        # Positions list
        self.scan_crds = np.zeros((len(self.scan_grid), self.scan_Natm, 3), dtype=float)

        # Density and ESP cube file lists
        self.scan_fdns = [""] * len(self.scan_grid)
        self.scan_fesp = [""] * len(self.scan_grid)

        # Initialize task id list
        self.scan_tids = []

        # Just to be independent of upper or lower case
        self.scan_prgm = self.scan_prgm.lower()

        # Respective program template files
        self.scan_trun = os.path.join(
            self.dirs_modl, self.dirs_tmpl, self.tmps_srun.format(self.scan_prgm)
        )
        self.scan_tinp = os.path.join(
            self.dirs_modl, self.dirs_tmpl, self.tmps_sinp.format(self.scan_prgm)
        )

        # Write scan configuration to file
        save_config(
            os.path.join(self.dirs_main, self.data_cnfg.format(self.syst_ltag)), config
        )

        # Check program version
        if self.scan_prgm == "gaussian":
            self.scan_vprg = "16"
        elif self.scan_prgm == "gaussian16":
            self.scan_vprg = "16"
        elif self.scan_prgm == "gaussian09":
            self.scan_vprg = "09"

    def prepare_files(self, istp):
        """
        Central function that return absolute file paths
        """

        # Prepare file names
        step_srun = self.work_srun.format(self.scan_prgm, istp, self.syst_ltag)
        step_sinp = self.work_sinp.format(self.scan_prgm, istp, self.syst_ltag)
        step_sout = self.work_sout.format(self.scan_prgm, istp, self.syst_ltag)
        step_schk = self.data_gchk.format(self.scan_prgm, istp, self.syst_ltag)
        step_sfck = self.data_gfck.format(self.scan_prgm, istp, self.syst_ltag)
        step_sdns = self.data_dnsc.format(self.scan_prgm, istp, self.syst_ltag)
        step_sesp = self.data_espc.format(self.scan_prgm, istp, self.syst_ltag)

        return (
            step_srun,
            step_sinp,
            step_sout,
            step_schk,
            step_sfck,
            step_sdns,
            step_sesp,
        )

    def prepare_input(self, istp):
        """
        Prepare input files
        """

        # Respective step in scan grid
        stpi = self.scan_grid[istp]

        if self.syst_tcrd == "xyz":
            # Copy working reference system
            step_syst = self.syst_refr.copy()

        elif self.syst_tcrd == "zmat":
            # Read z-matrix file
            with open(self.syst_fcrd, "r") as f:
                step_lzmt = f.readlines()

            # Determine matrix and parameter part

            # Skip blank lines
            ilne = 0
            while not len(step_lzmt[ilne][:-1]):
                ilne += 1
            step_iskp = ilne
            ilne += 1

            # Matrix part until first blank line
            while len(step_lzmt[ilne][:-1]):
                ilne += 1
            ilne += 1
            step_imtx = ilne

            # Parameter part until next blank line
            while len(step_lzmt[ilne][:-1]):
                ilne += 1
                if ilne == len(step_lzmt):
                    break
            step_ipar = ilne

        # Prepare file names
        (
            step_srun,
            step_sinp,
            step_sout,
            step_schk,
            step_sfck,
            step_sdns,
            step_sesp,
        ) = self.prepare_files(istp)

        # Check if results already exists
        step_done = True
        if not os.path.exists(os.path.join(self.dirs_data, step_sout)):
            step_done = False
        if self.scan_espc:
            if not os.path.exists(os.path.join(self.dirs_data, step_sesp)):
                step_done = False
        if self.scan_dnsc:
            if not os.path.exists(os.path.join(self.dirs_data, step_sdns)):
                step_done = False

        if self.syst_tcrd == "xyz":
            # Iterate over scan parameter
            for indx, step_indc in enumerate(self.scan_dofs):
                # Set step parameter
                if len(step_indc) == 2:
                    step_syst.set_distance(
                        step_indc[0], step_indc[1], stpi[indx], fix=0
                    )
                elif len(step_indc) == 3:
                    if self.scan_pref[indx] == 180.0:
                        step_syst.positions[step_indc[0], 0] += 1e-4
                    step_syst.set_angle(
                        step_indc[0], step_indc[1], step_indc[2], stpi[indx]
                    )
                    if self.scan_pref[indx] == 180.0:
                        step_syst.positions[step_indc[0], 0] -= 1e-4
                elif len(step_indc) == 4:
                    step_syst.set_dihedral(
                        step_indc[0],
                        step_indc[1],
                        step_indc[2],
                        step_indc[3],
                        stpi[indx],
                    )

            # XYZ step lines
            step_lcrd = [
                "{:s}   {:>20.15f}   {:>20.15f}   {:>20.15f}\n".format(
                    step_syst.symbols[ia], *step_syst.positions[ia]
                )
                for ia in range(len(step_syst))
            ]
            step_lcrd = "".join(step_lcrd)

        elif self.syst_tcrd == "zmat":
            # Set step parameter
            for indx, step_indc in enumerate(self.scan_dofs):
                for ilne in range(step_imtx, step_ipar):
                    if step_indc[0] in step_lzmt[ilne]:
                        step_lzmt[ilne] = "{:s} = {:.4f} F\n".format(
                            step_indc[0], float(stpi[indx])
                        )

                        # Prevent linear angle
                        if step_indc[1] == "angle" and float(stpi[indx]) == 180.0:
                            step_lzmt[ilne] = "{:s} = {:.4f} F\n".format(
                                step_indc[0], stpi[indx] - 1e-4
                            )

            # Z-matrix lines
            step_lcrd = ""
            for ilne in range(step_iskp, step_ipar):
                step_lcrd += step_lzmt[ilne]
            step_lcrd = step_lcrd

        # ///////////////////////////////////////////////
        # Start Gaussian related part
        # ///////////////////////////////////////////////
        sinp = None
        if self.scan_prgm == "gaussian":
            # Read scan program template input file
            with open(self.scan_tinp, "r") as f:
                sinp = f.read()

            # Prepare parameters
            sinp = sinp.replace("%CPU%", "{:d}".format(self.scan_cpus))
            sinp = sinp.replace(
                "%MEM%", "{:d}".format(int(0.9 * self.scan_cpus * self.scan_memr))
            )
            sinp = sinp.replace("%CHK%", step_schk)
            sinp = sinp.replace("%MTD%", self.scan_mthd)
            sinp = sinp.replace("%BSS%", self.scan_bsst)
            sinp = sinp.replace("%CHG%", str(self.syst_chrg))
            sinp = sinp.replace("%SPS%", str(self.syst_spin))
            sinp = sinp.replace("%CRD%", str(step_lcrd))
            scan_modr = ""
            if self.scan_copt:
                if self.syst_tcrd == "xyz" and self.scan_vprg == "16":
                    sinp = sinp.replace("%OPT%", "opt geom(AddGIC)")

                    for indx, step_indc in enumerate(self.scan_dofs):
                        if len(step_indc) == 2:
                            scan_modr += "R({:d},{:d}) Freeze\n".format(
                                *(np.array(step_indc, dtype=int) + 1)
                            )
                        elif len(step_indc) == 3:
                            if stpi[indx] == 180.0:
                                scan_modr += "L({:d},{:d},{:d},0,-1) Freeze\n".format(
                                    *(np.array(step_indc, dtype=int) + 1)
                                )
                                scan_modr += "L({:d},{:d},{:d},0,-2) Freeze\n".format(
                                    *(np.array(step_indc, dtype=int) + 1)
                                )
                            else:
                                scan_modr += "A({:d},{:d},{:d}) Freeze\n".format(
                                    *(np.array(step_indc, dtype=int) + 1)
                                )
                        elif len(step_indc) == 4:
                            scan_modr += "D({:d},{:d},{:d},{:d}) Freeze\n".format(
                                *(np.array(step_indc, dtype=int) + 1)
                            )

                    sinp = sinp.replace("%MOD%", scan_modr)

                elif self.syst_tcrd == "xyz" and self.scan_vprg == "09":
                    #  change ModRedundant keyword to be compatible with G09
                    sinp = sinp.replace("%OPT%", "opt=ModRedundant)")
                    scan_modr = ""
                    if len(step_indc) == 2:
                        if len(step_indc) == 2:
                            scan_modr += "{:d},{:d} F\n".format(
                                *(np.array(step_indc, dtype=int) + 1)
                            )
                        elif len(step_indc) == 3:
                            if stpi[indx] == 180.0:
                                print(
                                    "linear angle... "
                                    "Gaussian is not gonna like this :("
                                )
                                scan_modr += "{:d} {:d} {:d} F\n".format(
                                    *(np.array(step_indc, dtype=int) + 1)
                                )
                                scan_modr += "{:d} {:d} {:d} F\n".format(
                                    *(np.array(step_indc, dtype=int) + 1)
                                )
                            else:
                                scan_modr += "{:d},{:d},{:d} F\n".format(
                                    *(np.array(step_indc, dtype=int) + 1)
                                )
                        elif len(step_indc) == 4:
                            scan_modr += "{:d} {:d} {:d} {:d} F\n".format(
                                *(np.array(step_indc, dtype=int) + 1)
                            )

                    sinp = sinp.replace("%MOD%", scan_modr)
                    scan_modr += "{:d} {:d} F\n".format(
                        *(np.array(step_indc, dtype=int) + 1)
                    )

                elif len(step_indc) == 3:
                    if stpi[indx] == 180.0:
                        print("linear angle... Gaussian is not gonna like this :(")
                        scan_modr += "L({:d},{:d},{:d},0,-1) Freeze\n".format(
                            *(np.array(step_indc, dtype=int) + 1)
                        )
                        scan_modr += "L({:d},{:d},{:d},0,-2) Freeze\n".format(
                            *(np.array(step_indc, dtype=int) + 1)
                        )
                    else:
                        scan_modr += "{:d} {:d} {:d} F\n".format(
                            *(np.array(step_indc, dtype=int) + 1)
                        )
                elif len(step_indc) == 4:
                    scan_modr += "{:d} {:d} {:d} {:d} F\n".format(
                        *(np.array(step_indc, dtype=int) + 1)
                    )

                sinp = sinp.replace("%MOD%", scan_modr)

            else:
                sinp = sinp.replace("%OPT%", "opt=Z-Matrix")
                sinp = sinp.replace("%MOD%", "")

        else:
            sinp = sinp.replace("%OPT%", "")
            sinp = sinp.replace("%MOD%", "")

        # Write input step file
        with open(os.path.join(self.dirs_work, step_sinp), "w") as f:
            f.write(sinp)

        # Read scan program template run file
        with open(self.scan_trun, "r") as f:
            srun = f.read()

        # Prepare parameters
        srun = srun.replace("%JOBNME%", "g_{:s}_{:d}".format(self.syst_ltag, istp))
        srun = srun.replace("%NTASKS%", "{:d}".format(self.scan_cpus))
        srun = srun.replace("%MEMCPU%", "{:d}".format(self.scan_memr))
        srun = srun.replace("%WORKDIR%", self.dirs_work)
        srun = srun.replace("%DATADIR%", self.dirs_data)
        srun = srun.replace("%INPFILE%", step_sinp)
        srun = srun.replace("%OUTFILE%", step_sout)
        srun = srun.replace("%CHKFILE%", step_schk)
        srun = srun.replace("%FCKFILE%", step_sfck)
        srun = srun.replace("%DNSFILE%", step_sdns)
        srun = srun.replace("%ESPFILE%", step_sesp)

        # Write run step file
        with open(os.path.join(self.dirs_work, step_srun), "w") as f:
            f.write(srun)

        # ///////////////////////////////////////////////
        # End Gaussian related part
        # ///////////////////////////////////////////////

        # Return execution file and flag if results exits
        return step_srun, step_done


def prepare_scan(self, steps=None):
    """
    Prepare scan to check written inputs
    """

    # Iterate over grid points
    for istp, stpi in enumerate(self.scan_grid):
        # If just certain steps are requested
        if steps is not None:
            if istp in steps:
                continue

        # Prepare input
        _, _ = self.prepare_input(istp)


def evaluate_scan(self, steps=None):
    """
    Read scan results
    """

    # Iterate over grid points
    for istp, stpi in enumerate(self.scan_grid):
        # If just certain steps are requested
        if steps is not None:
            if istp in steps:
                continue

        # Prepare file names
        (
            step_srun,
            step_sinp,
            step_sout,
            step_schk,
            step_sfck,
            step_sdns,
            step_sesp,
        ) = self.prepare_files(istp)

        # Check output
        if not os.path.exists(os.path.join(self.dirs_data, step_sout)):
            self.scan_epot[istp] = np.nan
            continue

        if self.scan_dnsc:
            if not os.path.exists(os.path.join(self.dirs_data, step_sdns)):
                continue
            else:
                self.scan_fdns[istp] = os.path.join(self.dirs_data, step_sdns)

        if self.scan_espc:
            if not os.path.exists(os.path.join(self.dirs_data, step_sesp)):
                continue
            else:
                self.scan_fesp[istp] = os.path.join(self.dirs_data, step_sesp)

        # Read output
        with open(os.path.join(self.dirs_data, step_sout), "r") as f:
            sout = f.readlines()

        # ///////////////////////////////////////////////
        # Start Gaussian related part
        # ///////////////////////////////////////////////

        if self.scan_prgm == "gaussian":
            # Get start line of result block
            step_lrst = 0
            for ilne, line in enumerate(sout):
                if "l9999.exe" in line:
                    step_lrst = ilne

            scan_stll = True
            ilne = 0
            step_rslt = ""
            while scan_stll:
                if (ilne + step_lrst) > len(sout):
                    break
                if "The archive entry for this job" in sout[step_lrst + ilne]:
                    scan_stll = False
                    break
                else:
                    step_rslt += sout[step_lrst + ilne][1:]
                    ilne += 1

            # Read potential energy
            step_rslt = step_rslt.replace("\n", "").split("\\")
            epot = np.nan
            for rslt in step_rslt:
                if rslt[:4] == "CCSD":
                    epot = float(rslt.split("=")[-1])
                elif rslt[:6] == "MP4SDQ":
                    epot = float(rslt.split("=")[-1])
                elif rslt[:3] == "MP3":
                    epot = float(rslt.split("=")[-1])
                elif rslt[:3] == "MP2":
                    epot = float(rslt.split("=")[-1])
                elif rslt[:2] == "HF":
                    epot = float(rslt.split("=")[-1])

            # Read positions
            step_syst = ase.io.read(
                os.path.join(self.dirs_data, step_sout), format="gaussian-out"
            )
            self.scan_crds[istp, :, :] = step_syst.positions[:, :]

        # ///////////////////////////////////////////////
        # End Gaussian related part
        # ///////////////////////////////////////////////

        # if np.isnan(epot):
        #     print("No result in output {:s}".format(scan_sout))
        self.scan_epot[istp] = epot


def execute_scan(self, steps=None):
    """
    Execute QM scan
    """

    # Iterate over grid points
    for istp, stpi in enumerate(self.scan_grid):
        # If just certain steps are requested
        if steps is not None:
            if istp in steps:
                continue

        # Prepare input
        step_srun, step_done = self.prepare_input(istp)

        # If step is already successfully performed, skip step
        if step_done:
            continue

        # Check number of active tasks
        while len(self.scan_tids) >= self.scan_tsks:
            # Get active tasks
            scan_tlst = subprocess.run(["squeue"], capture_output=True)
            scan_actv = [
                int(tkid.split()[0])
                for tkid in scan_tlst.stdout.decode().split("\n")[1:-1]
            ]

            # Check if task are active
            scan_aids = []
            for tkid in self.scan_tids:
                if tkid in scan_actv:
                    scan_aids.append(tkid)

            self.scan_tids = scan_aids.copy()

            # Wait if maximum number of tasks are still active
            if len(self.scan_tids) >= self.scan_tsks:
                time.sleep(self.scan_tslp)

        # Run calculation
        os.chdir(self.dirs_work)
        task = subprocess.run(["sbatch", step_srun], capture_output=True)
        self.scan_tids.append(int(task.stdout.decode().split()[-1]))
        os.chdir(self.dirs_main)

    # Wait until all jobs.py are done
    while len(self.scan_tids):
        # Get active tasks
        scan_tlst = subprocess.run(["squeue"], capture_output=True)
        scan_actv = [
            int(tkid.split()[0]) for tkid in scan_tlst.stdout.decode().split("\n")[1:-1]
        ]

        # Check if task are active
        scan_aids = []
        for tkid in self.scan_tids:
            if tkid in scan_actv:
                scan_aids.append(tkid)

        self.scan_tids = scan_aids.copy()

        # Wait if tasks are still active
        if len(self.scan_tids):
            time.sleep(self.scan_tslp)


def get_potential(self, steps=None):
    """
    Return a list of electrostatic potential cube file paths
    """

    if steps is not None:
        # If just certain steps are requested
        return self.scan_epot[self.array(steps, dtype=int)]
    else:
        # Else return everything
        return self.scan_epot


def get_files_cube_esp(self, steps=None):
    """
    Return a list of electrostatic potential cube file paths
    """

    if steps is not None:
        # If just certain steps are requested
        return list(np.array(self.scan_fesp)[self.array(steps, dtype=int)])
    else:
        # Else return everything
        return self.scan_fesp


def get_files_cube_dens(self, steps=None):
    """
    Return a list of electron density cube file paths
    """

    if steps is not None:
        # If just certain steps are requested
        return list(np.array(self.scan_fdns)[self.array(steps, dtype=int)])
    else:
        # Else return everything
        return self.scan_fdns
