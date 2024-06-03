# Kernel-based Minimal Distributed Charge Models (kMDCM)


## Installation
After cloning the repository, you can install the package by running the following commands:
```bash
python3 -m venv kmdcm-env
source kmdcm-env/bin/activate
pip install -r requirements.txt
pip install -e .
```
PyDCM requires the Fortran code to be compiled. To do so, run the following commands:
```bash
export CFLAGS="-std=c99"
cd kmdcm/pydcm
bash compile_f2py.sh
```
## Usage
First, create the input json file. An example is given in "create_json_input.py". To create the json file, run the following command:
```bash
python kmdcm/pydcm/create_json.py
```

To optimize the charges and fit the kernels, run the following command in the tests directory:
```bash
 python test_dcm.py --alpha 0.000001 --n_factor 8 --n_atoms 3  --l2 0.0 --json water_pbe0.json --fname water_pbe0 --do_opt
```

- `--alpha` is the regularization parameter
- `--n_factor` is the ratio between the test and training set
- `--n_atoms` is the number of atoms in the molecule
- `--l2` is the l2 penalty for moving the charges from their initial positions
- `--json` is the json file containing the filenames for fitting
- `--fname` is the name of the job

