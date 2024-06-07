# Kernel-based Minimal Distributed Charge Models (kMDCM)

<img src="images/kmdcm-methanol.gif" width="300" height="300" />

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

To optimize the charges and fit the kernels, run the following command in the "tests" directory:

### Example 1: Water
```bash
 python test_dcm.py --alpha 0.000001 --n_factor 8 --n_atoms 3  --l2 0.0 --json water_pbe0.json --fname water_pbe0 --do_opt
```
### Example 2: Methanol
```bash
python test_dcm.py --alpha 0.000001 --n_factor 16 --n_atoms 6  --l2 40.0 --json shaked-methanol.json --fname shaked-methanol --do_opt
```
### Keyword arguments
- `--alpha` is the regularization parameter
- `--n_factor` is the ratio between the test and training set
- `--n_atoms` is the number of atoms in the molecule
- `--l2` is the l2 penalty for moving the charges from their initial positions
- `--json` is the json file containing the filenames for fitting
- `--fname` is the name of the job

## Exporting kernel matrices to CHARMM
To export the kernel matrices to CHARMM, run the following command:
```bash
python kmdcm/utils/save_charmm_input.py --kernel 4af4d6a1-a66c-4339-bafa-82db5e7529fc
```
This creates a directory in the tests/coeffs which stores the kernel coefficients ("coefs0.txt", etc.) and the reference distances ("x_fit.txt").

```apex
156 10 3 6
acec8c45-bfe5-4acc-8574-925372ecb40d/x_fit.txt
acec8c45-bfe5-4acc-8574-925372ecb40d/coefs0.txt
...
acec8c45-bfe5-4acc-8574-925372ecb40d/coefs29.txt
```

## Running kMDCM in CHARMM
```fortran
! 
open unit 12 card read name @input/methanol.kern
! regular MDCM input file
open unit 10 card read name @input/10charges.dcm
DCM KERN 12 IUDCM 10 TSHIFT 
'''