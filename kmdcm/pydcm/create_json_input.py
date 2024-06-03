import json
from pathlib import Path, PosixPath
from kmdcm.pydcm.dcm import FFE_PATH, espform, densform

FFEPATH = FFE_PATH #Path(__file__).parent.parent.parent

print("FFEPATH", FFEPATH)

if __name__ == "__main__":
    """
    $$$                                                             $$$
    This script is used to generate the json files for the DCM models
    $$$                                                            $$$
    Water
    """
    water_dict = {}
    water_mdcm_path = Path(__file__).parents[0].resolve() / "sources" / "water"
    water_cubes_path = FFEPATH / "cubes" / "water_pbe0"
    print(water_cubes_path)
    water_dict["scan_fesp"] = [
        str(_) for _ in list(water_cubes_path.glob("*.p.cube"))
    ]
    print(water_dict["scan_fesp"])
    water_dict["scan_fdns"] = [
        str(_) for _ in list(water_cubes_path.glob("*.d.cube"))
    ]
    water_dict["mdcm_cxyz"] = str(water_mdcm_path / "refined.xyz")
    water_dict["mdcm_clcl"] = str(water_mdcm_path / "pbe0_dz.mdcm")
    json.dump(water_dict, open(FFEPATH / "tests/water_pbe0.json", "w"))
