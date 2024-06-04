import json
from pathlib import Path
from kmdcm.pydcm.dcm import FFE_PATH

print("FFEPATH:", FFE_PATH)

if __name__ == "__main__":
    """
    This script is used to generate the json files for the DCM models
    Example for Water
    """
    water_dict = {}
    water_mdcm_path = Path(__file__).parents[0].resolve() / "sources" / "water"
    water_cubes_path = FFE_PATH / "cubes" / "water_pbe0"
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
    json.dump(water_dict, open(FFE_PATH / "tests/water_pbe0.json", "w"))

    """
    Methanol
    """
    methanol_dict = {}
    methanol_mdcm_path = Path(__file__).parents[0] / "sources" / "methanol"
    methanol_cubes_path = FFE_PATH / "cubes" / "shaked-methanol"
    methanol_dict["scan_fesp"] = [
        str(_) for _ in list(methanol_cubes_path.glob("*.p.cube"))
    ]
    methanol_dict["scan_fdns"] = [
        str(_) for _ in list(methanol_cubes_path.glob("*.d.cube"))
    ]
    methanol_dict["mdcm_cxyz"] = str(methanol_mdcm_path / "refined.xyz")
    methanol_dict["mdcm_clcl"] = str(methanol_mdcm_path / "meoh_pbe0dz.mdcm")
    json.dump(methanol_dict, open(FFE_PATH / "tests/shaked-methanol.json", "w"))
