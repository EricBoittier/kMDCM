import json
from kmdcm.pydcm.dcm import mdcm_set_up


class MDCM:
    def __init__(self, mdcm_dict):
        self.mdcm_dict = mdcm_dict
        self.mdcm = None
        self.local_pos = None
        self.mdcm_clcl = None
        self.mdcm_cxyz = None
        self.mdcm_xyz = None
        self.mdcm_esp = None
        self.mdcm_dens = None
        self.mdcm_esp = None

        if mdcm_dict is not None:
            if type(mdcm_dict) is str:
                # Load the dictionary from a json file
                with open(mdcm_dict, "r") as json_file:
                    self.mdcm_dict = json.load(json_file)
            elif type(mdcm_dict) is dict:
                # Unload the contents of the dictionary into the class
                for k in mdcm_dict.keys():
                    setattr(self, k, mdcm_dict[k])
            else:
                raise TypeError("mdcm_dict must be a str or dict")

    def asDict(self):
        return self.mdcm_dict

    def asJson(self, filename=None):
        if filename is None:
            return json.dumps(self.mdcm_dict)
        else:
            json.dump(self.mdcm_dict, open(filename, "w"))

    def get_mdcm(self):
        return mdcm_set_up(
            self.mdcm_esp, self.mdcm_dens, self.mdcm_xyz, self.mdcm_clcl, self.local_pos
        )


def load_from_json(filename):
    return MDCM(filename)

