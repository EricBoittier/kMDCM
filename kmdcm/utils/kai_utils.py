import numpy as np

# --------------- ** Conversion parameters ** ---------------

# Avaiable distance and energy units
R_units = ["Bohr", "Ang"]
E_units = ["Ha", "eV", "kJ/mol", "kcal/mol"]

# Conversation factor: Bohr to Angstrom
a02A = 5.291772109e-01
# Conversation factor: Hartree to eV
Ha2eV = 2.721138505e01
# Conversation factor: Hartree to kJ/mol
Ha2kJmol = 2.625499639e03
# Conversation factor: Hartree to kcal/mol
Ha2kcalmol = 6.275096080e02


def get_conv_R(unit):
    if unit == "Bohr":
        return 1.0
    elif unit == "Ang":
        return a02A
    else:
        msg = "Distance unit is unknown: {unit:s}?"
        raise IOError(msg)


def get_conv_E(unit):
    if unit == "Ha":
        return 1.0
    elif unit == "eV":
        return Ha2eV
    elif unit == "kJ/mol":
        return Ha2kJmol
    elif unit == "kcal/mol":
        return Ha2kcalmol
    else:
        msg = "Energy unit is unknown: {unit:s}?"
        raise IOError(msg)


# --------------- ** Checking input types ** ---------------

# Integer types
dint_all = (int, np.int, np.int16, np.int32, np.int64)
# Float types
dflt_all = (float, np.float, np.float16, np.float32, np.float64)
# Combined to numeric types
dnum_all = dint_all + dflt_all
# List/array types
darr_all = (tuple, list, np.ndarray)


def is_string(x):
    return isinstance(x, str)


def is_bool(x):
    return isinstance(x, bool)


def is_numeric(x):
    return type(x) in dnum_all and not is_bool(x)


def is_integer(x):
    return type(x) in dint_all and not is_bool(x)


def is_dictionary(x):
    return isinstance(x, dict)


def is_array_like(x):
    return isinstance(x, darr_all)


def is_numeric_array(x):
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            np.asarray(x, dtype=float)
            return True
        except (ValueError, TypeError):
            return False
    return False


def is_integer_array(x):
    if is_numeric_array(x):
        if (np.asarray(x, dtype=float) == np.asarray(x, dtype=int)).all():
            return True
    return False


def is_string_array(x):
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            np.asarray(x, dtype=str)
            return True
        except (ValueError, TypeError):
            return False
    return False


def is_None_array(x):
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            return (np.array(x) is None).all()
        except (ValueError, TypeError):
            return False
    return False


# --------------- ** Save and Load Config ** ---------------


def save_config(configfile, configdict, keywords=None, marker="--", marker2="  $"):
    """
    Save config file content to text file
    """

    # Key order
    if keywords is None:
        keywords = configdict.keys()

    # Initialize string
    configtxt = ""

    # Determine max key length
    maxlenkey = np.max([len(key) for key in keywords])
    keyformat = "{:" + "{0:d}".format(maxlenkey + len(marker)) + "s}"

    # Write config dictionary content
    for key in keywords:
        # Get item
        item = configdict[key]

        # Write keyword and item
        configtxt += write_content(key, item, marker, marker2, keyformat=keyformat)

    # Write config text to file
    with open(configfile, "w") as fconfig:
        fconfig.write(configtxt)


def write_content(key, item, marker, marker2, keyformat="{:s}"):
    """
    Write item line
    """

    line = keyformat.format(marker + key)
    if is_string(item):
        line += " {:s}\n".format(item)
    elif is_numeric(item):
        if is_integer(item):
            line += " {:d}\n".format(item)
        else:
            line += " {:1.6E}\n".format(item)
    elif is_bool(item):
        if item:
            line += " True\n"
        else:
            line += " False\n"
    elif item is None:
        line += " None\n"
    elif is_array_like(item):
        if len(item) == 0:
            line += " None\n"
        else:
            for item1 in item:
                if is_string(item1):
                    line += " {:s}".format(item1)
                elif is_numeric(item1):
                    if is_integer(item1):
                        line += " {:d}".format(item1)
                    else:
                        line += " {:1.6E}".format(item1)
                elif is_bool(item1):
                    if item1:
                        line += " True"
                    else:
                        line += " False"
                elif item1 is None:
                    line += " None"
                elif is_array_like(item1):
                    line += "\n"
                    for item2 in item1:
                        if is_string(item2):
                            line += " {:s}".format(item2)
                        elif is_numeric(item2):
                            if is_integer(item2):
                                line += " {:d}".format(item2)
                            else:
                                line += " {:1.6E}".format(item2)
                        elif is_bool(item2):
                            if item2:
                                line += " True"
                            else:
                                line += " False"
                        elif item2 is None:
                            line += " None"
                        elif is_array_like(item2):
                            raise IOError("Only max 2D arrays can be written")
            line += "\n"
    elif is_dictionary(item):
        line += "\n"
        for keyi, itemi in item.items():
            line += "  " + write_content(keyi, itemi, marker2, marker2)
    else:
        line += " " + str(item) + "\n"

    return line


def load_content(configfile, marker="--", marker2="$"):
    """
    Load config file content from text file
    """

    # Read config text from file
    with open(configfile, "r") as f:
        clns = f.readlines()

    # Initialize config dictionary
    config = {}

    # Loop over file lines
    not_done = True
    ilne = 0
    key = ""
    sitm = []
    while not_done:
        # Skip if line is empty else split by spaces
        if len(clns[ilne]) == 0:
            ilne += 1
            if ilne == len(clns):
                not_done = False
                config[key] = read_item(sitm)
                sitm = []
            continue
        else:
            spln = clns[ilne].split()
            # Check if columns of equal signs are used and split them too
            spln = [spli for splt in spln for spli in splt.split(":")]
            spln = [spli for splt in spln for spli in splt.split("=")]
            spln = [spli for splt in spln for spli in splt.split(",")]

        if spln[0][: len(marker)] == marker:
            # If key is active, add to dictionary and empty item list
            if len(key):
                config[key] = read_item(sitm)
                sitm = []

            # Read key
            key = spln[0][len(marker) :]
            # Prepare item list
            if len(spln) > 1:
                sitm.append([litm for litm in spln[1:]])

        else:
            # Prepare item list
            sitm.append([litm for litm in spln])

        # Increment line counter
        ilne += 1
        if ilne == len(clns):
            not_done = False
            # Add content to config dictionary
            config[key] = read_item(sitm)
            sitm = []

    return config


def read_item(sitm):
    """
    Read key and item from line
    """

    if len(sitm) == 0:
        item = None

    elif len(sitm) == 1 and len(sitm[0]) == 1:
        item = read_element(sitm[0][0].strip())

    elif len(sitm) == 1:
        item = [read_element(itmi.strip()) for itmi in sitm[0]]

    else:
        item = []

        for item1 in item:
            item.append([read_element(itmi.strip()) for itmi in item1])

    return item


def read_element(item):
    """
    Convert item
    """

    if item == "None":
        return None
    elif item == "True":
        return True
    elif item == "False":
        return False
    elif "." in item:
        try:
            return float(item)
        except ValueError:
            return item
    else:
        try:
            return int(item)
        except ValueError:
            return item


# --------------- ** Reference databases ** ---------------

ELEMENT_NUMBER = 118 + 1  # 118 Elements + 1 undefined one (0)

ATOMIC_NUMBER = {
    "": 0,
    "h": 1,
    "he": 2,
    "li": 3,
    "be": 4,
    "b": 5,
    "c": 6,
    "n": 7,
    "o": 8,
    "f": 9,
    "ne": 10,
    "na": 11,
    "mg": 12,
    "al": 13,
    "si": 14,
    "p": 15,
    "s": 16,
    "cl": 17,
    "ar": 18,
    "k": 19,
    "ca": 20,
    "sc": 21,
    "ti": 22,
    "v": 23,
    "cr": 24,
    "mn": 25,
    "fe": 26,
    "co": 27,
    "ni": 28,
    "cu": 29,
    "zn": 30,
    "ga": 31,
    "ge": 32,
    "as": 33,
    "se": 34,
    "br": 35,
    "kr": 36,
    "rb": 37,
    "sr": 38,
    "y": 39,
    "zr": 40,
    "nb": 41,
    "mo": 42,
    "tc": 43,
    "ru": 44,
    "rh": 45,
    "pd": 46,
    "ag": 47,
    "cd": 48,
    "in": 49,
    "sn": 50,
    "sb": 51,
    "te": 52,
    "i": 53,
    "xe": 54,
    "cs": 55,
    "ba": 56,
    "la": 57,
    "ce": 58,
    "pr": 59,
    "nd": 60,
    "pm": 61,
    "sm": 62,
    "eu": 63,
    "gd": 64,
    "tb": 65,
    "dy": 66,
    "ho": 67,
    "er": 68,
    "tm": 69,
    "yb": 70,
    "lu": 71,
    "hf": 72,
    "ta": 73,
    "w": 74,
    "re": 75,
    "os": 76,
    "ir": 77,
    "pt": 78,
    "au": 79,
    "hg": 80,
    "tl": 81,
    "pb": 82,
    "bi": 83,
    "po": 84,
    "at": 85,
    "rn": 86,
    "fr": 87,
    "ra": 88,
    "ac": 89,
    "th": 90,
    "pa": 91,
    "u": 92,
    "np": 93,
    "pu": 94,
    "am": 95,
    "cm": 96,
    "bk": 97,
    "cf": 98,
    "es": 99,
    "fm": 100,
    "md": 101,
    "no": 102,
    "lr": 103,
    "rf": 104,
    "db": 105,
    "sg": 106,
    "bh": 107,
    "hs": 108,
    "mt": 109,
    "ds": 110,
    "rg": 111,
    "cn": 112,
    "nh": 113,
    "fl": 114,
    "mc": 115,
    "lv": 116,
    "ts": 117,
    "og": 118,
}

ELEMENT_NAME = {
    0: "",
    1: "h",
    2: "he",
    3: "li",
    4: "be",
    5: "b",
    6: "c",
    7: "n",
    8: "o",
    9: "f",
    10: "ne",
    11: "na",
    12: "mg",
    13: "al",
    14: "si",
    15: "p",
    16: "s",
    17: "cl",
    18: "ar",
    19: "k",
    20: "ca",
    21: "sc",
    22: "ti",
    23: "v",
    24: "cr",
    25: "mn",
    26: "fe",
    27: "co",
    28: "ni",
    29: "cu",
    30: "zn",
    31: "ga",
    32: "ge",
    33: "as",
    34: "se",
    35: "br",
    36: "kr",
    37: "rb",
    38: "sr",
    39: "y",
    40: "zr",
    41: "nb",
    42: "mo",
    43: "tc",
    44: "ru",
    45: "rh",
    46: "pd",
    47: "ag",
    48: "cd",
    49: "in",
    50: "sn",
    51: "sb",
    52: "te",
    53: "i",
    54: "xe",
    55: "cs",
    56: "ba",
    57: "la",
    58: "ce",
    59: "pr",
    60: "nd",
    61: "pm",
    62: "sm",
    63: "eu",
    64: "gd",
    65: "tb",
    66: "dy",
    67: "ho",
    68: "er",
    69: "tm",
    70: "yb",
    71: "lu",
    72: "hf",
    73: "ta",
    74: "w",
    75: "re",
    76: "os",
    77: "ir",
    78: "pt",
    79: "au",
    80: "hg",
    81: "tl",
    82: "pb",
    83: "bi",
    84: "po",
    85: "at",
    86: "rn",
    87: "fr",
    88: "ra",
    89: "ac",
    90: "th",
    91: "pa",
    92: "u",
    93: "np",
    94: "pu",
    95: "am",
    96: "cm",
    97: "bk",
    98: "cf",
    99: "es",
    100: "fm",
    101: "md",
    102: "no",
    103: "lr",
    104: "rf",
    105: "db",
    106: "sg",
    107: "bh",
    108: "hs",
    109: "mt",
    110: "ds",
    111: "rg",
    112: "cn",
    113: "nh",
    114: "fl",
    115: "mc",
    116: "lv",
    117: "ts",
    118: "og",
}
