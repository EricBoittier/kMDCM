import functools as ft

import numpy as np


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
