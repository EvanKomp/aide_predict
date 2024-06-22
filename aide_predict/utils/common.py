# aide_predict/utils/common.py
'''
* Author: Evan Komp
* Created: 6/11/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Common utility functions

'''
from aide_predict.utils.constants import AA_ALPHABET

from typing import Iterable
from types import SimpleNamespace


def convert_dvc_params(dvc_params_dict: dict):
    """DVC Creates a nested dict with the parameters.
    
    We want an object that has nested attributes so that we can
    access parameters with dot notation.
    """
    def _dict_to_obj(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _dict_to_obj(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [_dict_to_obj(x) for x in d]
        else:
            return d
    return _dict_to_obj(dvc_params_dict)
