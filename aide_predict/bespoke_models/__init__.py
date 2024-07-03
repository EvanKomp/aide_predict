# aide_predict/bespoke_models/__init__.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
'''
from .base import ProteinModelWrapper
from .predictors.hmm import HMMWrapper

TOOLS = [
    ProteinModelWrapper,
    HMMWrapper,
]
