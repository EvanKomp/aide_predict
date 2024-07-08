# aide_predict/bespoke_models/__init__.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
'''
from .base import ProteinModelWrapper
from .predictors.hmm import HMMWrapper
from .predictors.esm2 import ESM2LikelihoodWrapper

from .embedders.esm2 import ESM2Embedding
from .embedders.ohe import OneHotAlignedEmbedding, OneHotProteinEmbedding

TOOLS = [
    ProteinModelWrapper,
    HMMWrapper,
    ESM2LikelihoodWrapper,

    # embedders
    ESM2Embedding,
    OneHotAlignedEmbedding,
    OneHotProteinEmbedding
]
