# aide_predict/bespoke_models/__init__.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
'''
from .base import ProteinModelWrapper
from .predictors.hmm import HMMWrapper
from .predictors.esm2 import ESM2LikelihoodWrapper
from .predictors.msa_transformer import MSATransformerLikelihoodWrapper

from .embedders.esm2 import ESM2Embedding
from .embedders.ohe import OneHotAlignedEmbedding, OneHotProteinEmbedding
from .embedders.msa_transformer import MSATransformerEmbedding


TOOLS = [
    ProteinModelWrapper,
    HMMWrapper,
    ESM2LikelihoodWrapper,
    MSATransformerLikelihoodWrapper,

    # embedders
    ESM2Embedding,
    OneHotAlignedEmbedding,
    OneHotProteinEmbedding,
    MSATransformerEmbedding
]
