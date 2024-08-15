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
from .predictors.evmutation import EVMutationWrapper
from .predictors.pretrained_transformers import model_device_context
from .predictors.saprot import SaProtLikelihoodWrapper
from .predictors.vespa import VESPAWrapper

from .embedders.esm2 import ESM2Embedding
from .embedders.ohe import OneHotAlignedEmbedding, OneHotProteinEmbedding
from .embedders.msa_transformer import MSATransformerEmbedding
from .embedders.saprot import SaProtEmbedding


TOOLS = [
    HMMWrapper,
    ESM2LikelihoodWrapper,
    MSATransformerLikelihoodWrapper,
    EVMutationWrapper,
    SaProtLikelihoodWrapper,
    VESPAWrapper,


    # embedders
    ESM2Embedding,
    OneHotAlignedEmbedding,
    OneHotProteinEmbedding,
    MSATransformerEmbedding,
    SaProtEmbedding
]
