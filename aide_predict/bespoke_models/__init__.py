# aide_predict/bespoke_models/__init__.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
'''
from .predictors.hmm import HMMWrapper
from .predictors.esm2 import ESM2LikelihoodWrapper
from .predictors.msa_transformer import MSATransformerLikelihoodWrapper
from .predictors.evmutation import EVMutationWrapper
from .predictors.pretrained_transformers import model_device_context
from .predictors.saprot import SaProtLikelihoodWrapper
from .predictors.vespa import VESPAWrapper
from .predictors.eve import EVEWrapper
from .predictors.ssemb import SSEmbWrapper

from .embedders.esm2 import ESM2Embedding
from .embedders.ohe import OneHotAlignedEmbedding, OneHotProteinEmbedding
from .embedders.msa_transformer import MSATransformerEmbedding
from .embedders.saprot import SaProtEmbedding
from .embedders.kmer import KmerEmbedding
from .embedders.ssemb import SSEmbEmbedding


TOOLS = [
    HMMWrapper,
    ESM2LikelihoodWrapper,
    MSATransformerLikelihoodWrapper,
    EVMutationWrapper,
    SaProtLikelihoodWrapper,
    VESPAWrapper,
    EVEWrapper,
    SSEmbWrapper,

    # embedders
    ESM2Embedding,
    OneHotAlignedEmbedding,
    OneHotProteinEmbedding,
    MSATransformerEmbedding,
    SaProtEmbedding,
    KmerEmbedding,
    SSEmbEmbedding,
]
