---
title: Resource benchmarking
---
# Resource testing

See below for the cost it takes to run each tool. This test was run with a Dual socket Intel Xeon Sapphire Rapids 52 core CPU. When the model supports GPU, one NVIDIA H100 was provided.

The test system was a GFP (238) amino acids, MSA depth (when applicable) was 201. Times measure the total time to fit the model (when applicable) and run prediction on 50 variants. Missing values are either because the core model does not support that type of prediction or because AIDE's wrapper does not support it.

NOTE: The cost of some of these (*) is significantly impacted by hyperparameters.

## Zero shot predictors

| Model Name | Marginal Method | GPU Total Time (s) | CPU Total Time (s) |
|------------|----------------|-------------------|-------------------|
| HMMWrapper | - | - | 0.136 |
| ESM2LikelihoodWrapper | wildtype_marginal | 0.980 | 2.560 |
| ESM2LikelihoodWrapper | mutant_marginal | 0.534 | 30.837 |
| ESM2LikelihoodWrapper | masked_marginal | 0.718 | 62.507 |
| MSATransformerLikelihoodWrapper | wildtype_marginal | 4.067 | 33.974 |
| MSATransformerLikelihoodWrapper | mutant_marginal | 57.297 | Timeout (>1800s) |
| MSATransformerLikelihoodWrapper | masked_marginal | 110.086 | Timeout (>1800s) |
| EVMutationWrapper | - | - | 96.697 |
| SaProtLikelihoodWrapper | wildtype_marginal | 5.356 | 24.291 |
| SaProtLikelihoodWrapper | mutant_marginal | 7.326 | 220.906 |
| SaProtLikelihoodWrapper | masked_marginal | 14.814 | 429.626 |
| VESPAWrapper | - | 244.852 | - |
| EVEWrapper * | - | 925.930 | - |
| SSEmbWrapper | - | 192.999 | - |

## Embedders

Cost for embedding 21 GFP sequences.

| Model Name | GPU Total Time (s) | CPU Total Time (s) |
|------------|-------------------|-------------------|
| ESM2Embedding | 0.887 | 1.477 |
| OneHotAlignedEmbedding | - | 0.092 |
| OneHotProteinEmbedding | - | 0.023 |
| MSATransformerEmbedding | 18.962 | 62.653 |
| SaProtEmbedding | 10.439 | 32.360 |
| KmerEmbedding | - | 0.005 |
| SSEmbEmbedding | 665.772 | - |



