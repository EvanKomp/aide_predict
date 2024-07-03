# Protein Prediction 
[![Python Tests](https://github.com/EvanKomp/aide_predict/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/EvanKomp/aide_predict/actions/workflows/ci-tests.yml)
[![codecov](https://codecov.io/gh/EvanKomp/aide_predict/branch/main/graph/badge.svg)](https://codecov.io/gh/EvanKomp/aide_predict)

This repository serves fundementally to increase the accessibility of protein engineering tasks that fall into the following catagory:

$$\hat{y}=f(X)$$

Here, $X$ is a set of proteins, eg. their sequence and optionaly structure. $y$ is a property of the protein that is difficult to measure, such as binding affinity, stability, or catalytic activity. $\hat{y}$ is the predicted value of $y$ given $X$.

Existing models $f$ in the literature are varied, and a huge amount of work has gone into designing clever algorithms that leverage labeled and unlabeled data. For example, models differ in the following ways (non exhaustive):
- Some require supervised labels $y$, while others [do not](https://www.nature.com/articles/s41587-021-01146-5)
- Unsupervised models can be trained on [vast sets of sequences](https://ieeexplore.ieee.org/document/9477085), or [MSAs of the related proteins](https://www.nature.com/articles/s41592-018-0138-4)
- Models exist to predict the effect of mutations on a wild type sequence, or to globally predict protein properties
- Some models incorporate [structural information](https://www.biorxiv.org/content/10.1101/2024.07.01.600583v1)
- Some models are [pretrained](http://biorxiv.org/lookup/doi/10.1101/2021.07.09.450648)
- Some models are capable of position specific predictions, which can be useful for [some tasks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009853) 

The variety an nuance of each of these means that each application is a bespoke, independent codebase, and are generally inaccessible to those with little or no coding exprience. Some applications alleviate the second problem by hosting web servers. Add to this problem is a lack of standardization in API across applications, where individual code bases can be extremely poorly documented or hard to use due to hasty development to minimize time to publication.

The goals of this project are succinctly as follows:
1. __Create a generalizable, unittested, API for protein prediction tasks that is compatible with scikit learn__. This API will allow those who are familiar with the gold standard of ML libraries to conduct protein prediction tasks in much the same way you'd see on an intro to ML Medium article. Further, it makes it much easier for bespoke strategies to be accessed and compared; any new method whose authors wrap their code in the API are easily accessed by the community without spending hours studying the codebase.
2. __Use API components to create a DVC tracked pipeline for protein prediction tasks__. This pipeline will allow for those with zero software experience to conduct protein prediction tasks with a few simple commands. After (optionally) editting a config file, inputing their training data and their putative proteins, they can train and get predictions as simply as executing `dvc repro`.

## Supported tools
Import `aide_predict.utils.common.get_supported_tools` to see the tools that are available based on your environment.
The base package has few dependancies and concurrently few tools. Additional tools can be accessed with additional
dependancy steps. This choice was made to reduce dependancy clashes for the codebase. For example, the base
package does not include `pytorch`, but the the environment can be extended with "trasnformers_requirements.txt" to access
ESM2 embeddings and log likelihood predictors.

### Base package
#### Utilities
- Jackhmmer and MSA processing pipelines. Please see section "3rd party software" for more information
- Data structures for protein sequences and structures that are directly accepted by protein models

#### Prediction models
- HMMs
- EVCouplings (TODO)
#### Embeddings for downstream ML
- One Hot Encoding (fixed length) (TODO)
- One Hot Encoding (to an alignment) (TODO)

### Transformers
#### Embeddings for downstream ML
- ESM2 embeddings, mean pooled or position specific (TODO)
#### Prediction models
- ESM2 Mutant, WT, and masked marginal likelihoods, pooled or position specific

## Installation
```
conda env create -f environment.yaml
pip install .
```

## API

TODO

## DVC pipeline

TODO

## TODO:
- Write embeddings classes
- Write EVcouplings wrapper
- Write EVE wrapper
- Write Tranception wrapper * (low priority, PN did not provide a clear entry point so will require some finagling)
- Write MSATransformer wrapper. should be esasy if we enforce WT and fixed length. Maybe in the future extend to no WT compare by ensureing all sequences are in the passed are in the MSA
- Write "training" pipeline to init, potentially fit, and save all sklearn estimators and pipelines
- Write "predict" pipeline to load all sklearn estimators and pipelines, and predict on the passed data

## Project Structure

> UPDATE


## Usage

To use the tools for ML aided prediction of protein mutation combinatorial libraries, follow these steps:

1. Install the required dependencies by running the following command:

   ```
   conda env create -f environment.yaml
   ```

2. Prepare the data by placing it in the `data/` directory.

3. Configure the parameters for the ML models and prediction tools in the `params.yaml` file.

4. Run `dvc repro` to execute the ML models and prediction tools, spitting out what to select next


## Third party software

1. EVCouplings is a dependancy and their software is used to run jackhmmer searches and available as a position specific predictor (CITE)
2. MSA processing steps from EVE are refactored here. The original authors should be credited for the method. Here, we implement a 1-2 order of magnitude speedup with vecot processing and GPU acceleration. (CITE)

## Citations
No software or code with viral licenses was used in the creation of this project.

TODO

## License

This project is licensed under the [MIT License](LICENSE).
