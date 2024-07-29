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

## API examples:

The following should look and feel like canonical sklearn tasks/code. See the `demo` folder for more details and executable examples.

#### In silico mutagenesis using MSATransformer
```python
# data preparation
wt = ProteinSequence(
    "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR",
)
msa = ProteinSequences.from_fasta("data/msa.fasta")
library = wt.saturation_mutagenesis()
mutations = [p.id for p in library]
print(mutations[0])
>>> 'L1A'

# model fitting
model = MSATransformerLikelihoodWrapper(
   wt=wt,
   marginal_method="masked_marginal"
)
model.fit(msa)

# make predictions for each mutated sequence
predictions = model.predict(library)

results = pd.DataFrame({'mutation': mutations, 'seqeunce': library,'prediction': predictions})
```

#### Compare a couple of zero shot predictors against experimental data
```python
# data preparation
data = pd.read_csv("data/experimental_data.csv")
X = ProteinSequences.from_list(data['sequence'])
y = data['experimental_value']
wt = X['my_id_for_WT']
msa = ProteinSequences.from_fasta("data/msa.fasta")

# model defenitions
evmut = EVMutation(wt=wt, metadata_folder='./tmp/evm')
esm2 = ESM2LikelihoodWrapper(wt=wt, model_checkpoint='esm2_t33_650M_UR50S')
models = {'evmut': evmut, 'esm2': esm2}

# model fitting and scoring
for name, model in models.items():
    model.fit(msa)
    score = model.score(X, y)
    print(f"{name} score: {score}")
```

#### Train a supervised model to predict activity on an experimental combinatorial library, test on sequences with greater mutational depth than training
```python
# data preparation
data = pd.read_csv("data/experimental_data.csv")
sequences = ProteinSequences.from_list(data['sequence'])
sequences.aligned
>>> True
sequences.fixed_length
>>> True

wt = sequences['my_id_for_WT']
data['sequence'] = sequences
data['mutational_depth'] = data['sequence'].apply(lambda x: x.num_mutations(wt))
test = data[data['mutational_depth'] > 5]
train = data[data['mutational_depth'] <= 5]
train_X, train_y = train['sequence'], train['experimental_value']
test_X, test_y = test['sequence'], test['experimental_value']

# embeddings protein sequences
# use mean pool embeddings of esm2
embedder = ESM2Embedding(pool=True)
train_X = embedder.fit_transform(train_X)
test_X = embedder.transform(test_X)

# model fitting
model = RandomForestRegressor()
model.fit(train_X, train_y)

# model scoring
train_score = model.score(train_X, train_y)
test_score = model.score(test_X, test_y)
print(f"Train score: {train_score}, Test score: {test_score}")
```

#### Train a supervised predictor on a set of homologs, focusing only on positions of known importance, wrap the entire process into an sklearn pipeline including some standard sklearn transormers, and make predictions for a new set of homologs
```python
# data preparation
data = pd.read_csv("data/experimental_data.csv")
data.set_index('id', inplace=True)
sequences = ProteinSequences.from_dict(data['sequence'].to_dict())
y_train = data['experimental_value']

wt = sequences['my_id_for_WT']
wt_important_positions = [20, 21, 22, 33, 45] # zero indexed, known from analysis elsewhere
sequences.aligned
>>> False
sequences.fixed_length
>>> False

# align the training sequences and get the important positions
msa = sequences.align_all()
msa.fixed_length
>>> False
msa.aligned
>>> True

aligned_important_positions = msa['my_id_for_WT'].get_aligned_positions(wt_important_positions)

# model defenitions
embedder = OneHotAlignedEmbedding(important_positions=aligned_important_positions)
scaler = StandardScaler()
feature_selector = VarianceThreshold(threshold=0.2)
predictor = RandomForestRegressor()
pipeline = Pipeline([
    ('embedder', embedder),
    ('scaler', scaler),
    ('feature_selector', feature_selector),
    ('predictor', predictor)
])

# model fitting
pipeline.fit(sequences, y_train)

# score new analigned homologs
new_homologs = ProteinSequences.from_fasta("data/new_homologs.fasta")
y_pred = pipeline.predict(new_homologs)
```

## Supported tools
Import `aide_predict.utils.common.get_supported_tools()` to see the tools that are available based on your environment.
The base package has few dependencies and concurrently few tools. Additional tools can be accessed with additional
dependency steps. This choice was made to reduce dependency clashes for the codebase. For example, the base
package does not include `pytorch`, but the environment can be extended with "transformers_requirements.txt" to access
ESM2 embeddings and log likelihood predictors.

### Base package
#### Utilities
- Jackhmmer and MSA processing pipelines. Please see section "3rd party software" for more information
- Data structures for protein sequences and structures that are directly accepted by protein models

#### Prediction models

1. HMM (Hidden Markov Model)
   - Requires MSA for fitting
   - Can handle aligned sequences during inference
   - Scientific context: HMMs capture position-specific amino acid preferences and insertions/deletions in protein families. Columns are treated independantly. 

2. EVMutation 
   - Requires MSA for fitting
   - Requires wild-type sequence for inference
   - Requires fixed-length sequences
   - Scientific context: EVMutation uses evolutionary couplings to predict the effects of mutations. It captures both site-specific conservation and pairwise epistatic effects in proteins. [X](https://evcouplings.org/)

#### Embeddings for downstream ML

1. One Hot Protein Embedding
   - Requires fixed-length sequences
   - Position specific, eg. positions in the sequence can be passed and the embedding is subset to those positions.
   - Scientific context: One-hot encoding represents each amino acid as a binary vector, providing a simple numerical representation of sequences for machine learning models.

2. One Hot Aligned Embedding
   - Requires MSA for fitting
   - Position specific, eg. positions in the alignment can be passed and the embedding is subset to those positions.
   - Incoming sequences are aligned if not already to the stored MSA, and only aligned columns used in the OHE
   - Scientific context: This embedding applies one-hot encoding to aligned sequences, preserving positional information in the context of related sequences.

### Transformers package
See "requirements-transformers.txt" for the additional dependencies required to access these tools.

#### Embeddings for downstream ML

1. ESM2 Embedding
   - Language model embeddings have been shown to be useful for downstream ML tasks [X](https://doi.org/10.1016/j.trac.2024.117540)
   - Accepts aligned sequences
   - Position specific, eg. positions in the sequence (or alignment if aligned sequences are passed) can be passed and the embedding is subset to those positions.
   - Scientific context: ESM2 is a large language model trained on millions of protein sequences. Its embeddings capture rich, contextual information about protein sequence and structure.

2. SaProt Embedding
   - The `foldseek` executable must be available in the PATH
   - Position specific, eg. positions in the sequence can be passed and the embedding is subset to those positions.
   - Scientific context: SaProt is a structure-aware protein language model. Its embeddings incorporate both sequence and structural information: structure is tokenized a la foldseek and included in the vocabulary. If passeds sequences do not have structures, it is expected that a WT is passed and has a structure. [X](https://doi.org/10.1101/2023.10.01.560349)

#### Prediction models

1. ESM2 Likelihood Wrapper
   - Evolutionary scale protein language model has shown to have zero shot correlation to function. [X](https://doi.org/10.1101/2021.07.09.450648)
   - Can handle aligned sequences
   - Scientific context: This model uses ESM2 to compute log-likelihoods of sequences, which can be used to predict the effects of mutations or assess sequence fitness.

2. SaProt Likelihood Wrapper
   - Currently (7.29.24) holds the best mean score on ProteinGym [X](https://proteingym.org/benchmarks)
   - Requires fixed-length sequences
   - Uses WT structure if structures of sequences are not passed
   - Scientific context: SaProt's likelihood predictions incorporate both sequence and structural information, potentially providing more accurate assessments of mutation effects.

Each model in this package is implemented as a subclass of `ProteinModelWrapper`, which provides a consistent interface for all models. The specific behaviors (e.g., requiring MSA, fixed-length sequences, etc.) are implemented using mixins, making it easy to understand and extend the functionality of each model.

### FAIR ESM package
ESM2 can be used with the transformers package, but MSA transformer is not available as a class in the transformers package, and required fair-esm to be installed.
See "requirements-fair-esm.txt" for the additional dependencies required to access these tools.

#### Embeddings for downstream ML

1. MSA Transformer Embedding
   - Requires MSA for fitting
   - Requires fixed-length sequences
   - Scientific context: This model leverages information from multiple sequence alignments to generate embeddings that capture evolutionary context and conservation patterns. Column and row attention means that hypothetically the emneddings of an amino acid are conditioned on the observed primary structure and conservation of the whole MSA

#### Prediction models
1. MSA Transformer Likelihood Wrapper
   - Evolutionary scale protein language model has shown to have zero shot correlation to function. [X](https://doi.org/10.1101/2021.07.09.450648)
   - Requires MSA for fitting
   - Requires wild-type sequence during inference
   - Scientific context: This model computes log-likelihoods based on the MSA Transformer, which captures evolutionary context and conservation patterns in protein sequences.

## Installation
```
conda env create -f environment.yaml
pip install .
```

## DVC pipeline

TODO

## TODO:
- Write EVE wrapper
- Write Tranception wrapper * (low priority, PN did not provide a clear entry point so will require some finagling)
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
