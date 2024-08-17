# AIDE
[![Python Tests](https://github.com/EvanKomp/aide_predict/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/EvanKomp/aide_predict/actions/workflows/ci-tests.yml)
[![codecov](https://codecov.io/gh/EvanKomp/aide_predict/branch/main/graph/badge.svg)](https://codecov.io/gh/EvanKomp/aide_predict)

![alt text](./images/fig1.png)

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
- [X] __Create a generalizable, unittested, API for protein prediction tasks that is compatible with scikit learn__. This API will allow those who are familiar with the gold standard of ML libraries to conduct protein prediction tasks in much the same way you'd see on an intro to ML Medium article. Further, it makes it much easier for bespoke strategies to be accessed and compared; any new method whose authors wrap their code in the API are easily accessed by the community without spending hours studying the codebase.
- [ ] __Use API components to create a DVC tracked pipeline for protein prediction tasks__. This pipeline will allow for those with zero software experience to conduct protein prediction tasks with a few simple commands. After (optionally) editting a config file, inputing their training data and their putative proteins, they can train and get predictions as simply as executing `dvc repro`.

## API examples:

The following should look and feel like canonical sklearn tasks/code. See the `demo` folder for more details and executable examples. Also see the [colab notebook](https://colab.research.google.com/drive/1baz4DdYkxaw6pPRTDscwh2o-Xqum5Krp#scrollTo=AV9VXhM6ebgI) to play with some if its capabilities in the cloud.

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
package does not include `pytorch`, but the environment can be extended with "requirements-transformers.txt" to access
ESM2 embeddings and log likelihood predictors.

## Available Tools

### Data Structures and Utilities
- Protein Sequence and Structure data structures
- Jackhmmer and MSA processing pipelines
  - Uses EVCoouplings pipeline itnernally (see "3rd party software" section for more information)

### Prediction Models

1. [HMM (Hidden Markov Model)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002195)
   - Computes statistics over matching columns in an MSA, treating each column independantly but allowing for alignment of query sequences before scoring
   - Requires MSA for fitting
   - Can handle aligned sequences during inference

2. [EVMutation](https://academic.oup.com/bioinformatics/article/35/9/1582/5124274)
   - Computes pairwise couplings between AAs in an MSA for select positions well represented in the MSA, variants are scored by the change in coupling energy.
   - Requires MSA for fitting
   - Requires wild-type sequence for inference
   - Requires fixed-length sequences

3. [ESM2 Likelihood Wrapper](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
   - Pretrained PLM (BERT style) model for protein sequences, scores variants according to masked, mutant, or wild type marginal likelihoods. Mutant marginal computes likelihoods in the context of the mutant sequence, while masked and wild type marginal compute likelihoods in the context of the wild type sequence. These methods are apprximations of the joint likelihood.
   - Can handle aligned sequences
   - Requires additional dependencies (see `requirements-transformers.txt`)

4. [SaProt Likelihood Wrapper](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v2)
   - ESM except using a size 400 vocabulary including local structure tokens from Foldseek's VAE. The authors only used Masked marginal, but we've made Wild type, Mutant, and masked marginals avialable.
   - Requires fixed-length sequences
   - Uses WT structure if structures of sequences are not passed
   - Requires additional dependencies:
     - `requirements-transformers.txt`

5. [MSA Transformer Likelihood Wrapper](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1.full)
   - Like ESM but with a transformer model that is trained on MSAs. The variants are placed at the top position in the MSA and scores are computed along that row. Wild type, Mutant, and masked marginals avialable.
   - Requires MSA for fitting
   - Requires wild-type sequence during inference
   - Requires additional dependencies (see `requirements-fair-esm.txt`)

6. [VESPA](https://link.springer.com/article/10.1007/s00439-021-02411-y)
   - Conservation head model trained on PLM embeddings and logistic regression used to predict if mutation is detrimental.
   - Requires wild type, only works for single point mutations
   - Requires fixed-length sequences
   - Requires additional dependencies (see `requirements-vespa.txt`)

### Embeddings for Downstream ML

1. One Hot Protein Embedding
   - Columnwise one hot encoding of amino acids for a fixed length set of sequences
   - Requires fixed-length sequences
   - Position specific

2. One Hot Aligned Embedding
   - Columnwise one hot encoding including gaps for sequences aligned to an MSA.
   - Requires MSA for fitting
   - Position specific

3. Kmer Embedding
   - Counts of observed amino acid kmers in the sequences
   - Allows for variable length sequences

4. [ESM2 Embedding](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
   - Pretrained PLM (BERT style) model for protein sequences, outputs embeddings for each amino acid in the sequece from the last transformer layer.
   - Position specific
   - Requires additional dependencies (see `requirements-transformers.txt`)

5. [SaProt Embedding](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v2)
   - ESM except using a size 400 vocabulary including local structure tokens from Foldseek's VAE. AA embeddings from the last layer of the transformer are used.
   - Position specific
   - Requires additional dependencies:
     - `requirements-transformers.txt`
     - `foldseek` executable must be available in the PATH

6. [MSA Transformer Embedding](https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1.full)
   - Like ESM but with a transformer model that is trained on MSAs. The embeddings are computed for each amino acid in the query sequence in the context of an existing MSA
   - Requires MSA for fitting
   - Requires fixed-length sequences
   - Requires additional dependencies (see `requirements-fair-esm.txt`)

Each model in this package is implemented as a subclass of `ProteinModelWrapper`, which provides a consistent interface for all models. The specific behaviors (e.g., requiring MSA, fixed-length sequences, etc.) are implemented using mixins, making it easy to understand and extend the functionality of each model.

## Installation
```
conda env create -f environment.yaml
pip install .
```

## Installation of additional modules
Tools that require additional dependancies can be installed with the corresponding requirements file. See above for those files. For example, to access VESPA:
```
pip install -r requirements-vespa.txt
```

## TODO:
- Write EVE wrapper
- Write Tranception wrapper * (low priority, PN did not provide a clear entry point so will require some finagling)
- Write "training" pipeline to init, potentially fit, and save all sklearn estimators and pipelines
- Write "predict" pipeline to load all sklearn estimators and pipelines, and predict on the passed data

## DVC pipeline

TODO


### Usage (NOT CURRENTLY AVAILABLE)

To use the tools for ML aided prediction of protein mutation combinatorial libraries, follow these steps:

1. Install the required dependencies by running the following command:

   ```
   conda env create -f environment.yaml
   ```

2. Prepare the data by placing it in the `data/` directory.

3. Configure the parameters for the ML models and prediction tools in the `params.yaml` file.

4. Run `dvc repro` to execute the ML models and prediction tools, spitting out what to select next


## Third party software

1. [EVCouplings](https://academic.oup.com/bioinformatics/article/35/9/1582/5124274) is a dependancy and their software is used to run jackhmmer searches and available as a position specific predictor.
2. Of course, many of the tools here are just wrapping of the work of others - see above.

## Citations
No software or code with viral licenses was used in the creation of this project.


## License

This project is licensed under the [MIT License](LICENSE).
