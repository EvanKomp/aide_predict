---
title: Zero-Shot Prediction
---

# Zero-Shot Prediction

## Overview

Zero-shot predictors in AIDE can assess protein variants without requiring training data. These models leverage different types of information:
- Pretrained language models that capture protein sequence patterns
- Multiple sequence alignments that capture evolutionary information
- Structural information for 3D context and conservation signals
- Combinations of these approaches for more robust predictions
- Note that the examples below are all single point mutations, but many of the models can also be used for multiple mutations.

## Transformer-Based Models

### ESM2

[ESM2](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full) uses masked language modeling to predict mutation effects based on the likelihood of amino acids in context:

```python
from aide_predict import ESM2LikelihoodWrapper, ProteinSequence

# Setup wild type sequence
wt = ProteinSequence(
    "MKLLVLGLPGAGKGT",
    id="wild_type"
)

# Choose marginal method for computing likelihoods
model = ESM2LikelihoodWrapper(
    wt=wt,
    marginal_method="masked_marginal",  # or "wildtype_marginal" or "mutant_marginal"
    pool=True  # True to get single score per sequence
)

# No training needed
model.fit()

# Score mutations
mutants = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

The marginal method determines how likelihoods are computed:
- `masked_marginal`: Masks each position to compute direct probability
- `wildtype_marginal`: Uses wild type context only
- `mutant_marginal`: Uses mutant sequence context

### MSA Transformer

[MSA Transformer](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full) extends ESM's approach by incorporating evolutionary information from multiple sequence alignments:

```python
from aide_predict import MSATransformerLikelihoodWrapper, ProteinSequence

# Setup wild type sequence with MSA
wt = ProteinSequence.from_a3m("protein_family.a3m")

# Create model with MSA context
model = MSATransformerLikelihoodWrapper(
    wt=wt,
    marginal_method="masked_marginal",
    n_msa_seqs=360  # Number of MSA sequences to use
)

# Fit to MSA 
model.fit()

# Score mutations
mutants = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

MSA Transformer combines the power of language models with evolutionary information, often improving predictions for proteins with rich evolutionary profiles.

### VESPA

[VESPA](https://link.springer.com/article/10.1007/s00439-021-02411-y) uses a pretrained model head on top of transformer embeddings specifically trained to predict variant effects:

```python
from aide_predict import VESPAWrapper, ProteinSequence

# Setup wild type sequence
wt = ProteinSequence(
    "MKLLVLGLPGAGKGT",
    id="wild_type"
)

# Create VESPA model (light version by default)
model = VESPAWrapper(
    wt=wt,
    light=True  # Use lighter VESPAl model instead of full VESPA
)

# No training needed
model.fit()

# Score single mutations (VESPA is only for single mutations)
mutants = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

VESPA was trained on human disease variants and is particularly useful for predicting pathogenicity of human protein variants.

## Structure-Aware Models

### SaProt

[SaProt](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v1) incorporates protein structure information with sequence to improve predictions:

```python
from aide_predict import SaProtLikelihoodWrapper, ProteinStructure

# Load sequences and map structures
wt = ProteinStructure.from_pdb("structures/structure.pdb")

# Create model
model = SaProtLikelihoodWrapper(
    wt=wt,
    marginal_method="masked_marginal"
)

# No training needed
model.fit()

# Score mutations with structure info
mutatnts = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

SaProt is particularly valuable for proteins where structural context plays a significant role in function or stability.

### SSEmb

[SSEmb](https://www.nature.com/articles/s41467-024-53982-z) combines structure and sequence information through a joint embedding approach:

```python
from aide_predict import SSEmbWrapper, ProteinSequence, StructureMapper

# Setup environment variables first
# os.environ['SSEMB_CONDA_ENV'] = 'ssemb_env'
# os.environ['SSEMB_REPO'] = '/path/to/ssemb/repo'

# Setup wild type with structure and MSA
wt = ProteinSequence.from_a3m("protein_family.a3m")
wt.structure = "structures/structure.pdb"

# Create model
model = SSEmbWrapper(wt=wt)

# Fit using MSA
model.fit()

# Score mutations
mutants = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

SSEmb is especially effective for scoring mutations in proteins with known structures and rich evolutionary information.

## Evolutionary Models

### HMM

Hidden Markov Models capture position-specific amino acid preferences from MSAs:

```python
from aide_predict import HMMWrapper, ProteinSequences

# Load MSA
msa = ProteinSequences.from_fasta("protein_family.a3m")

# Create and fit model
model = HMMWrapper(threshold=100)  # bit score threshold
model.fit(msa)

# Score new sequences
sequences = ProteinSequences.from_fasta("variants.fasta")
scores = model.predict(sequences)
```

HMMs are fast and interpretable but don't capture dependencies between positions.

### EVMutation

[EVMutation](https://pubmed.ncbi.nlm.nih.gov/28092658/) analyzes co-evolution patterns in MSAs to capture epistatic effects:

```python
from aide_predict import EVMutationWrapper

# Load MSA and wild type
wt = ProteinSequence.from_a3m("protein_family.a3m")

# Create and fit model
model = EVMutationWrapper(
    wt=wt,
    theta=0.8,  # Sequence weighting parameter
    protocol="standard"  # or "complex" or "mean_field"
)

# Fit using MSA
model.fit()

# Score mutations
mutants = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

EVMutation captures pairwise dependencies between positions, making it effective for predicting epistatic effects where multiple mutations interact.

### EVE

[EVE](https://www.nature.com/articles/s41586-021-04043-8) constructs a posterior latent distribution over an MSA and scores how "in-distribution" a sequence is:

```python
from aide_predict import EVEWrapper

# Load MSA and wild type
wt = ProteinSequence.from_a3m("protein_family.a3m")

# Create model with custom parameters
model = EVEWrapper(
    wt=wt,
    encoder_z_dim=50,  # Dimensionality of latent space
    training_steps=400000  # Number of training steps
)

# Fit using MSA
model.fit()

# Score mutations
mutants = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

## Contributing

If you have a zero-shot method you would like to have added, please reach out:

evan.komp (at) nrel.gov