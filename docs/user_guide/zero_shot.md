---
title: Zero-Shot Prediction
---

# Zero-Shot Prediction

## Overview

Zero-shot predictors in AIDE can assess protein variants without requiring training data. These models leverage different types of information:
- Pretrained language models that capture protein sequence patterns
- Multiple sequence alignments that capture evolutionary information
- Structural information and conservation signals

## Transformer-Based Models

Language models like [ESM2](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full), [MSATransformer](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2.full), and [SaProt](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v1) use masked language modeling to predict mutation effects:

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
model.fit([])

# Score mutations
mutants = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

The marginal method determines how likelihoods are computed:
- `masked_marginal`: Masks each position to compute direct probability
- `wildtype_marginal`: Uses wild type context only
- `mutant_marginal`: Uses mutant sequence context

[VESPA](https://link.springer.com/article/10.1007/s00439-021-02411-y) has a pretrained model head on top of transformer embeddings to predict variant affect.

## Hidden Markov Models

HMMs capture position-specific amino acid preferences from MSAs:

```python
from aide_predict import HMMWrapper, ProteinSequences

# Load MSA
msa = ProteinSequences.from_fasta("protein_family.a3m")

# Create and fit model
model = HMMWrapper(threshold=100)  # bit score threshold
model.fit(msa)

# Score new sequences
scores = model.predict(sequences)
```

HMMs are fast and interpretable but don't capture dependencies between positions.

## Evolutionary Coupling Models 

[EVMutation](https://pubmed.ncbi.nlm.nih.gov/28092658/) analyzes co-evolution patterns in MSAs:

```python
from aide_predict import EVMutationWrapper

# Load MSA and wild type
msa = ProteinSequences.from_fasta("protein_family.a3m")
wt = msa[0]  # First sequence is wild type

# Create and fit model
model = EVMutationWrapper(wt=wt)
model.fit(msa)

# Score mutations
mutants = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

EVMutation captures pairwise dependencies between positions, making it effective for predicting epistatic effects.

[EVE](https://www.nature.com/articles/s41586-021-04043-8) contructs a posterior latent distribution over an MSA, and scores how "in" distribution a sequence is.

```python
from aide_predict import EVEWrapper

# Load MSA and wild type
msa = ProteinSequences.from_fasta("protein_family.a3m")
wt = msa[0]  # First sequence is wild type

# Create and fit model
model = EVEWrapper(wt=wt)
model.fit(msa)

# Score mutations
mutants = wt.saturation_mutagenesis()
scores = model.predict(mutants)
```

## Structure-Aware Models

[SaProt](https://www.biorxiv.org/content/10.1101/2023.10.01.560349v1) incorporates structural information when available:

```python
from aide_predict import SaProtLikelihoodWrapper, StructureMapper

# Load sequences and map structures
mapper = StructureMapper("structures/")
wt = mapper.get_protein_sequences()[0]

model = SaProtLikelihoodWrapper(wt=wt)
model.fit([])

# Score mutations with structure info
mutants = wt.saturation_mutagenesis()
mutants = mapper.assign_structures(mutants)
scores = model.predict(mutants)
```

## Contributing

If you have a zero shot method you would like to have added please reach out to me and we can work together!

evan.komp (at) nrel.gov