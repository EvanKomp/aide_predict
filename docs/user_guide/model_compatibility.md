---
title: Model Compatibility
---

# Model Compatibility

## Understanding Model Requirements

AIDE models have different requirements and capabilities that determine whether they can be used with your data. Key considerations include:

- Whether the model requires training data (supervised vs zero-shot)
- Whether sequences must be aligned or of fixed length
- Whether the model requires a Multiple Sequence Alignment (MSA)
- Whether the model requires or can use structural information
- Whether the model needs a wild-type sequence for comparison
- Whether the model can handle variable-length sequences

## Checking Model Compatibility

AIDE provides a utility function to check which models are compatible with your data:

```python
from aide_predict.utils.checks import check_model_compatibility
from aide_predict import ProteinSequences, ProteinSequence

# Example setup
sequences = ProteinSequences.from_fasta("my_sequences.fasta")
msa = ProteinSequences.from_fasta("family_msa.fasta")
wt = ProteinSequence("MKLLVLGLPGAGKGT", id="wild_type")

# Check compatibility
compatibility = check_model_compatibility(
    training_sequences=sequences,  # Optional: sequences for supervised learning
    training_msa=msa,            # Optional: MSA for models that need it
    wt=wt                        # Optional: wild-type sequence
)

print("Compatible models:", compatibility["compatible"])
print("Incompatible models:", compatibility["incompatible"])
```

## Model Categories

AIDE models fall into several categories:

### 1. Zero-Shot Predictors
These models don't require training data but may have other requirements:

```python
# ESM2 - Requires only sequences
from aide_predict import ESM2LikelihoodWrapper
model = ESM2LikelihoodWrapper(wt=wt)
model.fit([])  # No training needed
scores = model.predict(sequences)

# MSATransformer - Requires MSA
from aide_predict import MSATransformerLikelihoodWrapper
model = MSATransformerLikelihoodWrapper(wt=wt)
model.fit(msa)  # Needs MSA for context
scores = model.predict(sequences)

# SaProt - Can use structural information
from aide_predict import SaProtLikelihoodWrapper
model = SaProtLikelihoodWrapper(wt=wt)
model.fit([])
scores = model.predict(sequences)  # Will use structure if available
```

others: HMM, EVMutation, VESPA, EVE

### 2. Embedding Models
These models convert sequences into numerical features for downstream ML:

```python
# Simple one-hot encoding
from aide_predict import OneHotProteinEmbedding
embedder = OneHotProteinEmbedding()
X = embedder.fit_transform(sequences)

# Advanced language model embeddings
from aide_predict import ESM2Embedding
embedder = ESM2Embedding(pool=True)  # pool=True for sequence-level embeddings
X = embedder.fit_transform(sequences)
```
Others: MSATransformerEMbedding, SaProtEmbedding, OneHotAlignedProteinEmbedding