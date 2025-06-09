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
wt.msa = msa 

# Check compatibility
compatibility = check_model_compatibility(
    training_sequences=sequences,  # Optional: sequences for supervised learning
    testing_sequences=None,        # Optional: test sequences if different from training
    wt=wt                          # Optional: wild-type sequence, may have structure, MSA
)

print("Compatible models:", compatibility["compatible"])
print("Incompatible models:", compatibility["incompatible"])
```

The compatibility checker performs several validation steps:
- Verifies if structure information is available (either in sequences or wild-type)
- Checks if MSAs are available and properly aligned
- Validates that sequence lengths match requirements
- Ensures wild-type sequences are available when needed
- Verifies that per-sequence MSAs match sequence lengths when required

You can also check which tools are available in your current installation:

```python
from aide_predict.utils.checks import get_supported_tools
print(get_supported_tools())
```

## Model Categories

AIDE models fall into several categories:

### 1. Zero-Shot Predictors
These models don't require training data but may have other requirements:

```python
# ESM2 - Requires only sequences
from aide_predict import ESM2LikelihoodWrapper
model = ESM2LikelihoodWrapper(wt=wt)
model.fit()  # No training needed
scores = model.predict(sequences)

# MSATransformer - Requires MSA for the WT
from aide_predict import MSATransformerLikelihoodWrapper
model = MSATransformerLikelihoodWrapper(wt=wt) 
model.fit() 
scores = model.predict(sequences)

# SaProt - Can use structural information
from aide_predict import SaProtLikelihoodWrapper
model = SaProtLikelihoodWrapper(wt=wt) # wt must have structure
model.fit()
scores = model.predict(sequences)  # Will use structure if available
```

Other zero-shot predictors include:
- **HMM**: Creates Hidden Markov Models from MSAs
- **EVMutation**: Uses evolutionary couplings from MSAs
- **VESPA**: Pre-trained model for variant effect prediction
- **EVE**: Evolutionary model using latent space representations
- **SSEmb**: Structure and sequence-based variant effect predictor

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

# K-mer based embeddings
from aide_predict import KmerEmbedding
embedder = KmerEmbedding(k=3)
X = embedder.fit_transform(sequences)
```

Other embedding models include:
- **MSATransformerEmbedding**: Produces embeddings using MSAs
- **SaProtEmbedding**: Structure-aware protein language model embeddings
- **OneHotAlignedEmbedding**: One-hot encodings for aligned sequences

## Importance of Data Structure

The compatibility of models depends heavily on the structure of your data:

| Data Characteristic | Compatible Models | Incompatible Models |
|---------------------|-------------------|---------------------|
| Fixed-length sequences | All models | - |
| Variable-length sequences | Models without `RequiresFixedLengthMixin` | Models with `RequiresFixedLengthMixin` |
| Has MSA | All models | - |
| No MSA | Models without MSA requirements | MSATransformer, EVMutation, EVE |
| Has structure | All models | - |
| No structure | Models without structure requirements | SaProt, SSEmb |
| Has wild-type | All models | - |
| No wild-type | Models without WT requirements | Models with `RequiresWTToFunctionMixin` |

Using the appropriate data structure for your specific modeling task ensures that AIDE can provide the most accurate predictions.