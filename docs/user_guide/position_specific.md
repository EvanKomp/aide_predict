---
title: Position-Specific Models
---

# Position-Specific Models

## Overview

Some protein models can generate outputs for each amino acid position in a sequence. These models use the `PositionSpecificMixin` to handle position selection and output formatting. EG. lanmguage models or one hot encodings. You might want to do this if only a few positions are changing among variants or you have a specific hypothesis about the importance of certain positions.

## Using Position-Specific Models

Position-specific models have three key parameters that control their output. Flatten and pool are mutually exclusive.

```python
from aide_predict import ESM2Embedding

# Basic usage - outputs pooled across all positions
model = ESM2Embedding(
    positions=None,  # Consider all positions
    pool='mean',      # Average across positions
    flatten=False    # because pooling by mean
)

# Position-specific - get embeddings for specific positions
model = ESM2Embedding(
    positions=[0, 1, 2],  # Only these positions
    pool=False,          # Keep positions separate
    flatten=True         # Flatten features for each position so we get a single vector
)
```

## Output Shapes

The output shape depends on the parameter combination:

```python
# Example with ESM2 (1280-dimensional embeddings)
X = ProteinSequences.from_fasta("sequences.fasta")

# Default: pooled across positions
model = ESM2Embedding(pool=True)
output = model.transform(X)  # Shape: (n_sequences, 1280)

# Selected positions, no pooling
model = ESM2Embedding(
    positions=[0, 1, 2],
    pool=False
)
output = model.transform(X)  # Shape: (n_sequences, 3, 1280)

# Selected positions, no pooling, flattened
model = ESM2Embedding(
    positions=[0, 1, 2],
    pool=False,
    flatten=True
)
output = model.transform(X)  # Shape: (n_sequences, 3*1280)
```

## Position Specificity for Variable Length Sequences

In some cases models can be position specific even if not all sequences are the same length, such as when working with homologs. However, to map positions between sequences properly, we need to:
1. Know the positions of interest in a reference sequence (usually wild type)
2. Align all sequences
3. Map the reference positions to positions in the alignment

AIDE provides tools to handle this workflow:

```python
# Start with unaligned sequences
X = ProteinSequences.from_fasta("sequences.fasta")
wt = X['wt']
wt_positions = [1, 2, 3]  # 0-indexed positions of interest in wild type

# Align sequences
X = X.align_all()
wt.msa = X

# Get alignment mapping and convert positions
alignment_mapping = X.get_alignment_mapping()
wt_alignment_mapping = alignment_mapping[wt.id]  # or use str(hash(wt)) if no ID
aligned_positions = wt_alignment_mapping[wt_positions]

# Now use these positions in any position-specific model
model = MSATransformerEmbedding(
    positions=aligned_positions,
    pool=False,
    wt=wt,  # used to get the alignment to align incoming sequence to. Alternative, wt can be None if all seqs in X have the msa attribute set to X 

)
model.fit()
embeddings = model.transform(X)
```

## Implementation Notes

- If `positions` is specified but `pool=True`, the model will first select the positions then pool across them
- `flatten=True` only applies when `pool=False` and there are multiple dimensions
- Models will raise an error if `positions` are specified but the sequences are not aligned or of fixed length
```