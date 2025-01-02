---
title: In silico Saturation Mutagenesis
---
# Saturation Mutagenesis

## Overview

We provide tools to quickly run in silico saturation mutagenesis.


Create a `ProteinSequences` object of all single point mutations.

```python
from aide_predict import ProteinSequence, ESM2LikelihoodWrapper
import pandas as pd

# Define wild type sequence
wt = ProteinSequence(
    "MKLLVLGLPGAGKGT", 
    id="wild_type"
)

# Generate all single mutants
mutant_library = wt.saturation_mutagenesis()
print(f"Generated {len(mutant_library)} variants")
>>> Generated 285 variants  # (15 positions × 19 possible mutations)
```

Then pass these to a zero shot predictor of your choice:

```python
# Score variants using a zero-shot predictor
model = ESM2LikelihoodWrapper(
    wt=wt,
    marginal_method="masked_marginal",
    pool=True  # Get one score per variant
)
model.fit([])  # No training needed
scores = model.predict(mutant_library)

# Create results dataframe
results = pd.DataFrame({
    'mutation': mutant_library.ids,  # e.g., "M1A", "K2R", etc.
    'sequence': mutant_library,
    'prediction': scores
})

# Sort by predicted effect
results = results.sort_values('prediction', ascending=False)
print("Top 5 predicted beneficial mutations:")
print(results.head())
```

## Visualizing Results

AIDE provides built-in visualization tools for mutation effects:

```python
from aide_predict.utils.plotting import plot_mutation_heatmap

# Create heatmap of mutation effects
plot_mutation_heatmap(results['mutation'], results['prediction'])
```

The heatmap shows the predicted effect of each possible amino acid substitution at each position, making it easy to identify patterns and hotspots for engineering.

## Notes
- The `mutation` IDs follow standard notation: "M1A" means the M at position 1 was mutated to A