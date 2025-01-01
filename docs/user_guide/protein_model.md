---
title: Protein model basics and definition
---

# ProteinModelWrapper

## Overview

`ProteinModelWrapper` is the base class for all protein prediction models in AIDE. It inherits from scikit-learn's `BaseEstimator` and `TransformerMixin`, providing a familiar API while adding protein-specific validation and functionality.

## Core Behavior

The wrapper handles all protein-specific validation through its public methods:

```python
# Public methods do all validation
def fit(self, X, y=None):
    # Validates sequences
    # Checks requirements based on mixins
    # Then calls _fit()
    
def transform(self, X):
    # Validates sequences
    # Checks requirements based on mixins
    # Then calls _transform()
```

You never need to call the private methods directly - they implement just the core model logic.

## Key Attributes

### Data Paths and References
- `metadata_folder`: Directory for model files (weights, checkpoints, etc.), randomly generated if not given.
  - This serves to give each model a dedicated place to store necessary files, for example if a fasta file needs to be passed to an external program. Some models do not use it. If the model is capable of Caching (see the caching section), it is stored here. Currently this location may break when saving and loading models across machines.
- `wt`: Optional wild-type sequence for comparative predictions

### Automated Properties
These are set based on which mixins you use (non exausitve list):
```python
# Model requirements
model.requires_msa_for_fit
model.requires_fixed_length  
model.requires_wt_to_function
model.requires_structure

# Model capabilities  
model.per_position_capable
model.can_regress
model.can_handle_aligned_sequences
model.accepts_lower_case
```

## Wrapping a New Model

To wrap a new model:

1. Inherit from `ProteinModelWrapper` and any needed mixins
2. Implement only the core logic in private methods

Example:
```python
class MyModel(CanRegressMixin, ProteinModelWrapper):
    def __init__(self, metadata_folder=None, my_param=1.0):
        super().__init__(metadata_folder=metadata_folder)
        self.my_param = my_param
    
    def _fit(self, X, y=None):
        # Just core training logic
        # X is guaranteed to be valid
        # set a fitted attribute so sklearn can check if it's been fit
        self.fitted_ = True
        return self
    
    def _transform(self, X):
        # Just core transformation
        # X is guaranteed to be valid
        return features
```

## Mixins

Mixins define model requirements and capabilities. When combined with `ProteinModelWrapper`, they automatically enable appropriate validation and behavior. Mixins are grouped by their general purpose:

### Input Requirements

#### RequiresMSAMixin
For models trained on multiple sequence alignments:
- Sets `requires_msa_for_fit = True`
- Base class ensures input is aligned during fit
- Model receives guaranteed aligned sequences in `_fit`
- Used by evolutionary models like EVMutation, MSATransformer

#### RequiresFixedLengthMixin  
For models requiring uniform sequence length:
- Sets `requires_fixed_length = True`
- Base class validates all sequences are same length
- Common in neural networks and position-specific models
- Validates wild-type length matches if present

#### RequiresStructureMixin
For models using structural information:
- Sets `requires_structure = True`  
- Ensures structures available or falls back to wild-type structure
- Used by structure-aware models like SaProt

#### RequiresWTToFunctionMixin
For models that need a reference sequence:
- Sets `requires_wt_to_function = True`
- Ensures wild-type sequence provided at initialization
- Used by models computing mutation effects

### Output Capabilities

#### CanRegressMixin
For models producing numeric predictions:
- Sets `can_regress = True`
- Enables `predict()` method
- Adds Spearman correlation scoring
- Common in variant effect predictors

#### PositionSpecificMixin
For models with per-position outputs:
- Sets `per_position_capable = True`
- Adds position selection with `positions` parameter
- Controls output format with `pool` and `flatten` parameters
- Handles dimension validation and reshaping
- Common in language models and conservation analysis

### Data Processing

#### CanHandleAlignedSequencesMixin
For models working with gapped sequences:
- Sets `can_handle_aligned_sequences = True`
- Prevents automatic gap removal by base class
- Essential for MSA-based models
- Ensures gap characters preserved during processing

#### AcceptsLowerCaseMixin
For case-sensitive models:
- Sets `accepts_lower_case = True`
- Disables automatic uppercase conversion
- Useful when case represents conservation or focus columns

### Computational Behavior

#### CacheMixin
For caching model outputs:
- Adds disk-based caching system using SQLite and HDF5
- Thread-safe for parallel processing
- Automatic cache invalidation on parameter changes
- Particularly useful for computationally expensive models

#### ShouldRefitOnSequencesMixin
For models that need retraining on new sequences:
- Sets `should_refit_on_sequences = True`
- By default sklearn will refit when fit is called, this is undesirable for some models eg. that fit using an MSA - we want them to not refit when used in pipelines
- That default behavior was disabled for `ProteinModelWrapper` but can be re-enabled by using this mixin

### Using Multiple Mixins

Mixins can be combined to define complex model requirements:

```python
class ComplexModel(
    RequiresMSAMixin,           # Needs MSA for training
    CanRegressMixin,            # Makes predictions
    PositionSpecificMixin,      # Per-position outputs
    CacheMixin,                 # Caches results
    ProteinModelWrapper
):
    def _fit(self, X: ProteinSequences, y=None):
        # X is guaranteed to be aligned
        # Implementation focuses on core logic
        pass

    def _transform(self, X: ProteinSequences):
        # All requirements validated by base class
        # Implementation focuses on core logic
        pass