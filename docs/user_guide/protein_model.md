---
title: Protein Model Framework
---

# ProteinModelWrapper

## Overview

`ProteinModelWrapper` is the base class for all protein prediction models in AIDE. It inherits from scikit-learn's `BaseEstimator` and `TransformerMixin`, providing a familiar API while adding protein-specific validation and functionality. The framework uses a powerful mixin architecture to define model requirements and capabilities.

## Core Behavior

The wrapper handles all protein-specific validation through its public methods:

```python
# Public methods do all validation
def fit(self, X, y=None):
    # Validates sequences
    # Checks requirements based on mixins
    # Runs pre-fit hooks from mixins
    # Then calls _fit()
    # Runs post-fit hooks from mixins
    
def transform(self, X):
    # Validates sequences
    # Checks requirements based on mixins
    # Runs pre-transform hooks from mixins
    # Then calls _transform()
    # Runs post-transform hooks from mixins
```

You never need to call the private methods directly - they implement just the core model logic while the public methods handle validation, hooks, and other processing.

## Key Attributes

### Data Paths and References
- `metadata_folder`: Directory for model files (weights, checkpoints, etc.), randomly generated if not given.
  - This serves to give each model a dedicated place to store necessary files, for example if a fasta file needs to be passed to an external program. Some models do not use it. If the model is capable of Caching (see the caching section), it is stored here. Currently this location may break when saving and loading models across machines.
- `wt`: Optional wild-type sequence for comparative predictions

### Automated Properties
These are set based on which mixins you use:
```python
# Model requirements
model.expects_no_fit               # Whether the model expects no fit
model.requires_msa_for_fit         # Whether the model requires an MSA as input for fitting
model.requires_wt_msa              # Whether the model requires a wild type MSA
model.requires_msa_per_sequence    # Whether the model requires an MSA for each sequence
model.requires_fixed_length        # Whether the model requires a fixed length input
model.requires_wt_to_function      # Whether the model requires the wild type sequence
model.requires_wt_during_inference # Whether wild type is needed during inference
model.requires_structure           # Whether the model requires structure information

# Model capabilities  
model.per_position_capable         # Whether the model can output per position scores
model.can_regress                  # Whether the model outputs are estimates of activity
model.can_handle_aligned_sequences # Whether the model can handle unaligned sequences
model.accepts_lower_case           # Whether the model can accept lowercase sequences
model.should_refit_on_sequences    # Whether model should refit on new sequences
```

## Mixin Hooks System

AIDE uses a system of hooks to allow mixins to modify the behavior of fitting and transformation. Hooks are registered during subclass creation and executed in order:

- `_mixin_init_handlers`: Runs during initialization to extract mixin parameters
- `_pre_fit_hooks`: Runs before fitting to prepare data
- `_post_fit_hooks`: Runs after fitting to process results
- `_pre_transform_hooks`: Runs before transformation to prepare data
- `_post_transform_hooks`: Runs after transformation to process results

For example, the `CacheMixin` uses `_pre_transform_hook` to check if results are already cached and `_post_transform_hook` to store new results in the cache.

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

## Availability Checking

Models can specify their availability based on installed dependencies:

```python
try:
    import some_required_package
    AVAILABLE = MessageBool(True, "Model is available")
except ImportError:
    AVAILABLE = MessageBool(False, "Requires some_required_package")

class MyModel(ProteinModelWrapper):
    _available = AVAILABLE
```

## Mixins

Mixins define model requirements and capabilities. When combined with `ProteinModelWrapper`, they automatically enable appropriate validation and behavior. Mixins are grouped by their general purpose:

### Training Behavior

#### ExpectsNoFitMixin
For models that don't require training:
- Sets `expects_no_fit = True`
- Used by pretrained models like ESM2, SaProt
- Allows `fit()` to be called with no data

#### ShouldRefitOnSequencesMixin
For models that need retraining on new sequences:
- Sets `should_refit_on_sequences = True`
- By default sklearn will refit when fit is called, this is undesirable for some models (e.g., those that fit using an MSA)
- That default behavior was disabled for `ProteinModelWrapper` but can be re-enabled by using this mixin

### Input Requirements

#### RequiresMSAForFitMixin
For models trained on multiple sequence alignments:
- Sets `requires_msa_for_fit = True`
- Base class ensures input is aligned during fit
- Model receives guaranteed aligned sequences in `_fit`
- Used by evolutionary models like EVMutation, HMM

#### RequiresWTMSAMixin
For models that need the wild-type MSA during training:
- Sets `requires_wt_msa = True`
- Ensures wild-type sequence has an associated MSA
- Used by models like EVE that need WT context

#### RequiresMSAPerSequenceMixin
For models requiring MSA information for each sequence:
- Sets `requires_msa_per_sequence = True` 
- Ensures each sequence has its own MSA or falls back to WT MSA
- Used by MSA Transformer for embedding generation

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

#### RequiresWTDuringInferenceMixin
For models that need wild-type context during inference:
- Sets `requires_wt_during_inference = True`
- Ensures WT sequence is accessible during transform
- Used by models that normalize predictions against WT

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

### Using Multiple Mixins

Mixins can be combined to define complex model requirements. Always put `ProteinModelWrapper` last in the inheritance chain:

```python
class ComplexModel(
    RequiresMSAForFitMixin,       # Needs MSA for training
    RequiresWTToFunctionMixin,    # Needs wild-type reference
    CanRegressMixin,              # Makes predictions
    PositionSpecificMixin,        # Per-position outputs
    CacheMixin,                   # Caches results
    ProteinModelWrapper           # Always last
):
    def _fit(self, X: ProteinSequences, y=None):
        # X is guaranteed to be aligned
        # Implementation focuses on core logic
        pass

    def _transform(self, X: ProteinSequences):
        # All requirements validated by base class
        # Implementation focuses on core logic
        pass
```

## Complete List of Mixins

| Mixin | Purpose | Sets |
|-------|---------|------|
| `ExpectsNoFitMixin` | For models without fitting | `expects_no_fit = True` |
| `RequiresMSAForFitMixin` | For MSA-based training | `requires_msa_for_fit = True` |
| `RequiresWTMSAMixin` | For models needing WT MSA | `requires_wt_msa = True` |
| `RequiresMSAPerSequenceMixin` | For per-sequence MSA | `requires_msa_per_sequence = True` |
| `RequiresFixedLengthMixin` | For fixed-length inputs | `requires_fixed_length = True` |
| `RequiresStructureMixin` | For structure-aware models | `requires_structure = True` |
| `RequiresWTToFunctionMixin` | For models needing WT | `requires_wt_to_function = True` |
| `RequiresWTDuringInferenceMixin` | For WT during inference | `requires_wt_during_inference = True` |
| `CanRegressMixin` | For prediction models | `can_regress = True` |
| `PositionSpecificMixin` | For position outputs | `per_position_capable = True` |
| `CanHandleAlignedSequencesMixin` | For gapped sequences | `can_handle_aligned_sequences = True` |
| `AcceptsLowerCaseMixin` | For case sensitivity | `accepts_lower_case = True` |
| `CacheMixin` | For result caching | Adds caching hooks |
| `ShouldRefitOnSequencesMixin` | For retraining | `should_refit_on_sequences = True` |