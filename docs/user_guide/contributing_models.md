---
title: Contributing Models to AIDE
---

# Contributing Models to AIDE

## Overview

AIDE is designed to make it easy to wrap new protein prediction models into a scikit-learn compatible interface. This guide walks through the process of contributing a new model.

### 1. Setting Up Development Environment

```bash
git clone https://github.com/beckham-lab/aide_predict
cd aide_predict
conda env create -f environment.yaml
conda activate aide_predict
pip install -e ".[dev]"  # Installs in editable mode with development dependencies
```

### 2. Understanding Model Dependencies

AIDE uses a tiered dependency system to minimize conflicts and installation complexity:

1. **Base Dependencies**: If your model only needs numpy, scipy, scikit-learn, etc., it can be included in the base package.

2. **Optional Dependencies**: If your model needs additional pip-installable packages:
   - Create or update a `requirements-<feature>.txt` file
   - Example: `requirements-transformers.txt` for models using HuggingFace transformers

3. **Complex Dependencies**: If your model requires a specific environment or complex setup:
   - Package should be installed separately
   - AIDE will call it via subprocess
   - Model checks for environment variables pointing to installation
   - Example: EVE model checking for `EVE_REPO` and `EVE_CONDA_ENV`

### 3. Creating the Model Class

Models should be placed in one of two directories:
- `aide_predict/bespoke_models/embedders/`: For models that create numerical features
- `aide_predict/bespoke_models/predictors/`: For models that predict protein properties

Basic structure:

```python
from aide_predict.bespoke_models.base import ProteinModelWrapper
from aide_predict.utils.common import MessageBool

# Check dependencies
try:
    import some_required_package
    AVAILABLE = MessageBool(True, "Model is available")
except ImportError:
    AVAILABLE = MessageBool(False, "Requires some_required_package")

class MyModel(ProteinModelWrapper):
    """Documentation in NumPy style.
    
    Parameters
    ----------
    param1 : type
        Description
    metadata_folder : str, optional
        Directory for model files
    wt : ProteinSequence, optional
        Wild-type sequence for comparative predictions
        
    Attributes
    ----------
    fitted_ : bool
        Whether model has been fitted
    """
    _available = AVAILABLE  # Class attribute for availability

    def __init__(self, param1, metadata_folder=None, wt=None):
        super().__init__(metadata_folder=metadata_folder, wt=wt)
        self.param1 = param1  # Save user parameters as attributes
        
    def _fit(self, X, y=None):
        """Fit the model. Called by public fit() method."""
        # Implementation
        self.fitted_ = True  # Mark as fitted
        return self
        
    def _transform(self, X):
        """Transform sequences. Called by public transform() method."""
        # Implementation
        return features
```

### 4. Adding Model Requirements with Mixins

AIDE uses mixins to declare model requirements and capabilities. Common mixins:

```python
# Input requirements
RequiresMSAMixin          # Needs MSA for training
RequiresFixedLengthMixin  # Sequences must be same length
RequiresStructureMixin    # Uses structural information
RequiresWTMixin          # Needs wild-type sequence

# Output capabilities  
CanRegressMixin          # Can predict numeric values
PositionSpecificMixin    # Outputs per-position scores

# Processing behavior
CacheMixin               # Enables result caching
AcceptsLowerCaseMixin    # Handles lowercase sequences
```

Example with mixins:

```python
class MyModel(
    RequiresMSAMixin,      # Needs MSA for training
    CanRegressMixin,       # Makes predictions
    PositionSpecificMixin, # Per-position outputs
    CacheMixin,           # Caches results
    ProteinModelWrapper    # Always last
):
    pass
```

Ensure that the `_avialable` attribute is set to a valid `MessageBool` object that is computed on import based on the availability of the model's dependencies.

### 5. Testing Your Model

If applicable, add scientific validation tests in `tests/test_not_base_models/`:
```python
from aide_predict.bespoke_models.embedders.my_model import MyModel
def test_my_model_benchmark():
    """Test against published benchmark."""
    model = MyModel()
    score = model.score(benchmark_data)
    assert score >= expected_performance
```

Run the tests with `pytest tests/test_not_base_models/test_my_model.py`, and copy the results.

Ensure that this test is not tracked by coverage, as we do not run CI on non-base models that have additional dependencies:

Update `.coveragerc`:
```
omit = 
    ... other omitted files are here ...
    aide_predict/bespoke_models/embedders/my_model.py
```

### 7. Expose your model so that AIDE can find it and test it against user data

Update `aide_predict/bespoke_models/__init__.py` to include your model in the `TOOLS` list:

```python
from .embedders.my_model import MyModel

TOOLS = [
    ...other tools are here...
    MyModel
]
```

### 7. Submitting Your Contribution

1. Create a new branch
2. Implement your model in its own module
3. Add any tests
4. Submit a pull request, add any test results to the pull request so the expected performance can be verified

