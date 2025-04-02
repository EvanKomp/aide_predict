---
title: Caching expensive model calls
---

# Caching Model Outputs

## Overview

Some AIDE models support caching their outputs to disk to avoid recomputing expensive transformations. This is made available with the `CacheMixin` class, which is inherited by models that support caching. You can check if a model supports caching by checking if it inherits from `CacheMixin`:

```python
from aide_predict.bespoke_models.base import CacheMixin
assert isinstance(model, CacheMixin)  # True if model supports caching
```

## Using Caches

Caching is enabled by default for models that support it. To explicitly control caching:

```python
from aide_predict import ESM2Embedding

# Disable caching
model = ESM2Embedding(use_cache=False)

# Enable caching (default)
model = ESM2Embedding(use_cache=True)
```

## How It Works

- Each protein sequence gets a unique hash based on its sequence, ID, and structure (if present)
- Outputs are stored in HDF5 format for efficient retrieval
- Cache also hashes the model parameters, so if model parameters change it will not use previous cache values
- Stores metadata in SQLite for quick cache checking
- Caches are stored in the model's metadata folder

## Models Supporting Caching

You can check if a model supports caching by checking if it inherits from `CacheMixin`:
```python
from aide_predict.bespoke_models.base import CacheMixin

isinstance(model, CacheMixin)  # True if model supports caching
```

## Cache Location

Caches are stored in a `cache` subdirectory of the model's metadata folder:
```python
# Specify cache location
model = ESM2Embedding(metadata_folder="my_model")
# Creates: my_model/cache/cache.db (metadata)
#         my_model/cache/embeddings.h5 (outputs)

# Random temporary directory if not specified
model = ESM2Embedding()
```