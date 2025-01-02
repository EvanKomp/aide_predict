---
title: Installation Guide
---

# Installing AIDE

AIDE is designed with a modular architecture to minimize dependency conflicts while providing access to a wide range of protein prediction tools. The base package has minimal dependencies and provides core functionality, while additional components can be installed based on your specific needs.

## Quick Install

For basic functionality, simply install AIDE using:

```bash
# Create and activate a new conda environment
conda env create -f environment.yaml

# Install AIDE
pip install .
```

## Supported Tools by Installation Level

AIDE provides bespoke embedders and predictors as additional modules that can be installed. These fall into three categories, with environment weight in mind: those available in the base package, those that can be installed with minimal additional pip dependencies, and those that should be built as an independant environment. 

### Base Installation
The base installation provides:
- Core data structures for protein sequences and structures
- Sequence alignment utilities 
- One-hot encoding embeddings
- K-mer based embeddings
- Basic Hidden Markov Model support
- mmseqs2 MSA generation pipeline

### Minor Pip Dependencies
#### Pure `transformers` models
ESM2 and SaProt can be defined with the transformers library. To install these models:
```bash
pip install -r requirements-transformers.txt
```
This enables:
- ESM2 embeddings and likelihood scoring
- SaProt structure-aware embeddings and scoring

#### MSA Transformer
MSA transformer requires bespoke components from fair-esm:
```bash
pip install -r requirements-fair-esm.txt
```
This enables:
- MSA transformer embeddings and likelihood scoring

#### EVmutation
For evolutionary coupling analysis:
```bash
pip install -r requirements-evmutation.txt
```
This enables:
- EVMutation for protein mutation effect prediction

#### VESPA Integration
For conservation-based variant effect prediction:
```bash
pip install -r requirements-vespa.txt
```

#### Structure Prediction with ESMFold
For structure prediction capabilities:
```bash
pip install -r requirements-esmfold.txt
```
Note: ESMFold requires a CUDA-capable GPU and is not compatible with Apple Silicon.


### Independent Environment
#### EVE Integration

EVE (Evolution Via Energy) requires special handling due to its complex environment requirements:

1. Clone the EVE repository outside your AIDE directory:
```bash
git clone https://github.com/OATML/EVE.git
```

2. Set required environment variables:
```bash
export EVE_REPO=/path/to/eve/repo
```

3. Create a dedicated conda environment for EVE following their installation instructions.

4. Set the EVE environment name:
```bash
export EVE_CONDA_ENV=eve_env
```

## Verifying Your Installation

You can check which components are available in your installation:

```python
from aide_predict.utils.checks import get_supported_tools
print(get_supported_tools())
```

## Common Installation Issues

### CUDA Compatibility
If you're using GPU-accelerated components (ESMFold, transformers), ensure your CUDA drivers are compatible:
- Check CUDA version: `nvidia-smi`
- Match PyTorch installation with CUDA version
- For Apple Silicon users: Some components may require alternative installations
