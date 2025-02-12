---
title: Structure Prediction with SoloSeq
---

# Structure Prediction with SoloSeq

We provide a wrapper interface to get protein structure predictions using SoloSeq, a deep learning model for protein structure prediction that requires no MSAs. It is recommended to use crystal structures or run AlphaFold2 for more accurate predictions if your task is deemed very structure sensitive.

## Installation

SoloSeq requires additional setup beyond the base AIDE installation. 

1. Follow the setup steps [here](https://openfold.readthedocs.io/en/latest/Installation.html)

Once the environment is setup and unit tests pass:

2. Download the SoloSeq model weights:
```bash
bash scripts/download_openfold_soloseq_params.sh openfold/resources
```

3. Set environment variables (add to your `.bashrc` or equivalent):
```bash
export OPENFOLD_CONDA_ENV=openfold_env  # Name of conda environment
export OPENFOLD_REPO=/path/to/openfold  # Full path to OpenFold repo
```

## Basic Usage

AIDE provides a simplified interface to SoloSeq for predicting protein structures:

```python
from aide_predict import ProteinSequences
from aide_predict.utils.soloseq import run_soloseq

# Load sequences
sequences = ProteinSequences.from_fasta("proteins.fasta")

# Run prediction
pdb_paths = run_soloseq(
    sequences=sequences,
    output_dir="./predicted_structures"
)

# attach predicted structures to sequence using structure mapper
from aide_predict.utils.data_structures.structures import StructureMapper
mapper = StructureMapper("./predicted_structures")
mapper.assign_structures(sequences)
```

### Command Line Interface

You can also run predictions directly from the command line:

```bash
python -m aide_predict.utils.soloseq proteins.fasta predicted_structures
```

## Advanced Options

The function provides several options to control prediction:

```python
pdb_paths = run_soloseq(
    sequences=sequences,
    output_dir="predicted_structures",
    use_gpu=True,          # Set to False for CPU-only
    skip_relaxation=False, # Skip refinement step
    save_embeddings=True,  # Keep ESM embeddings
    device="cuda:0",       # Specific GPU device
    force=False           # Force rerun of existing predictions
)
```

Command line equivalents:

```bash
python -m aide_predict.utils.soloseq proteins.fasta predicted_structures \
    --no_gpu \
    --skip_relaxation \
    --save_embeddings \
    --device cuda:1 \
    --force
```
