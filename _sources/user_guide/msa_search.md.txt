---
title: Generating MSAs with MMseqs2
---

# Generating MSAs with MMseqs2

For problems where you have not already determined an MSA with another tool (eg. Jackhmmer, EVCouplings, MMseqs, etc.) AIDE provides a high lavel wrapper for generating Multiple Sequence Alignments (MSAs) using MMseqs2, implementing the sensitive search similar to colabfold. This can be useful when you need MSAs for models like EVMutation, MSATransformer, or EVE. This is literally just calling MMseqs with a few parameters set - all credit should go to the authors of MMseqs and Colabfold:

Steinegger M and Soeding J. MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets. Nature Biotechnology, doi: 10.1038/nbt.3988 (2017).

Mirdita, M., Schütze, K., Moriwaki, Y. et al. ColabFold: making protein folding accessible to all. Nat Methods 19, 679–682 (2022). https://doi.org/10.1038/s41592-022-01488-1

## Installation

1. Ensure MMseqs2 is installed and available in your PATH:
```bash
conda install -c bioconda mmseqs2
```

2. Download the ColabFold database(s): https://colabfold.mmseqs.com/. You will need to point towards this database to run the search.

## Basic Usage

### Python Interface

```python
from aide_predict import ProteinSequences
from aide_predict.utils.mmseqs_msa_search import run_mmseqs_search

# Load sequences
sequences = ProteinSequences.from_fasta("proteins.fasta")

# Generate MSAs
msa_paths = run_mmseqs_search(
    sequences=sequences,
    uniref_db="path/to/uniref30_2302",
    output_dir="./msas"
)

# Load MSAs for use with models
from aide_predict import ProteinSequences
msas = [ProteinSequences.from_a3m(path) for path in msa_paths]
```

### Command Line Interface

You can also run MSA generation directly from the command line:

```bash
python -m aide_predict.utils.mmseqs_msa_search \
    proteins.fasta \
    path/to/uniref30_2302 \
    ./msas
```

## Advanced Options

The search can be customized with several parameters:

```python
msa_paths = run_mmseqs_search(
    sequences=sequences,
    uniref_db="path/to/uniref30_2302",
    output_dir="./msas",
    mode='sensitive',     # Search sensitivity: 'fast', 'standard', or 'sensitive'
    threads=8,           # Number of CPU threads
)
```

Command line equivalents:

```bash
python -m aide_predict.utils.mmseqs_msa_search \
    proteins.fasta \
    path/to/uniref30_2302 \
    ./msas \
    --mode sensitive \
    --threads 8 \
    --keep-tmp
```

## Search Modes

Three sensitivity modes are available:
- `fast`: Quick search with sensitivity 4.0
- `standard`: Balanced approach with sensitivity 5.7 (default)
- `sensitive`: More thorough search with sensitivity 7.5

Higher sensitivity will find more distant homologs but takes longer to run.

## Output Format

MSAs are generated in A3M format, one file per input sequence. The files are named based on the sequence IDs in your input FASTA file. These files can be directly used with AIDE's MSA-based models:

```python
# Use MSA with a model
from aide_predict import MSATransformerLikelihoodWrapper

msa = ProteinSequences.from_a3m("msas/sequence1.a3m")
model = MSATransformerLikelihoodWrapper(wt=wt)
model.fit(msa)
```