---
title: Data Structures, IO
---

# Data Structures

AIDE provides several key data structures for working with protein sequences and structures. Models will receive these data structures as input in analog to how sklearn receives numpy arrays.

## ProteinSequence

`ProteinSequence` is the basic unit for representing a protein sequence. It behaves like a string but provides additional functionality specific to protein sequences.

```python
from aide_predict import ProteinSequence

# Create a basic sequence
seq = ProteinSequence("MKLLVLGLPGAGKGT")

# or get from a pdb (see ProteinStructure below)
seq = ProteinSequence.from_pdb("path/to/structure.pdb")

# Add identifier and optional structure
seq = ProteinSequence(
    "MKLLVLGLPGAGKGT",
    id="P1234",
    structure="path/to/structure.pdb"
)

# Create a sequence with an associated MSA
seq = ProteinSequence(
    "MKLLVLGLPGAGKGT",
    id="P1234",
    msa=msa_sequences  # ProteinSequences object containing the MSA
)

# From a fasta file, assumed to be aligned. Uses first sequence in file, set's the MSA attribute
seq = ProteinSequence.from_fasta("path/to/aligned.fasta")
seq.has_msa
>>> True
```


Key attributes and methods:

- `id`: Optional identifier for the sequence
- `structure`: Optional associated structure (as ProteinStructure object)
- `msa`: Optional associated multiple sequence alignment (as ProteinSequences object)
- `has_gaps`: Boolean indicating if sequence contains gaps ('-' or '.')
- `base_length`: Length excluding gaps
- `has_non_canonical`: Boolean indicating presence of non-standard amino acids
- `as_array`: Sequence as numpy array
- `has_msa`: Boolean indicating if the sequence has an associated MSA
- `msa_same_width`: Boolean indicating if the MSA has the same width as the sequence
- `is_in_msa`: Boolean indicating if the sequence is present in its MSA

Common operations:

```python
# Sequence manipulation
ungapped = seq.with_no_gaps()  # Remove gaps
upper_seq = seq.upper()  # Convert to uppercase

# Mutation operations
mutated = seq.mutate("A123G")  # Single mutation
mutated = seq.mutate(["A123G", "L45R"])  # Multiple mutations

# Compare sequences
positions = seq.mutated_positions(other_seq)  # Get positions that differ
mutations = seq.get_mutations(other_seq)  # Get mutation strings (e.g., "A123G")

# Create all possible mutations
library = seq.saturation_mutagenesis()  # Generate all single mutants
library = seq.saturation_mutagenesis(positions=[1,2,3])  # Specific positions

# align to another sequence
seq2 = seq2.align(seq)

# align to an existing alignment
seq2 = seq2.align(msa)  # Align to existing MSA

```

## ProteinSequences

`ProteinSequences` manages collections of protein sequences, similar to a list but with additional functionality for protein-specific operations.

```python
from aide_predict import ProteinSequences

# Create from various sources
sequences = ProteinSequences([seq1, seq2, seq3])  # From ProteinSequence objects
sequences = ProteinSequences.from_fasta("sequences.fasta")  # From FASTA file
sequences = ProteinSequences.from_a3m("alignment.a3m")  # From A3M file
sequences = ProteinSequences.from_list(["MKLL...", "MKLT..."])  # From strings
sequences = ProteinSequences.from_dict({"seq1": "MKLL...", "seq2": "MKLT..."})
sequences = ProteinSequences.from_dict(my_dataframe["sequence"].to_dict())
sequences = ProteinSequences.from_df(my_dataframe) # assumes first column is sequences
sequences = ProteinSequences.from_df(my_dataframe, sequence_col="seq_col", id_col="id_col") # specify columns
sequences, labels = ProteinSequences.from_df(my_dataframe, label_cols='label') # get a array of labels
```

Key attributes:

- `aligned`: Boolean indicating if all sequences have same length (including gaps)
- `fixed_length`: Boolean indicating if all sequences have same base length (excluding gaps)
- `width`: Length of sequences if aligned, None otherwise
- `has_gaps`: Boolean indicating if any sequence contains gaps
- `mutated_positions`: List of positions with variation (for aligned sequences)
- `weights`: Optional weights for each sequence (used in some models)
- `ids`: List of sequence IDs

Common operations:

```python
# Sequence alignment
aligned = sequences.align_all()  # Align all sequences
aligned = sequences.align_to(reference_msa)  # Align to existing MSA

# Access and manipulation
sequences[0]  # Access by index
sequences["seq1"]  # Access by ID
sequences.with_no_gaps()  # Remove gaps from all sequences
sequences.upper()  # Convert all to uppercase

# MSA operations
msa = sequences.msa_process(
    focus_seq_id="wild_type",
    theta=0.8  # Sequence reweighting parameter
)

# Sampling operations
sampled = sequences.sample(n=100, replace=False, keep_first=True)  # Sample sequences, keeping the first one

# Save/export
sequences.to_fasta("output.fasta")
sequences_dict = sequences.to_dict()

# Batching for large datasets
for batch in sequences.iter_batches(batch_size=32):
    # Process batch
    pass

# Get mapping between aligned and original positions
if sequences.aligned:
    alignment_mapping = sequences.get_alignment_mapping()
    # Returns dict mapping sequence IDs to lists of positions

# Convert to on-file representation for large MSAs
on_file = sequences.to_on_file("output.fasta")  # Save to file and return a file-based object
```

## ProteinStructure

`ProteinStructure` represents the 3D structure of a protein, integrating with common structure file formats and analysis tools.

```python
from aide_predict import ProteinStructure

# Create from PDB file
structure = ProteinStructure(
    pdb_file="protein.pdb",
    chain="A",  # Optional chain identifier
    plddt_file="confidence.json"  # Optional AlphaFold2 confidence scores
)

# Create from AlphaFold2 output folder
structure = ProteinStructure.from_af2_folder(
    folder_path="af2_results",
    chain="A"
)
```

Key methods:

```python
# Access structure information
sequence = structure.get_sequence()  # Get amino acid sequence
plddt = structure.get_plddt()  # Get pLDDT confidence scores
dssp = structure.get_dssp()  # Get secondary structure assignments

# Validation
structure.validate_sequence("MKLLVLGLPGAGKGT")  # Check if sequence matches structure

# Access underlying structure objects
structure_obj = structure.get_structure()  # Get BioPython Structure object
chain_obj = structure.get_chain()  # Get specific chain
positions = structure.get_residue_positions()  # Get residue numbers
```

## StructureMapper

`StructureMapper` helps manage multiple structures and map them to sequences, particularly useful when working with structure-aware models.

```python
from aide_predict import StructureMapper

# Initialize with folder containing structures
mapper = StructureMapper("path/to/structures")

# Structure can be PDB files or AlphaFold2 prediction folders
# Example folder structure:
# structures/
#   ├── protein1.pdb
#   ├── protein2.pdb
#   └── protein3/  # AlphaFold2 output folder
#       ├── ranked_0.pdb
#       └── ranking_confidence.json

# Assign structures to ProteinSequences already loaded 
sequences = mapper.assign_structures(sequences)

# Get available structures
available_ids = mapper.get_available_structures()

# Get ProteinSequences with structures
sequences = mapper.get_protein_sequences()
```

The StructureMapper is particularly useful when working with structure-aware models like SaProt, which can use structure information to improve predictions:

```python
# Example workflow with structure-aware model
mapper = StructureMapper("structures/")
sequences = ProteinSequences.from_fasta("sequences.fasta")
sequences = mapper.assign_structures(sequences)

model = SaProtLikelihoodWrapper(wt=sequences["wild_type"])
predictions = model.predict(sequences)  # Will use structures where available, falling back to the WT structure
```

## ProteinSequencesOnFile

`ProteinSequencesOnFile` provides a memory-efficient way to work with large alignments by only loading sequences when needed.

```python
from aide_predict import ProteinSequencesOnFile

# Create from FASTA file
on_file_sequences = ProteinSequencesOnFile("large_alignment.fasta")

# Access properties efficiently without loading everything
print(on_file_sequences.aligned)  # Check if aligned
print(on_file_sequences.width)    # Get width if aligned
print(len(on_file_sequences))     # Get count of sequences

# Load a specific sequence
seq = on_file_sequences[0]  # Get by index
seq = on_file_sequences["seq1"]  # Get by ID

# Iterate through sequences without loading all at once
for seq in on_file_sequences:
    process_sequence(seq)

# Load into memory if needed
in_memory = on_file_sequences.to_memory()
```

## ProteinTrajectory

NOT YET IMPLEMENTED