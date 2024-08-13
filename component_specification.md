# Component Specification: Protein Property Prediction Package

## 1. Overview

This software package provides a framework for protein property prediction tasks, designed to be compatible with scikit-learn's API. It includes base classes, mixins, and data structures to allow for easy wrapping of various protein-based models and transformers. This tool is NOT (yet) supporting of finetuning/training of protein language models. Here they are used as zero shot estimators or embedders. If you finetune such a model using your own code, you can easily target those weights with this package.

## 2. Core Components

### 2.1 ProteinModelWrapper

**File**: aide_predict/bespoke_models/base.py

**Description**: The base class for all protein-based models in the package.

**Key Features**:
- Inherits from `TransformerMixin` and `BaseEstimator` from scikit-learn
- Provides a standard interface for fitting, transforming, and predicting protein sequences
- Handles metadata and wild-type sequences

**Key Methods**:
```python
def __init__(self, metadata_folder: str = None, wt: Optional[Union[str, ProteinSequence]] = None)
def fit(self, X: Union[ProteinSequences, List[str]], y: Optional[np.ndarray] = None, force: bool = False) -> 'ProteinModelWrapper'
def transform(self, X: Union[ProteinSequences, List[str]]) -> np.ndarray
def predict(self, X: Union[ProteinSequences, List[str]]) -> np.ndarray
def partial_fit(self, X: Union[ProteinSequences, List[str]], y: Optional[np.ndarray] = None) -> 'ProteinModelWrapper'
```

**Abstract Methods**:
These should be implemented in subclasses:
```python
def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> None
def _transform(self, X: ProteinSequences) -> np.ndarray
```

### 2.2 LikelihoodTransformerBase

**File**: aide_predict/bespoke_models/predictors/pretrained_transformers.py

**Description**: Base class for likelihood based transformer models. Eg. any model that can outputs a (L, V) vector of probabilities where L is the length of the sequence and V is the vocabulary size. Abstracts marginal computation, positional indexing and pooling etc. The child must define the method to return the log probabilities of the model (with options for masking), how to get the likelihood  values by indexing the log probabilities given protein sequences and how to load and cleanup the model.

**Key Features**:
- Inherits from `PositionSpecificMixin`, `CanRegressMixin`, `RequiresWTDuringInferenceMixin`, and `ProteinModelWrapper`
- Provides methods for computing various types of marginal likelihoods: wildtype, mutant, and masked. See:
> Meier, J. et al. Language models enable zero-shot prediction of the effects of mutations on protein function. Preprint at https://doi.org/10.1101/2021.07.09.450648 (2021).

**Key Methods**:
```python
def __init__(self, metadata_folder: str = None, 
             marginal_method: MarginalMethod = MarginalMethod.WILDTYPE,
             positions: Optional[List[int]] = None, 
             flatten: bool = False,
             pool: bool = True,
             batch_size: int = 2,
             device: str = 'cpu',
             wt: Optional[Union[str, ProteinSequence]] = None)
def _transform(self, X: ProteinSequences) -> np.ndarray
```

**Abstract Methods**:
These should be implemented in subclasses. 
```python
def _compute_log_likelihoods(self, X: ProteinSequences, mask_positions: Optional[List[List[int]]] = None) -> List[np.ndarray]
def _index_log_probs(self, log_probs: np.ndarray, sequences: ProteinSequences) -> np.ndarray
def _load_model(self) -> None
def _cleanup_model(self) -> None
```

### 2.3 Mixins
These are key for minimizing effort in implementing new models and maintining expected behavior. They are designed to be inherited along with `ProteinModelWrapper`.

**File**: aide_predict/bespoke_models/base.py

**Description**: A set of mixins to extend the functionality of `ProteinModelWrapper`.

**Mixins**:
1. `RequiresMSAMixin`  If the model requires multiple sequence alignment to function, expected to be passed to fit.
2. `RequiresFixedLengthMixin` If the model requires fixed length sequences to function.
3. `CanRegressMixin` If the model is a regressor, eg. its outputs are expected to have some correlation with functional targets.
4. `RequiresWTDuringInferenceMixin` If the model requires the wildtype sequence at inference time in order to make wt comparison. This does not need to be mixed for any model that can pass any sequence X and the wild type sequence SEPERATELY to predict, and the noramlized score would be simply the difference between the two. In such a case, if WT is not none, this rudementary noramlization is used. If the model requires the WT to make a comparison, this should be mixed and the child class is expected to handle WT noramlization.
5. `RequiresWTToFunctionMixin` If the model requires the wildtype sequence to function, which is expected to be passed to init.
6. `PositionSpecificMixin` If the model is position specific, eg. it outputs a value for each position in the sequence. This mixin provided behavior to specify which positions to consider, and whether to output those scores seperately or pool them.
7. `CanHandleAlignedSequencesMixin` If the model can handle aligned sequences, eg. it can take a multiple sequence alignment as input __at predict time__
8. `CacheMixin` Utility mixing for any model that is expected to be expensive. Implements cacheing for proteins sequence scores/embeddingsm such that if that particular sequence is passed again, the model does not need to recompute the score. This should not be mixed for stochastic models.

**Key Methods** (for `PositionSpecificMixin`):
```python
def __init__(self, positions: bool = None, pool: bool = True, flatten: bool = True, *args, **kwargs)
def transform(self, X: Union[ProteinSequences, List[str]]) -> np.ndarray
```

### 2.4 ProteinSequence

**File**: aide_predict/utils/data_structures/sequences.py

**Description**: Represents a single protein sequence.

**Key Attributes**:
- `_id: Optional[str]`: The identifier of the sequence
- `_structure: Optional[Union[str, "ProteinStructure"]]`: The structure associated with the sequence
- `_characters: List[ProteinCharacter]`: List of individual amino acid characters

**Key Properties**:
- `id: Optional[str]`: Getter/setter for the sequence identifier
- `structure: Optional[Union[str, "ProteinStructure"]]`: Getter/setter for the sequence structure
- `has_gaps: bool`: Whether the sequence contains any gaps
- `has_non_canonical: bool`: Whether the sequence contains any non-canonical amino acids
- `as_array: np.ndarray`: The sequence as a numpy array of characters
- `num_gaps: int`: The number of gaps in the sequence
- `base_length: int`: The length of the sequence excluding gaps

**Key Methods**:
```python
def __new__(cls, seq: str, id: Optional[str] = None, structure: Optional[Union[str, "ProteinStructure"]] = None)
def mutate(self, position: int, new_character: str) -> 'ProteinSequence'
def mutated_positions(self, other: Union[str, 'ProteinSequence']) -> List[int]
def get_mutations(self, other: Union[str, 'ProteinSequence']) -> List[str]
def with_no_gaps(self) -> 'ProteinSequence'
def slice_as_protein_sequence(self, start: int, end: int) -> 'ProteinSequence'
def iter_protein_characters(self) -> Iterator[ProteinCharacter]
```

### 2.5 ProteinSequences

**File**: aide_predict/utils/data_structures/sequences.py

**Description**: Represents a collection of `ProteinSequence` objects.

**Key Attributes**:
- Inherits from `UserList`, so it has a `data` attribute containing the list of `ProteinSequence` objects

**Key Properties**:
- `aligned: bool`: True if all sequences have the same length (including gaps)
- `fixed_length: bool`: True if all sequences have the same base length (excluding gaps)
- `width: Optional[int]`: The length of the sequences if aligned, None otherwise
- `has_gaps: bool`: True if any sequence has gaps
- `mutated_positions: Optional[List[int]]`: List of positions that have more than one character across all sequences

**Key Methods**:
```python
def __init__(self, sequences: List[ProteinSequence])
def to_dict(self) -> Dict[str, str]
def to_fasta(self, output_path: str)
@classmethod
def from_fasta(cls, input_path: str) -> 'ProteinSequences'
@classmethod
def from_dict(cls, sequences: Dict[str, str]) -> 'ProteinSequences'
@classmethod
def from_list(cls, sequences: List[str]) -> 'ProteinSequences'
def align_all(self, output_fasta: Optional[str] = None) -> Union['ProteinSequences', 'ProteinSequencesOnFile']
def align_to(self, existing_alignment: Union['ProteinSequences', 'ProteinSequencesOnFile'], 
             realign: bool = False, return_only_new: bool = False,
             output_fasta: Optional[str] = None) -> Union['ProteinSequences', 'ProteinSequencesOnFile']
def with_no_gaps(self) -> 'ProteinSequences'
def iter_batches(self, batch_size: int) -> Iterable['ProteinSequences']
def get_id_mapping(self) -> Dict[str, int]
def get_alignment_mapping(self) -> Dict[str, List[Optional[int]]]
def apply_alignment_mapping(self, mapping: Dict[str, List[Optional[int]]]) -> 'ProteinSequences'
def as_array(self) -> np.ndarray
```

### 2.6 ProteinStructure

**File**: data_structures.py

**Description**: Represents the structure of a protein.

**Key Methods**:
```python
def __init__(self, pdb_file: str, chain: str = 'A', plddt_file: Optional[str] = None)
def get_sequence(self) -> str
def get_plddt(self) -> Optional[np.ndarray]
def get_dssp(self) -> Dict[str, str]
def validate_sequence(self, protein_sequence: str) -> bool
@classmethod
def from_af2_folder(cls, folder_path: str, chain: str = 'A') -> 'ProteinStructure'
```

## 3. Utility Components

### 3.1 Alignment Utilities

**File**: aide_predict/utils/alignment_calls.py

**Key Functions**:
```python
def sw_global_pairwise(seq1: "ProteinSequence", seq2: "ProteinSequence", matrix: str = 'BLOSUM62', gap_open: float = -10, gap_extend: float = -0.5) -> tuple['ProteinSequence', 'ProteinSequence']
def mafft_align(sequences: "ProteinSequences",
                existing_alignment: Optional["ProteinSequences"] = None,
                realign: bool = False,
                output_fasta: Optional[str] = None) -> "ProteinSequences"
```

### 3.2 Model Device Manager

**File**: aide_predict/bespoke_models/predictors/pretrained_transformers.py
Context manager that can be built into into models that can leverage advanced hardware to help with cleanup. For example, if one wanted to test two different transformer methods on their laptop GPU, it would be very easy to accidently have both models on device and throw a CUDA memory error. Desiging models that use this context manager makes it easy to only have the model on device a the time of inference, at a small cost of moving the model to the device each time.

**Key Classes and Methods**:
```python
class ModelDeviceManager:
    def __init__(self, model_instance: Any, device: str = 'cpu')
    @contextmanager
    def model_on_device(self, load_func: Callable[[], None], cleanup_func: Callable[[], None])

@contextmanager
def model_device_context(model_instance: Any, load_func: Callable[[], None], cleanup_func: Callable[[], None], device: str = 'cpu')
```

## 4. Bespoke Models

The package includes several bespoke models implemented using the `ProteinModelWrapper` and `LikelihoodTransformerBase` classes:

- ESM2 (Embeddings and Log Likelihood Predictor)
- MSA Transformer (Embeddings and Log Likelihood Predictor)
- SaProt (Embeddings and Log Likelihood Predictor)
- EVMutation (Predicts hamiltonian changes using pairwise potentials)

These models follow the structure and API defined by the base classes and mixins.

## 5. Integration and Usage

The components are designed to work together seamlessly:

1. Users create `ProteinSequence` and `ProteinSequences` objects from their data.
2. These objects are passed to the bespoke models, which inherit from `ProteinModelWrapper`.
3. The models use the appropriate mixins to define their behavior and requirements.
4. Utility functions like alignment calls can be used to preprocess the data when necessary.
5. The models can be used in scikit-learn pipelines or individually for fitting and prediction tasks.

## 6. Extensibility

The modular design allows for easy extension:

1. New bespoke models can be created by inheriting from `ProteinModelWrapper` and using the appropriate mixins.
2. Additional mixins can be created to define new behaviors or requirements for models.
3. The `ProteinSequence` and `ProteinSequences` classes can be extended to support new data formats or analysis methods.