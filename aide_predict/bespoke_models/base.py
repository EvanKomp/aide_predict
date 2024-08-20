# aide_predict/bespoke_models/base.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Base classes for models to be wrapped into the API as sklearn estimators
'''
import os
from abc import abstractmethod
import time
import pickle
import warnings
import hashlib
import json
import hashlib
import tempfile

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, NotFittedError
import numpy as np

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool

from typing import Union, Optional, List, Dict, Any

import logging
logger = logging.getLogger(__name__)

def is_jsonable(x):
    """Checks if an object is JSON serializable."""
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

#############################################
# BASE CLASSES FOR DOWNSTREAM MODELS
#############################################
class ProteinModelWrapper(TransformerMixin, BaseEstimator):
    """
    Base class for bespoke models that take proteins as input.
    
    This class serves as a foundation for creating protein-based models that can be used
    in machine learning pipelines, particularly those compatible with scikit-learn.
    It provides a standard interface for fitting, transforming, and predicting protein sequences,
    as well as handling metadata and wild-type sequences.

    All models that take proteins as input should inherit from this class. They are considered
    transformers and can be used natively to produce features in the AIDE pipeline. Models can
    additionally be made regressors by inheriting from RegressorMixin.

    X values for fit, transform, and predict are expected to be ProteinSequences objects.

    Attributes:
        metadata_folder (str): The folder where the metadata is stored.
        wt (Optional[ProteinSequence]): The wild type sequence if present.
    
    Class Attributes:
        requires_msa_for_fit (bool): Whether the model requires an MSA as input for fitting.
        requires_wt_to_function (bool): Whether the model requires the wild type sequence to function.
        requires_wt_during_inference (bool): Whether the model requires the wild type sequence during inference.
        per_position_capable (bool): Whether the model can output per position scores.
        requires_fixed_length (bool): Whether the model requires a fixed length input.
        can_regress (bool): Whether the model outputs from transform can also be considered estimates of activity label.
        can_handle_aligned_sequences (bool): Whether the model can handle unaligned sequences at predict time.
        _available (bool): Flag to indicate whether the model is available for use.

    To subclass ProteinModelWrapper:
    1. Implement the abstract methods:
       - _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> None
       - _transform(self, X: ProteinSequences) -> np.ndarray
    2. If your model supports partial fitting, implement:
       - _partial_fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> None
    3. If your model requires specific metadata, override:
       - check_metadata(self) -> None
       - _construct_necessary_metadata(cls, model_directory: str, necessary_metadata: dict) -> None
    4. If your model has additional parameters, implement __init__ and call super().__init__
       with the metadata_folder and wt arguments.
    5. If your model requires specific behavior, consider inheriting from the provided mixins. See the mixins for
       the provided behaviors:
       - RequiresMSAMixin - if the model requires an MSA for fitting
       - RequiresFixedLengthMixin - if the model requires fixed length sequences at predict time
       - CanRegressMixin - if the model can regress, otherwise it is assumed to be a transformer only eg. embedding
       - RequiresWTToFunctionMixin - if the model requires the wild type sequence to function
       - RequiresWTDuringInferenceMixin - if the model requires the wild type sequence duing inference in order to normalize by wt
       - PositionSpecificMixin - if the model can output per position scores
       - RequiresStructureMixin - if the model requires structure information
       - AcceptsLowerCaseMixin - if the model can accept lower case sequences
    6. If the model requires more than the base package, set the _available attribute to be dynamic based on a check in the module.

    Example:
        ESM2 using WT marginal can be used as a "regressor".

        try:
            import transformers
            AVALABLE = MessageBool(True, "This model is available.")
        except ImportError:
            AVALABLE = MessageBool(False, "This model is not available, make sure transformers is installed.")

        class ESM2Model(CanRegressMixin, PositionSpecificMixin, ProteinModelWrapper):
            _available = AVAILABLE

            def __init__(self, model_checkpoint: str, metadata_folder: str, wt: Optional[Union[str, ProteinSequence]] = None):
                super().__init__(metadata_folder, wt)
                self.model_checkpoint = model_checkpoint

            def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> None:
                # Fit the model
                ...
                return self

            def _transform(self, X: ProteinSequences) -> np.ndarray:
                # Transform the sequences
                ...
                return outputs

    """

    _requires_msa_for_fit: bool = False
    _requires_wt_during_inference: bool = False
    _requires_wt_to_function: bool = False
    _per_position_capable: bool = False
    _requires_fixed_length: bool = False
    _can_regress: bool = False
    _can_handle_aligned_sequences: bool = False
    _requires_structure: bool = False
    _accepts_lower_case: bool = False
    _available: bool = MessageBool(True, "This model is available for use.")

    def __init__(self, metadata_folder: str=None, wt: Optional[Union[str, ProteinSequence]] = None):
        """
        Initialize the ProteinModelWrapper.

        Args:
            metadata_folder (str): The folder where the metadata is stored.
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence if present.

        Raises:
            ValueError: If the wild type sequence contains gaps.
        """
        if not self._available:
            raise ValueError(self._available.message)
        
        # generate a folder if metadata is None
        self._metadata_folder = metadata_folder
        self._processed_metadata_folder = self._process_metadata_folder(metadata_folder)
        self._ensure_metadata_folder()

        self._wt = wt
        self._processed_wt = self._process_wt(wt)

        if self._processed_wt is None and self.requires_wt_to_function:
            raise ValueError("This model requires a wild type sequence to function.")

        self.check_metadata()

    @property
    def wt(self):
        return self._processed_wt

    @wt.setter
    def wt(self, value):
        self._wt = value
        self._processed_wt = self._process_wt(value)

    def _process_wt(self, wt):
        if wt is None:
            return None
        if not isinstance(wt, ProteinSequence):
            wt = ProteinSequence(wt)
        if wt.has_gaps:
            raise ValueError("Wild type sequence cannot have gaps.")
        processed_wt = wt.upper()
        if self.requires_structure and processed_wt.structure is None:
            raise ValueError("This model acts on structure but a wild type structure was not given.")
        return processed_wt

    def check_metadata(self) -> None:
        """
        Ensures that everything this model class needs is in the metadata folder.
        """
        pass

    def _validate_input(self, X: Union[ProteinSequences, List[str]]) -> ProteinSequences:
        """
        Validate and convert input to ProteinSequences.

        Args:
            X (Union[ProteinSequences, List[str]]): Input sequences.

        Returns:
            ProteinSequences: Validated input sequences.
        """
        if not isinstance(X, ProteinSequences):
            X = ProteinSequences([ProteinSequence(seq) for seq in X])

        if X.has_lower() and not self.accepts_lower_case:
            X = X.upper()
        return X

    def _assert_aligned(self, X: ProteinSequences) -> None:
        """
        Assert that input sequences are aligned if required.

        Args:
            X (ProteinSequences): Input sequences.

        Raises:
            ValueError: If input sequences are not aligned and alignment is required.
        """
        if not X.aligned:
            raise ValueError("Input sequences must be aligned for this model.")

    def _assert_fixed_length(self, X: ProteinSequences) -> None:
        """
        Assert that input sequences are of fixed length if required.

        Args:
            X (ProteinSequences): Input sequences.

        Raises:
            ValueError: If input sequences are not of fixed length and fixed length is required.
        """
        if not X.fixed_length:
            raise ValueError("Input sequences must be aligned and of fixed length for this model.")
        if self.wt is not None and len(self.wt) != X.width:
            raise ValueError("Wild type sequence must be the same length as the sequences.")
        
    def _check_fixed_length(self, X: ProteinSequences) -> bool:
        """
        Check if the input sequences are of fixed length.

        Args:
            X (ProteinSequences): The input sequences.

        Returns:
            bool: True if sequences are of fixed length, False otherwise.
        """
        if not X.fixed_length:
            return False
        if self.wt is not None and len(self.wt) != X.width:
            return False
        return True
    
    def _enforce_aligned(self, X: ProteinSequences) -> ProteinSequences:
        """
        Enforce alignment of input sequences if not already aligned.

        Args:
            X (ProteinSequences): Input sequences.

        Returns:
            ProteinSequences: Aligned input sequences.
        """
        if not X.aligned:
            logger.info("Input sequences are not aligned. Performing alignment.")
            return X.align_all()
        return X

    @abstractmethod
    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the model. Must be implemented by child classes.

        Args:
            X (ProteinSequences): Input sequences.
            y (Optional[np.ndarray]): Target values.
        """
        raise NotImplementedError("This method must be implemented in the child class.")
    
    def fit(self, X: Union[ProteinSequences, List[str]], y: Optional[np.ndarray] = None, force: bool=False) -> 'ProteinModelWrapper':
        """
        Fit the model.
        
        Args:
            X (Union[ProteinSequences, List[str]]): Input sequences.
            y (Optional[np.ndarray]): Target values.

        Returns:
            ProteinModelWrapper: The fitted model.
        """
        try:
            check_is_fitted(self)
            if not force:
                logger.warning("Model is already fitted. Skipping")
                return self
            else:
                pass
        except NotFittedError:
            pass

        logger.info(f"Fitting {self.__class__.__name__}")
        X = self._validate_input(X)
        
        if self.requires_msa_for_fit:
            X = self._enforce_aligned(X)
        
        self._fit(X, y)
        
        return self
    
    def partial_fit(self, X: Union[ProteinSequences, List[str]], y: Optional[np.ndarray] = None) -> 'ProteinModelWrapper':
        """
        Partially fit the model to the given sequences.

        This method can be called multiple times to incrementally fit the model.

        Args:
            X (Union[ProteinSequences, List[str]]): The input sequences to partially fit the model on.
            y (Optional[np.ndarray]): The target values, if applicable.

        Returns:
            ProteinModelWrapper: The partially fitted model.
        """
        logger.info(f"Partial fitting {self.__class__.__name__}")
        X = self._validate_input(X)
        
        if self.requires_msa_for_fit:
            X = self._enforce_aligned(X)

        self._partial_fit(X, y)
        return self

    def _partial_fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> None:
        """
        Partially fit the model. Must be implemented by child classes that support partial fitting.

        Args:
            X (ProteinSequences): Input sequences.
            y (Optional[np.ndarray]): Target values.
        """
        raise NotImplementedError("Partial fitting is not implemented for this model.")
    
    @abstractmethod
    def _transform(self, X: ProteinSequences) -> np.ndarray:
        """
        Transform the sequences. Must be implemented by child classes.

        Args:
            X (ProteinSequences): Input sequences.

        Returns:
            np.ndarray: Transformed sequences.
        """
        raise NotImplementedError("This method must be implemented in the child class.")
        
    def transform(self, X: Union[ProteinSequences, List[str]]) -> np.ndarray:
        """
        Transform the sequences.

        Args:
            X (Union[ProteinSequences, List[str]]): Input sequences.

        Returns:
            np.ndarray: Transformed sequences.
        """
        logger.info(f"Transforming input using {self.__class__.__name__}")
        check_is_fitted(self)
        X = self._validate_input(X)

        if self.requires_fixed_length:
            self._assert_fixed_length(X)

        if self.requires_structure:
            if any(seq.structure is None for seq in X) and self.wt is None:
                raise ValueError("This model requires structure information, at least one of the sequences does not have it, and there is no avialable WT structure.")
            elif any(seq.structure is None for seq in X):
                if X.fixed_length and len(self.wt) == X.width:
                    pass
                else:
                    raise ValueError("This model requires structure information, at least one of the sequences does not have it, and the WT structure size does not match the sequence lengths.")

        if not self.can_handle_aligned_sequences and X.has_gaps:
            logger.info("Input sequences have gaps and the model cannot handle them. Removing gaps.")
            X = X.with_no_gaps()
        
        outputs = self._transform(X)
        if not self.requires_wt_during_inference and self.wt is not None:
            wt_output = self._transform(ProteinSequences([self.wt]))
            outputs -= wt_output
        return outputs
    
    def predict(self, X: Union[ProteinSequences, List[str]]) -> np.ndarray:
        """
        Predict the sequences.

        Args:
            X (Union[ProteinSequences, List[str]]): Input sequences.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            ValueError: If the model is not capable of regression.
        """
        if not self.can_regress:
            raise ValueError("This model is not capable of regression.")
        return self.transform(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.
        """
        params = {s: getattr(self, s) for s in self.__dict__.keys() if not s.startswith("_") and not callable(getattr(self, s)) and not s.endswith('_')}
        params['metadata_folder'] = self._metadata_folder
        params['wt'] = self._wt
        return params


    def set_params(self, **params: Any) -> 'ProteinModelWrapper':
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            ProteinModelWrapper: Estimator instance.
        """

        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (Optional[List[str]]): Input feature names.

        Returns:
            List[str]: Output feature names.
        """
        check_is_fitted(self)
        header = f"{self.__class__.__name__}"
        return [header]

    @property
    def requires_msa_for_fit(self) -> bool:
        """Whether the model requires an MSA for fitting."""
        return self._requires_msa_for_fit
    
    @property
    def per_position_capable(self) -> bool:
        """Whether the model can output per position scores."""
        return self._per_position_capable
    
    @property
    def requires_fixed_length(self) -> bool:
        """Whether the model requires fixed length input."""
        return self._requires_fixed_length
    
    @property
    def requires_wt_during_inference(self) -> bool:
        """Whether the model requires the wild type sequence during inference."""
        return self._requires_wt_during_inference

    @property
    def can_regress(self) -> bool:
        """Whether the model can perform regression."""
        return self._can_regress
    
    @property
    def can_handle_aligned_sequences(self) -> bool:
        """Whether the model can handle aligned sequences (with gaps) at predict time."""
        return self._can_handle_aligned_sequences
    
    @property
    def requires_wt_to_function(self) -> bool:
        """Whether the model requires the wild type sequence to function."""
        return self._requires_wt_to_function
    
    @property
    def requires_structure(self) -> bool:
        """Whether the model requires structure information."""
        return self._requires_structure
    
    @property
    def accepts_lower_case(self) -> bool:
        """Whether the model can accept lower case sequences."""
        return self._accepts_lower_case

    @staticmethod
    def _construct_necessary_metadata(model_directory: str, necessary_metadata: dict) -> None:
        """
        Construct the necessary metadata for a model.

        Args:
            model_directory (str): The directory to store the metadata.
            necessary_metadata (dict): Dictionary of necessary metadata.
        """
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
            logger.info(f"Created model directory: {model_directory}")
        logger.warning("This model class did not implement _construct_necessary_metadata. If the model requires anything other than raw sequences to be fit, this is unexpected.")

    @property
    def metadata_folder(self):
        return self._processed_metadata_folder

    @metadata_folder.setter
    def metadata_folder(self, value):
        self._metadata_folder = value
        self._processed_metadata_folder = self._process_metadata_folder(value)

    def _process_metadata_folder(self, metadata_folder):
        if metadata_folder is None:
            prefered_tempdir = tempfile.gettempdir()
            return os.path.join(prefered_tempdir, time.strftime("%Y%m%d_%H%M%S"))
        return os.path.abspath(metadata_folder)
    
    def _ensure_metadata_folder(self):
        if self.metadata_folder and not os.path.exists(self.metadata_folder):
            os.makedirs(self.metadata_folder)
            logger.info(f"Created metadata folder: {self.metadata_folder}")

    @classmethod
    def from_basic_info(
        cls,
        model_directory: str,
        necessary_metadata: dict = {},
        wt: Optional[str] = None,
        **kwargs
    ) -> 'ProteinModelWrapper':
        """
        Construct the required metadata for a model from basic information and instantiate.
        
        Args:
            model_directory (str): The directory to store the metadata in.
            necessary_metadata (dict): A dictionary of necessary metadata to construct.
            wt (Optional[str]): The wild type sequence.
            **kwargs: Additional arguments to pass to the model class.

        Returns:
            ProteinModelWrapper: The instantiated model.
        """
        cls._construct_necessary_metadata(model_directory, necessary_metadata)
        return cls(metadata_folder=model_directory, wt=wt, **kwargs)

#############################################
# MIXINS FOR MODELS
#############################################

class RequiresMSAMixin:
    """
    Mixin to ensure model receives aligned sequences at fit.

    This mixin overrides the requires_msa_for_fit attribute to be True.
    """
    _requires_msa_for_fit: bool = True

class RequiresFixedLengthMixin:
    """
    Mixin to ensure model receives fixed length sequences at transform.
    
    This mixin overrides the requires_fixed_length attribute to be True.
    """
    _requires_fixed_length: bool = True

class CanRegressMixin(RegressorMixin):
    """
    Mixin to ensure model can regress.
    
    This mixin overrides the can_regress attribute to be True. It also overrides the score method to use
    spearman correlation isntead of R2, such that it can be used out of the mox with zero shot predicors.
    """
    _can_regress: bool = True

    def score(self, X, y, sample_weight=None):
        """
        Return the Spearman correlation
        """
        from scipy.stats import spearmanr
        y_pred = self.predict(X)
        return spearmanr(y, y_pred).correlation
    
class RequiresStructureMixin:
    """
    Mixin to ensure model requires structure information.
    
    This mixin overrides the requires_structure attribute to be True.
    """
    _requires_structure: bool = True


class RequiresWTDuringInferenceMixin:
    """
    Mixin to ensure model requires wild type during inference.
    
    This mixin overrides the requires_wt_during_inference attribute to be True.
    """
    _requires_wt_during_inference: bool = True

class RequiresWTToFunctionMixin:
    """
    Mixin to ensure model requires wild type to function.
    
    This mixin overrides the requires_wt_to_function attribute to be True.
    """
    _requires_wt_to_function: bool = True

class PositionSpecificMixin:
    """
    Mixin for protein models that can output per position scores.
    
    This mixin:
    1. Overrides the per_position_capable attribute to be True.
    2. Checks that positions, pool, and flatten are attributes.
    3. Wraps the predict and transform methods to check that if positions were passed and not pooling, the output is the same length as the positions.
    4. Flattens the output if flatten is True.

    Note that you are responsible for selecting positions and pooling. This mixing only provides checks that
    the output is consistent with the specified positions. You DO NOT need to implement flattening, as this mixin
    will handle it for you.

    Attributes:
        positions (Optional[List[int]]): The positions to output scores for.
        pool (bool): Whether to pool the scores across positions.
        flatten (bool): Whether to flatten dimensions beyond the second dimension.
    """
    _per_position_capable: bool = True

    def __init__(self, positions: bool=None, pool: bool=True, flatten: bool=True, *args, **kwargs):
        """
        Initialize the PositionSpecificMixin.

        Raises:
            ValueError: If the model does not have a positions, pool, or flatten attribute.
        """
        self.positions = positions
        self.pool = pool
        self.flatten = flatten
        super().__init__(*args, **kwargs)

    def _is_ragged_array(self, arr):
        """Check if the input is a ragged array (list of arrays with different shapes)."""
        if not isinstance(arr, list):
            return False
        if len(arr) == 0:
            return False
        first_shape = arr[0].shape
        return any(a.shape != first_shape for a in arr[1:])

    def transform(self, X: Union[ProteinSequences, List[str]]) -> np.ndarray:
        """
        Transform the sequences, ensuring correct output dimensions for position-specific models.
        If flatten is True, flatten dimensions beyond the second dimension.

        Args:
            X (Union[ProteinSequences, List[str]]): Input sequences.

        Returns:
            np.ndarray: Transformed sequences.

        Raises:
            ValueError: If the output dimensions do not match the specified positions.
        """
        result = super().transform(X)

        # we need to determine if the output is a clean numpy array or a set of arrays of different sizes
        if self._is_ragged_array(result):
            warnings.warn("The output is a ragged array of embeddings of different sizes, cannot output as an array. Ignoring flatten.")
            return result
        else:
            if type(result) is list:
                if all(a.shape[0] == 1 for a in result):
                    result = np.vstack(result)
                else:
                    result = np.array(result)
        
        if self.positions is not None and not self.pool:
            dims = len(self.positions)
            if result.shape[1] != dims:
                raise ValueError(f"The output second dimension must have the same length as number of positions. Expected {dims}, got {result.shape[1]}.")
        
        if self.flatten and result.ndim > 2:
            result = result.reshape(result.shape[0], -1)
        
        return result
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation, considering position-specific output and flattening.

        Args:
            input_features (Optional[List[str]]): Input feature names.

        Returns:
            List[str]: Output feature names.
        """
        check_is_fitted(self)
        header = f"{self.__class__.__name__}"
        
        if self.positions is not None and not self.pool:
            base_names = [f"{header}_{pos}" for pos in self.positions]
            if self.flatten:
                # Since we don't know the exact shape of the output at this point,
                # we'll use a placeholder for additional dimensions
                return [f"{name}_dim{i}" for name in base_names for i in range(getattr(self, 'output_dim', 1))]
            return base_names
        
        return [header]
    
class CanHandleAlignedSequencesMixin:
    """
    Mixin to indicate that a model can handle aligned sequences (with gaps) during prediction.
    
    This mixin overrides the can_handle_aligned_sequences attribute to be True.
    """
    _can_handle_aligned_sequences: bool = True
    
class CacheMixin:
    """
    Mixin to provide per-protein caching functionality for ProteinModelWrapper subclasses.
    """
    def __init__(self, *args, use_cache: bool=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = use_cache
        self._cache_file = os.path.join(self.metadata_folder, 'model_cache.pkl')
        self._cache_metadata_file = os.path.join(self.metadata_folder, 'cache_metadata.json')
        self._cache = {}
        self._cache_metadata = {}

    def _get_model_state_hash(self) -> str:
        """Generate a hash representing the current state of the model."""
        model_params = json.dumps(self.get_params(), sort_keys=True)
        fitted_attrs = json.dumps({attr: getattr(self, attr) for attr in self.get_fitted_attributes()}, sort_keys=True)
        return hashlib.md5((model_params + fitted_attrs).encode()).hexdigest()

    def _get_protein_hash(self, protein: ProteinSequence) -> str:
        hash_input = str(protein)
        if protein.id is not None:
            hash_input += f"|id:{protein.id}"
        if protein.structure is not None:
            hash_input += f"|structure:{str(protein.structure.pdb_file)}{protein.structure.chain}{protein.structure.plddt_file}"
        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _load_cache(self):
        """Load the cache from disk if it exists."""
        if os.path.exists(self._cache_file) and os.path.exists(self._cache_metadata_file):
            with open(self._cache_file, 'rb') as f:
                self._cache = pickle.load(f)
            with open(self._cache_metadata_file, 'r') as f:
                self._cache_metadata = json.load(f)

    def _save_cache(self):
        """Save the cache to disk."""
        with open(self._cache_file, 'wb') as f:
            pickle.dump(self._cache, f)
        with open(self._cache_metadata_file, 'w') as f:
            json.dump(self._cache_metadata, f)

    def _clear_cache(self):
        """Clear the cache and remove cache files."""
        self._cache = {}
        self._cache_metadata = {}
        if os.path.exists(self._cache_file):
            os.remove(self._cache_file)
        if os.path.exists(self._cache_metadata_file):
            os.remove(self._cache_metadata_file)

    def get_fitted_attributes(self) -> List[str]:
        """
        Get a list of attributes that are set during fitting.
        This method automatically detects attributes ending with an underscore.
        """
        return [attr for attr in dir(self) if attr.endswith('_') and not attr.startswith('_') and is_jsonable(getattr(self, attr))]

    def transform(self, X: Union[ProteinSequences, List[str]]) -> np.ndarray:
        """Override transform to use cache when possible on a per-protein basis."""
        if not self.use_cache:
            return super().transform(X)

        try:
            check_is_fitted(self)
        except:
            return super().transform(X)

        X = self._validate_input(X)
        self._load_cache()

        current_model_state = self._get_model_state_hash()
        if self._cache_metadata.get('model_state') != current_model_state:
            warnings.warn("Cache found but model state has changed. Clearing cache.")
            self._clear_cache()
            self._cache_metadata['model_state'] = current_model_state

        cached_results = []
        proteins_to_process = []
        original_order = []

        for i, protein in enumerate(X):
            protein_hash = self._get_protein_hash(protein)
            if protein_hash in self._cache:
                cached_results.append((i, self._cache[protein_hash]))
            else:
                proteins_to_process.append((i, protein))
            original_order.append(i)

        if proteins_to_process:
            proteins_to_transform = ProteinSequences([p[1] for p in proteins_to_process])
            new_results = super().transform(proteins_to_transform)

            for (i, protein), result in zip(proteins_to_process, new_results):
                protein_hash = self._get_protein_hash(protein)
                # make sure we keep our dims
                if not isinstance(result, np.ndarray):
                    result = np.array(result).reshape(1, -1)
                if result.shape[0] != 1:
                    result = np.expand_dims(result, axis=0)
                self._cache[protein_hash] = result
                self._cache_metadata[protein_hash] = {
                    'timestamp': time.time(),
                    'protein_length': len(protein),
                    'output_shape': result.shape
                }
                cached_results.append((i, result))

        self._save_cache()

        # Reorder results to match original input order
        all_results = sorted(cached_results, key=lambda x: x[0])
        only_outputs = [r[1] for r in all_results]
        # determine if we can stack or not
        if len(set([o.shape for o in only_outputs])) == 1:
            return np.vstack(only_outputs)
        else:
            return only_outputs


class AcceptsLowerCaseMixin:
    """
    Mixin to indicate that a model can accept lower case sequences.

    This mixin overrides the accepts_lower_case attribute to be True.
    """
    _accepts_lower_case: bool = True

