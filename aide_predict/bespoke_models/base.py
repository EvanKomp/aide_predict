# aide_predict/bespoke_models/base.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Base classes for models to be wrapped into the API as sklearn estimators
'''
import os
from abc import abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, NotFittedError
import numpy as np

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.common import MessageBool

from typing import Union, Optional, List, Dict, Any

import logging
logger = logging.getLogger(__name__)

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
       - RequiresWTDuringInferenceMixin - if the model requires the wild type sequence duing inference in order to normalize by wt
       - PositionSpecificMixin - if the model can output per position scores
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
    _per_position_capable: bool = False
    _requires_fixed_length: bool = False
    _can_regress: bool = False
    _can_handle_aligned_sequences: bool = False
    _available: bool = MessageBool(True, "This model is available for use.")

    def __init__(self, metadata_folder: str, wt: Optional[Union[str, ProteinSequence]] = None):
        """
        Initialize the ProteinModelWrapper.

        Args:
            metadata_folder (str): The folder where the metadata is stored.
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence if present.

        Raises:
            ValueError: If the wild type sequence contains gaps.
        """
        self.metadata_folder = metadata_folder

        if not os.path.exists(metadata_folder):
            os.makedirs(metadata_folder)
            logger.info(f"Created metadata folder: {metadata_folder}")

        if not isinstance(wt, ProteinSequence) and wt is not None:
            wt = ProteinSequence(wt)
        
        if wt is not None:
            wt = ProteinSequence(wt)
            if wt.has_gaps:
                raise ValueError("Wild type sequence cannot have gaps.")
        
        self.wt = wt

        self.check_metadata()

    def check_metadata(self) -> None:
        """
        Ensures that everything this model class needs is in the metadata folder.
        """
        logger.warning("This model class did not implement check_metadata. If the model requires anything other than raw sequences to be fit, this is unexpected.")

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
    
    def fit(self, X: Union[ProteinSequences, List[str]], y: Optional[np.ndarray] = None) -> 'ProteinModelWrapper':
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
            logger.warning("Model is already fitted. Skipping")
            return self
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
        return {s: getattr(self, s) for s in self.__dict__.keys() if not s.startswith("_") and not callable(getattr(self, s)) and not s.endswith('_')}


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
    
    This mixin overrides the can_regress attribute to be True.
    """
    _can_regress: bool = True

class RequiresWTDuringInferenceMixin:
    """
    Mixin to ensure model requires wild type during inference.
    
    This mixin overrides the requires_wt_during_inference attribute to be True.
    """
    _requires_wt_during_inference: bool = True

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
        print(result.shape)
        print(self.positions)
        
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
    

