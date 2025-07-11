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
import sqlite3
import h5py

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
        expects_no_fit (bool): Whether the model expects no fit.
        requires_msa_for_fit (bool): Whether the model requires an MSA as input for fitting.
        requires_wt_msa (bool): Whether the model requires a wild type MSA for fitting.
        requires_msa_per_sequence (bool): Whether the model requires an MSA for each sequence during transform.
        requires_wt_to_function (bool): Whether the model requires the wild type sequence to function.
        requires_wt_during_inference (bool): Whether the any wt is used direectly in model inference, otherwise normalize by any wt
        per_position_capable (bool): Whether the model can output per position scores.
        requires_fixed_length (bool): Whether the model requires a fixed length input.
        can_regress (bool): Whether the model outputs from transform can also be considered estimates of activity label.
        can_handle_aligned_sequences (bool): Whether the model can handle unaligned sequences at predict time.
        should_refit_on_sequences (bool): Whether the model should refit on new sequences when given.
        requires_structure (bool): Whether the model requires structure information.
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
       - ExpectsNoFitMixin - if the model expects no fit
       - RequiresMSAForFitMixin - if the model requires aligned sequences at fit time
       - RequiresWTMSAMixin - if the model requires a wild type MSA for fit
       - RequiresMSAPerSequenceMixin - if the model requires an MSA for each sequence during transform
       - RequiresFixedLengthMixin - if the model requires fixed length sequences at predict time
       - CanRegressMixin - if the model can regress, otherwise it is assumed to be a transformer only eg. embedding
       - RequiresWTToFunctionMixin - if the model requires the wild type sequence to function
       - RequiresWTDuringInferenceMixin - if the model requires the wild type sequence duing inference in order to normalize by wt
       - PositionSpecificMixin - if the model can output per position scores
       - RequiresStructureMixin - if the model requires structure information
       - AcceptsLowerCaseMixin - if the model can accept lower case sequences
       - ShouldRefitOnSequencesMixin - if the model should refit on new sequences when given. Often, we are calling fit on NOT raw sequences, eg. MSAs.
          We still want to be able to use the model in the context of sklearn pipelines which will attempt to clone and refit the model on X data.
          We want the models to return themselves already fitted when cloned, unless this is mixex in
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
    _expects_no_fit: bool = False
    _requires_msa_for_fit: bool = False
    _requires_wt_msa: bool = False
    _requires_msa_per_sequence: bool = False
    _requires_wt_during_inference: bool = False
    _requires_wt_to_function: bool = False
    _per_position_capable: bool = False
    _requires_fixed_length: bool = False
    _can_regress: bool = False
    _can_handle_aligned_sequences: bool = False
    _requires_structure: bool = False
    _accepts_lower_case: bool = False
    _should_refit_on_sequences: bool = False
    _available: bool = MessageBool(True, "This model is available for use.")

    # Mixin registration for processing hooks
    _mixin_init_handlers = []
    _pre_fit_hooks = []
    _post_fit_hooks = []
    _pre_transform_hooks = []
    _post_transform_hooks = []

    def __init_subclass__(cls, **kwargs):
        """
        Initialize the subclass and register mixin hooks.
        
        This method is called when a subclass is created. It collects and registers
        all hooks from mixins in the inheritance chain, avoiding duplicates.
        """
        super().__init_subclass__(**kwargs)
        
        # Reset hooks for this subclass
        cls._mixin_init_handlers = []
        cls._pre_fit_hooks = []
        cls._post_fit_hooks = []
        cls._pre_transform_hooks = []
        cls._post_transform_hooks = []
        
        # Keep track of function objects we've already added to avoid duplicates
        seen_init_handlers = set()
        seen_pre_fit_hooks = set()
        seen_post_fit_hooks = set()
        seen_pre_transform_hooks = set()
        seen_post_transform_hooks = set()
        
        # Collect hooks from all mixin bases
        for base in cls.__mro__[1:]:  # Skip self, look at all parents
            # Init handlers
            if hasattr(base, '_init_handler') and callable(base._init_handler):
                handler = base._init_handler
                if handler not in seen_init_handlers:
                    cls._mixin_init_handlers.append(handler)
                    seen_init_handlers.add(handler)
            
            # Fit hooks
            if hasattr(base, '_pre_fit_hook') and callable(base._pre_fit_hook):
                hook = base._pre_fit_hook
                if hook not in seen_pre_fit_hooks:
                    cls._pre_fit_hooks.append(hook)
                    seen_pre_fit_hooks.add(hook)
                    
            if hasattr(base, '_post_fit_hook') and callable(base._post_fit_hook):
                hook = base._post_fit_hook
                if hook not in seen_post_fit_hooks:
                    cls._post_fit_hooks.append(hook)
                    seen_post_fit_hooks.add(hook)
            
            # Transform hooks
            if hasattr(base, '_pre_transform_hook') and callable(base._pre_transform_hook):
                hook = base._pre_transform_hook
                if hook not in seen_pre_transform_hooks:
                    cls._pre_transform_hooks.append(hook)
                    seen_pre_transform_hooks.add(hook)
                    
            if hasattr(base, '_post_transform_hook') and callable(base._post_transform_hook):
                hook = base._post_transform_hook
                if hook not in seen_post_transform_hooks:
                    cls._post_transform_hooks.append(hook)
                    seen_post_transform_hooks.add(hook)
            
            # Stop at ProteinModelWrapper (don't include its parents)
            if base is ProteinModelWrapper:
                break

    def __init__(self, metadata_folder: str=None, wt: Optional[Union[str, ProteinSequence]] = None, **kwargs):
        """
        Initialize the ProteinModelWrapper with core attributes and process mixin initializations.

        Args:
            metadata_folder (str): The folder where the metadata is stored.
            wt (Optional[Union[str, ProteinSequence]]): The wild type sequence if present.
            **kwargs: Additional keyword arguments handled by mixins.

        Raises:
            ValueError: If the wild type sequence contains gaps or if the model is not available.
        """
        if not self._available:
            raise ValueError(self._available.message)
        
        # Process core attributes
        self._metadata_folder = metadata_folder
        self._processed_metadata_folder = self._process_metadata_folder(metadata_folder)
        self._ensure_metadata_folder()
        self._wt = wt
        self._processed_wt = self._process_wt(wt)
        
        # Process mixin initializations - let each mixin extract its own parameters
        mixin_kwargs = kwargs.copy()
        for handler in self._mixin_init_handlers:
            handler(self, **mixin_kwargs)
        
        # Store any remaining kwargs as object attributes
        for key, value in mixin_kwargs.items():
            setattr(self, key, value)
        
        # Final validations
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
        
        if self.requires_wt_msa and processed_wt.msa is None:
            raise ValueError("This model requires an MSA for fitting, but the wild type sequence does not have one.")
        # Add this check:
        elif self.requires_wt_msa and not processed_wt.msa_same_width:
            raise ValueError("Wild type sequence and its MSA have different widths.")
        
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
            # check if it is a list of str
            if not hasattr(X, '__getitem__'):
                raise ValueError("Input must be a ProteinSequences object or an iterable of ProteinSequence like")
            if len(X) == 0:
                return ProteinSequences([])
            if type(X[0]) is str:
                X = ProteinSequences([ProteinSequence(seq) for seq in X])
            else:
                X = ProteinSequences(list(X))

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
    
    def __sklearn_clone__(self, safe: bool = True):
        """Overwrite the sklearn clone method to avoid resetting fitted attributes that are determined from data other than
        raw sequences, eg. MSA"""
        if self.should_refit_on_sequences:
            from sklearn.base import _clone_parametrized
            return _clone_parametrized(self, safe=safe)
        else:
            return self
    
    def fit(self, X: Union[ProteinSequences, List[str]] = None, y: Optional[np.ndarray] = None, force: bool=False) -> 'ProteinModelWrapper':
        """
        Fit the model with pre-processing and post-processing from mixins.
        
        Args:
            X (Union[ProteinSequences, List[str]]): Input sequences.
            y (Optional[np.ndarray]): Target values.
            force (bool): Whether to force refitting if already fitted.

        Returns:
            ProteinModelWrapper: The fitted model.
        """
        empty_X = X is None or (hasattr(X, '__len__') and len(X) == 0)
        
        # Handle no-fit models
        if self.expects_no_fit:
            if empty_X:
                pass
            else:
                warnings.warn(f"This model expects no fit, but received input. Ignoring input: {X}")
            self._fit(None, None)
            return self
            
        # Normal fitting path
        try:
            check_is_fitted(self)
            if not force and not self.should_refit_on_sequences:
                logger.warning("Model is already fitted. Skipping")
                return self
        except NotFittedError:
            pass

        logger.info(f"Fitting {self.__class__.__name__}")
        
        # Validate input
        X = self._validate_input(X) if not empty_X else X
        
        # Run pre-fit hooks from mixins
        for hook in self._pre_fit_hooks:
            X, y = hook(self, X, y)
        
        # Handle MSA requirements, structure checks, etc.
        if not empty_X:
            if self.requires_msa_for_fit:
                if empty_X:
                    if self.wt is not None and self.wt.has_msa:
                        warnings.warn("No input sequences provided, but the wild type sequence has an MSA. Attempting to use the wild type sequence MSA.")
                        X = self.wt.msa
                    else:
                        raise ValueError("No input sequences provided and the wild type sequence does not have an MSA. Cannot fit model.")

                X = self._enforce_aligned(X)

            if self.requires_structure:
                if any(seq.structure is None for seq in X) and self.wt is None:
                    raise ValueError("This model requires structure information, at least one of the sequences does not have it, and there is no avialable WT structure.")
                elif any(seq.structure is None for seq in X):
                    if X.fixed_length and len(self.wt) == X.width:
                        pass
                    else:
                        raise ValueError("This model requires structure information, at least one of the sequences does not have it, and the WT structure size does not match the sequence lengths.")
            
            if self.requires_msa_per_sequence:
                if any(not seq.has_msa for seq in X):
                    if self.wt is not None and self.wt.has_msa:
                        warnings.warn("Some sequences do not have an MSA, but the wild type sequence does. Attempting to use the wild type sequence MSA.")
                        wt_msa = self.wt.msa
                        for seq in X:
                            seq.msa = wt_msa
                    else:
                        raise ValueError("Some sequences do not have an MSA, and the wild type sequence does not have one either. Cannot fit model.")
                    
                if any(not seq.msa_same_width for seq in X) and not self.can_handle_aligned_sequences:
                    raise ValueError("Not all sequence MSAs have the same width as the sequence itself.")
                else:
                    # align the sequence itself to the MSA
                    new_X = []
                    for seq in X:
                        if not seq.msa_same_width:
                            seq = seq.align(seq.msa)
                            assert seq.msa_same_width
                        new_X.append(seq)
                    X = ProteinSequences(new_X)

        
        # Call the actual fit implementation
        self._fit(X, y)
        
        # Run post-fit hooks from mixins
        for hook in self._post_fit_hooks:
            hook(self, X, y)
        
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

        if self.requires_structure:
            if any(seq.structure is None for seq in X) and self.wt is None:
                raise ValueError("This model requires structure information, at least one of the sequences does not have it, and there is no avialable WT structure.")
            elif any(seq.structure is None for seq in X):
                if X.fixed_length and len(self.wt) == X.width:
                    pass
                else:
                    raise ValueError("This model requires structure information, at least one of the sequences does not have it, and the WT structure size does not match the sequence lengths.")
                
        if self.requires_msa_per_sequence:
            if any(not seq.has_msa for seq in X):
                if self.wt is not None and self.wt.has_msa:
                    warnings.warn("Some sequences do not have an MSA, but the wild type sequence does. Attempting to use the wild type sequence MSA.")
                    wt_msa = self.wt.msa
                    for seq in X:
                        if len(seq) == wt_msa.width:
                            seq.msa = wt_msa
                        else:
                            raise ValueError("Some sequences do not have an MSA, and the wild type sequence does not have one either. Cannot fit model.")
                else:
                    raise ValueError("Some sequences do not have an MSA, and the wild type sequence does not have one either. Cannot fit model.")
                
            if any(not seq.msa_same_width for seq in X):
                raise ValueError("Not all sequence MSAs have the same width as the sequence itself.")
            

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
        Transform sequences with pre-processing and post-processing from mixins.

        Args:
            X (Union[ProteinSequences, List[str]]): Input sequences.

        Returns:
            np.ndarray: Transformed sequences.
        """
        logger.info(f"Transforming input using {self.__class__.__name__}")
        
        # Ensure model is fitted
        check_is_fitted(self)

        # Store original input for post hooks
        original_X = X
        
        # Validate input
        X = self._validate_input(X)

        # Run pre-transform hooks from mixins
        for hook in self._pre_transform_hooks:
            X = hook(self, X)

         # If X is None after hooks, it indicates all results are cached
        # Skip to post-processing hooks with None result
        if X is None or len(X) == 0:
            outputs = None
        else:
            # Standard validations
            if self.requires_fixed_length:
                self._assert_fixed_length(X)

            if self.requires_structure:
                if any(seq.structure is None for seq in X) and self.wt is None:
                    raise ValueError("This model requires structure information, at least one of the sequences does not have it, and there is no available WT structure.")
                elif any(seq.structure is None for seq in X):
                    if X.fixed_length and len(self.wt) == X.width:
                        pass
                    else:
                        raise ValueError("This model requires structure information, at least one of the sequences does not have it, and the WT structure size does not match the sequence lengths.")

            if self.requires_msa_per_sequence:
                if any(not seq.has_msa for seq in X):
                    if self.wt is not None and self.wt.has_msa:
                        warnings.warn("Some sequences do not have an MSA, but the wild type sequence does. Attempting to use the wild type sequence MSA.")
                        wt_msa = self.wt.msa
                        for seq in X:
                            if seq.msa is None:
                                seq.msa = wt_msa
                    else:
                        raise ValueError("Some sequences do not have an MSA, and the wild type sequence does not have one either. Cannot fit model.")
                    
                if any(not seq.msa_same_width for seq in X) and not self.can_handle_aligned_sequences:
                    raise ValueError("Not all sequence MSAs have the same width as the sequence itself.")
                elif any(not seq.msa_same_width for seq in X):
                    # align the sequence itself to the MSA
                    new_X = []
                    for seq in X:
                        if not seq.msa_same_width:
                            seq = seq.align(seq.msa)
                            assert seq.msa_same_width
                        new_X.append(seq)
                    X = ProteinSequences(new_X)

            if not self.can_handle_aligned_sequences and X.has_gaps:
                logger.info("Input sequences have gaps and the model cannot handle them. Removing gaps.")
                X = X.with_no_gaps()
            
            # Call the actual transform implementation
            outputs = self._transform(X)
            
            # Handle wildtype normalization if needed
            if not self.requires_wt_during_inference and self.wt is not None:
                wt_output = self._transform(ProteinSequences([self.wt]))
                # make the outputs arrays so that we can subtract
                if not isinstance(outputs, np.ndarray):
                    try:
                        outputs = np.array(outputs)
                    except:
                        raise ValueError("The model outputs could not be converted to a numpy array, likely because of difference in protein length and no pooling.")
                    
                if not isinstance(wt_output, np.ndarray):
                    try:
                        wt_output = np.array(wt_output)
                        if wt_output.ndim == 1:
                            wt_output = wt_output.reshape(-1, 1)
                    except:
                        raise ValueError("The model outputs could not be converted to a numpy array, likely because of difference in protein length and no pooling.")
                    
                # take the difference
                if wt_output.shape[1:] != outputs.shape[1:]:
                    raise ValueError("The model outputs and WT outputs do not have the same shape.")
                
                outputs -= wt_output
        
        # Run post-transform hooks from mixins
        for hook in self._post_transform_hooks:
            outputs = hook(self, outputs, original_X)
        
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
    def expects_no_fit(self) -> bool:
        """Whether the model expects no fit."""
        return self._expects_no_fit

    @property
    def requires_msa_for_fit(self) -> bool:
        """Whether the model requires an MSA for fitting."""
        return self._requires_msa_for_fit
    
    @property
    def requires_wt_msa(self) -> bool:
        """Whether the model requires a wild type MSA for fitting."""
        return self._requires_wt_msa
    
    @property
    def requires_msa_per_sequence(self) -> bool:
        """Whether the model requires an MSA for each sequence during transform."""
        return self._requires_msa_per_sequence
    
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
        """Whether the model requires wild type during inference."""
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
    
    @property
    def should_refit_on_sequences(self) -> bool:
        """Whether the model should refit on new sequences when given."""
        return self._should_refit_on_sequences

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
            return os.path.join(prefered_tempdir, self.__class__.__name__+"_"+str(time.strftime("%Y%m%d_%H%M%S")))
        return os.path.abspath(metadata_folder)
    
    def _ensure_metadata_folder(self):
        if self.metadata_folder and not os.path.exists(self.metadata_folder):
            os.makedirs(self.metadata_folder)
            logger.info(f"Created metadata folder: {self.metadata_folder}")


#############################################
# MIXINS FOR MODELS
#############################################

class ExpectsNoFitMixin:
    """Either because the model is pretrained, or it will be trained based on WT sequence information only, the model expects the fit method to recieve None."""

    _expects_no_fit: bool = True

class RequiresMSAForFitMixin:
    """
    Mixin to ensure model receives aligned sequences at fit.

    This mixin overrides the requires_msa_for_fit attribute to be True.
    """
    _requires_msa_for_fit: bool = True

class RequiresWTMSAMixin:
    """
    Mixin to ensure model's WT has an alignment available.
    """
    _requires_wt_msa: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # check that requires wt is also true, otherwise this mixin makes no sense
        if not cls.requires_wt_to_function:
            raise ValueError("RequiresWTMSAMixin  requires that the model also requires the wild type sequence to function. Please add RequiresWTToFunctionMixin to the class.")

class RequiresMSAPerSequenceMixin:
    """
    Mixin to ensure model receives sequences at transform that each have an MSA.
    If that fails it will attempt to find one via the wild type sequence.
    """
    _requires_msa_per_sequence: bool = True

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
    Mixin to indicate whether the model uses a wild type directly during inference, otherwise outputs are noramlized to any wt sequence.
    
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
    
    This mixin adds functionality for handling position-specific outputs from protein models.
    It allows selecting specific positions to analyze, pooling across positions, and 
    flattening multi-dimensional outputs.
    
    Attributes:
        positions (Optional[List[int]]): The positions to output scores for. If None, all positions are used.
        pool (Union[bool, str, Callable]): Whether/how to pool scores across positions:
            - If True: Uses mean pooling
            - If string: Uses named numpy function (e.g. 'mean', 'max')
            - If callable: Uses the provided function for pooling
            - If False: No pooling is performed
        flatten (bool): Whether to flatten dimensions beyond the second dimension.
    """
    _per_position_capable: bool = True

    def _init_handler(self, positions=None, pool=True, flatten=True, **kwargs):
        """Initialize position-specific attributes from kwargs."""
        self.positions = positions
        self.pool = pool
        self.flatten = flatten
        
    def _is_ragged_array(self, arr):
        """Check if the input is a ragged array (list of arrays with different shapes)."""
        if not isinstance(arr, list):
            return False
        if len(arr) == 0:
            return False
        first_shape = arr[0].shape
        return any(a.shape != first_shape for a in arr[1:])
    
    def _pre_transform_hook(self, X):
        if X is None:
            return X
        # check that if positions are passed, all inputs are the same length
        # if positions are passed
        if self.positions is not None:
            if not (X.aligned or len(X) == 1):
                raise ValueError("Input sequences must be same length / aligned for position-specific output.")
            
        return X

    def _post_transform_hook(self, result, X):
        """
        Process the model output to handle position selection, pooling, and flattening.
        
        Args:
            result: The raw output from _transform
            X: The input sequences
            
        Returns:
            np.ndarray: Processed output with correct dimensions
        """
        if result is None or len(result) == 0:
            return result
        if self.pool:
            # get the pool function
            if isinstance(self.pool, str):
                pool_func = getattr(np, self.pool)
            elif callable(self.pool):
                pool_func = self.pool
            else:
                pool_func = np.mean

        # Handle ragged arrays (different shapes)
        if self._is_ragged_array(result):
            if not self.pool:
                raise ValueError("Pooling must be enabled to handle ragged arrays.")
            else:
                # apply pooling to each array in the list
                result = [pool_func(a, axis=-2) for a in result]
                # convert to a single array
                if self._is_ragged_array(result):
                    raise ValueError("Pooling failed to convert ragged arrays to a single array.")
            
        # Convert list of arrays to a single array if possible
        if isinstance(result, list):
            if all(a.shape[0] == 1 for a in result):
                result = np.vstack(result)
            else:
                try:
                    result = np.array(result)
                except ValueError:
                    # If we can't convert to a single array, return as is
                    return result
        
        # Validate dimensions for position-specific output
        if self.positions is not None and not self.pool:
            dims = len(self.positions)
            if result.ndim > 2:
                if result.shape[1] != dims:
                    # make sure that the number of values is equal to the length of the sequences before trying to do indexing ourselves
                    if result.shape[1] == X.width:
                        result = result[:, self.positions, :]
                    else:
                        raise ValueError(f"Model output has {result.shape[1]} positions, which does not match the sequence size {X.width}.")
                else:
                    # this is the expected shap if the model properly handled indexing positions itself
                    pass
        
        # Apply pooling if specified (and not already done by internal model)
        if self.pool and result.ndim > 2:
            # Pooling across the position dimension (dim=1)
            result = pool_func(result, axis=1)
        
        # Flatten output if requested
        if self.flatten and result.ndim > 2:
            result = result.reshape(result.shape[0], -1)
        
        return result
        
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation, considering position-specific output and flattening.
        
        This method overrides the base implementation to provide more descriptive feature names
        that include position information when relevant.

        Args:
            input_features (Optional[List[str]]): Input feature names (unused).

        Returns:
            List[str]: Output feature names.
        """
        check_is_fitted(self)
        header = self.__class__.__name__
        
        # If we don't have position-specific output, return simple name
        if self.positions is None or self.pool:
            if hasattr(self, 'embedding_dim_') and self.pool:
                return [f"{header}_emb{i}" for i in range(self.embedding_dim_)]
            return [header]
            
        # Handle position-specific naming
        base_names = [f"{header}_pos{pos}" for pos in self.positions]
        
        if self.flatten and hasattr(self, 'embedding_dim_'):
            # For flattened embeddings with known dimension
            return [f"{name}_emb{i}" for name in base_names for i in range(self.embedding_dim_)]
        elif self.flatten:
            # For flattened output with unknown dimension
            return [f"{name}_dim{i}" for name in base_names for i in range(getattr(self, 'output_dim_', 1))]
        
        # For non-flattened position-specific output
        return base_names
    
class CanHandleAlignedSequencesMixin:
    """
    Mixin to indicate that a model can handle aligned sequences (with gaps) during prediction.
    
    This mixin overrides the can_handle_aligned_sequences attribute to be True.
    """
    _can_handle_aligned_sequences: bool = True
    

class CacheMixin:
    """
    Mixin to provide per-protein caching functionality for ProteinModelWrapper subclasses.
    
    This mixin adds efficient caching of model outputs to avoid redundant computations. 
    It uses SQLite for metadata indexing and HDF5 for efficient embedding storage,
    optimized for batch operations and improved file handling.
    
    Attributes:
        use_cache (bool): Whether to enable caching. Default is True.
    """
    
    def _init_handler(self, use_cache=True, **kwargs):
        """Initialize cache-related attributes and setup cache infrastructure."""
        self.use_cache = use_cache
        if hasattr(self, 'metadata_folder'):
            self._cache_dir = os.path.join(self.metadata_folder, 'cache')
            self._db_file = os.path.join(self._cache_dir, 'cache.db')
            self._hdf5_file = os.path.join(self._cache_dir, 'embeddings.h5')
            self._ensure_cache_dir()
            self._init_db()
            self._hdf5 = None
    
    def _ensure_cache_dir(self):
        """Ensure the cache directory exists."""
        os.makedirs(self._cache_dir, exist_ok=True)

    def _init_db(self):
        """Initialize the SQLite database for caching metadata."""
        with sqlite3.connect(self._db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    protein_hash TEXT PRIMARY KEY,
                    model_state TEXT,
                    timestamp REAL,
                    protein_length INTEGER,
                    output_shape TEXT
                )
            ''')
            conn.commit()

    def _get_model_state_hash(self) -> str:
        """Generate a hash representing the current state of the model."""
        model_params = self.get_params()
        for k, v in model_params.items():
            if not isinstance(v, (str, int, float, bool, type(None))):
                model_params[k] = str(v)
        model_params = json.dumps(model_params, sort_keys=True)
        fitted_attrs = json.dumps({attr: str(getattr(self, attr)) for attr in self.get_fitted_attributes()}, sort_keys=True)
        return hashlib.md5((model_params + fitted_attrs).encode()).hexdigest()

    def _get_protein_hashes(self, proteins: ProteinSequences) -> List[str]:
        """Generate hashes for a batch of protein sequences."""
        return [hashlib.sha256((str(p) + f"|id:{p.id}" + (f"|structure:{str(p.structure.pdb_file)}{p.structure.chain}{p.structure.plddt_file}" if p.structure else "")).encode()).hexdigest() for p in proteins]

    def _safely_close_hdf5(self):
        """Safely close the HDF5 file if it's open."""
        if hasattr(self, '_hdf5') and self._hdf5 is not None:
            self._hdf5.close()
            self._hdf5 = None

    def _batch_is_cached(self, protein_hashes: List[str], model_state: str) -> Dict[str, bool]:
        """Check if multiple proteins are cached and valid."""
        with sqlite3.connect(self._db_file) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(protein_hashes))
            query = f"SELECT protein_hash, model_state FROM cache WHERE protein_hash IN ({placeholders})"
            cursor.execute(query, protein_hashes)
            results = cursor.fetchall()
        
        cache_status = {ph: False for ph in protein_hashes}
        for ph, ms in results:
            cache_status[ph] = (ms == model_state)
        return cache_status

    def _batch_get_cached_results(self, protein_hashes: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve cached results for multiple proteins from HDF5 file."""
        results = {}
        if not os.path.exists(self._hdf5_file):
            return results

        try:
            with h5py.File(self._hdf5_file, 'r', libver='latest', swmr=True) as f:
                for ph in protein_hashes:
                    if ph in f:
                        results[ph] = f[ph][:]
        except OSError as e:
            logger.error(f"Error reading HDF5 file: {e}")
            self._safely_close_hdf5()
        return results

    def _batch_cache_results(self, results: Dict[str, np.ndarray], model_state: str, protein_lengths: Dict[str, int]):
        """Cache results for multiple proteins in HDF5 file and update SQLite metadata."""
        self._safely_close_hdf5()  # Ensure the file is closed before opening in write mode
        try:
            with h5py.File(self._hdf5_file, 'a') as f:
                for protein_hash, result in results.items():
                    if protein_hash in f:
                        del f[protein_hash]
                    f.create_dataset(protein_hash, data=result, compression="gzip", compression_opts=9)
        except OSError as e:
            raise ValueError(f"Error writing to HDF5 file: {e}")

        with sqlite3.connect(self._db_file) as conn:
            cursor = conn.cursor()
            data = [
                (ph, model_state, time.time(), protein_lengths[ph], str(result.shape))
                for ph, result in results.items()
            ]
            cursor.executemany('''
                INSERT OR REPLACE INTO cache
                (protein_hash, model_state, timestamp, protein_length, output_shape)
                VALUES (?, ?, ?, ?, ?)
            ''', data)
            conn.commit()

    def get_fitted_attributes(self) -> List[str]:
        """Get a list of attributes that are set during fitting."""
        return [attr for attr in dir(self) if attr.endswith('_') and not attr.startswith('_')]
    
    def _pre_transform_hook(self, X):
        """
        Pre-transform hook that checks the cache before proceeding with transformation.
        If all sequences are cached, this hook short-circuits the transform process.
        
        Args:
            model: The model instance
            X: The input sequences
            
        Returns:
            X or None: The input sequences if transformation is needed, None if all cached
        """
        if not self.use_cache:
            return X
            
        try:
            check_is_fitted(self)
        except:
            return X

        # Store data for potential use in post-transform
        self._cache_data = {
            'X': X,
            'model_state': self._get_model_state_hash(),
            'protein_hashes': self._get_protein_hashes(X)
        }
        
        # Check which sequences are cached
        cache_status = self._batch_is_cached(
            self._cache_data['protein_hashes'], 
            self._cache_data['model_state']
        )
        
        self._cache_data['cached_hashes'] = [ph for ph, status in cache_status.items() if status]
        self._cache_data['uncached_hashes'] = [ph for ph, status in cache_status.items() if not status]
        self._cache_data['cached_results'] = self._batch_get_cached_results(self._cache_data['cached_hashes'])
        
        logger.info(f"Found {len(self._cache_data['cached_results'])} cached results, "
                   f"running on {len(self._cache_data['uncached_hashes'])} uncached sequences.")
        
        # If everything is cached, return None to short-circuit transform
        if len(self._cache_data['uncached_hashes']) == 0:
            self._cache_data['all_cached'] = True
            # Transform will be skipped, post_transform will handle returning cached results
            return None
        else:
            # Only transform the uncached sequences
            self._cache_data['all_cached'] = False
            proteins_to_transform = ProteinSequences([
                X[i] for i, ph in enumerate(self._cache_data['protein_hashes']) 
                if ph in self._cache_data['uncached_hashes']
            ])
            return proteins_to_transform

    def _post_transform_hook(self, result, X):
        """
        Post-transform hook that handles caching of new results and combining with cached results.
        
        Args:
            model: The model instance
            result: The raw output from _transform (or None if all cached)
            X: The input sequences that were transformed (or None if all cached)
            
        Returns:
            np.ndarray: Combined results from cache and new transformations
        """
        if not self.use_cache or not hasattr(self, '_cache_data'):
            return result
            
        # If everything was cached, just return the cached results
        if self._cache_data['all_cached']:
            # Reorder results to match original input order
            all_results = [self._cache_data['cached_results'][ph] for ph in self._cache_data['protein_hashes']]
            
            self._safely_close_hdf5()
            
            if len(set([o.shape for o in all_results])) == 1:
                return np.vstack(all_results)
            else:
                return all_results
                
        # Otherwise, combine new results with cached ones
        new_results_dict = {}
        for i, (ph, result_item) in enumerate(zip(self._cache_data['uncached_hashes'], result)):
            # Ensure consistent shape for storage
            result_array = np.asarray(result_item)
            if result_array.shape[0] != 1:
                result_array = result_array[np.newaxis, :]
            new_results_dict[ph] = result_array
            
        # Cache the new results
        protein_lengths = {
            ph: len(self._cache_data['X'][i]) 
            for i, ph in enumerate(self._cache_data['protein_hashes']) 
            if ph in self._cache_data['uncached_hashes']
        }
        self._batch_cache_results(
            new_results_dict, 
            self._cache_data['model_state'], 
            protein_lengths
        )
        
        # Combine cached and new results
        combined_results = dict(self._cache_data['cached_results'])
        combined_results.update(new_results_dict)
        
        # Reorder results to match original input order
        all_results = [combined_results[ph] for ph in self._cache_data['protein_hashes']]
        
        self._safely_close_hdf5()
        
        # Clean up
        del self._cache_data
        
        # Return appropriately formatted results
        if len(set([o.shape for o in all_results])) == 1:
            return np.vstack(all_results)
        else:
            return all_results

class AcceptsLowerCaseMixin:
    """
    Mixin to indicate that a model can accept lower case sequences.

    This mixin overrides the accepts_lower_case attribute to be True.
    """
    _accepts_lower_case: bool = True

class ShouldRefitOnSequencesMixin:
    """
    Mixin to indicate that a model should refit on new sequences when given.

    This mixin overrides the should_refit_on_sequences attribute to be True.
    """
    _should_refit_on_sequences: bool = True

