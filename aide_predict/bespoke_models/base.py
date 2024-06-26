# aide_predict/bespoke_models/base.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Base classes for models to be wrapped into the API as sklearn estimators
'''
from functools import wraps
import os
import shutil
from abc import abstractmethod
import inspect
import warnings

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence

#############################################
# BASE CLASSES FOR DOWNSTREAM MODELS
#############################################
class ProteinModelWrapper(TransformerMixin, BaseEstimator):
    """Writing new docstrings.
    
    Base class for bespoke models that take proteins as input.
    
    All models that take proteins as input should inherit from this class.
    All are considered transformers such that they can be used natively to
    produce features in the AIDE pipeline.
    Models can additionally be made regressors by inheriting from RegressorMixin.

    X values for fit, transform, predict are expected to be ProteinSequences objects.

    Params
    ------
    metadata_folder : str
        The folder where the metadata is stored. All models should have a metadata folder.
        Any that require additional metadata other than input sequences and parameters to train
        should implement the check_metadata method to ensure that the metadata is present.
        Additionally, models that create artifacts outside of the python should
        save them in the metadata folder.
    wt : str
        The wild type sequence if present. If so, outputs will be normalized by the wild type sequence.

        
    Attributes
    ----------
    metadata_folder : str
        The folder where the metadata is stored.
    wt : str
        The wild type sequence.
    
    Class Attributes
    ----------------
    requires_msa_for_fit : bool
        Whether the model requires an MSA as input for fitting. If so, please inherit from
        RequiresMSAMixin. If true, the model will check for alignment of input sequences
        at fit time. If not aligned, it will attempt to align them.
    requires_wt_during_inference : bool
        Whether the model requires the wild type sequence during inference if the intention is to get a score
        relative to wild type.
        If this is False, the model will automatically normalize the output by the wild type sequence if present
        and the user does not have to worry about it.
        If this is True, the child class is expected to implement its own normalization in _transform
        If so, please inherit from RequiresWTDuringInferenceMixin.
    per_position_capable : bool
        Whether the model can output per position scores.
        If so, the model should have a positions and a pool attribute.
        If your model has position specific capabilities, you should inherit from PositionSpecificMixin and
        all you have to do is ensure that your `_transform` method monitors self.postitions and self.pool
        to output the correct shape.
        If so, please inherit from PositionSpecificMixin.
    requires_fixed_length : bool
        Whether the model requires a fixed length input.
        If so, the model will check that the input sequences are fixed length.
        Please inherit from RequiresFixedLengthMixin if this is the case.
        If so, please inherit from RequiresFixedLengthMixin.
    can_regress : bool
        Whether the model outputs from transform can also be considered estimates of activity label.
        Eg. HMM scores can be correlated to activity (predictions) in addition to being input as features to a supervised
        If so, please inherit from CanRegressMixin.
    
    Methods
    -------
    fit(X, y=None)
        Fit the model.
    transform(X)
        Transform the sequences.
    fit_transform(X, y=None)
        Fit the model and transform the sequences.
    predict(X)
        Predict the sequences.
    check_metadata()
        Ensure that all necessary metadata is present in the metadata folder.
    _fit(X, y=None)
        YOU MUST IMPLEMENT.
    _transform(X)
        YOU MUST IMPLEMENT.
    _construct_necessary_metadata(model_directory, necessary_metadata)
        YOU CAN IMPLEMENT (if your model requires additional metadata).
    from_basic_info(model_directory, necessary_metadata, wt, **kwargs)
        Construct the model from basic information and instantiate.

    To subclass, please implement the following methods:
    - If your model has any parameters, implement `__init__` and call `super().__init__` with the
        `metadata_folder` and `wt`.
    - `_fit` : 
        his method should accept `X` (of class `ProteinSequences`) and `y` (of class `np.ndarray)
        and fit the model. Assign at least one trailing _ attribute to self so that the model is considered fitted.
    - `_transform` :
        This method should accept `X` (of class `ProteinSequences`) and return the transformed sequences as an array.
    - `check_metadata` :
        This method should check that all necessary metadata is present in the metadata folder
        if the model requires anything. If your model requires something, overload this.
    - `_construct_necessary_metadata` :
        This method can be implemented to construct the necessary metadata for the model
        from arguments, instead of making the user manually set up the metadata folder.
    
    Mixins to constrain model behavior:
    - RequiresMSAMixin :
        If the model requires an MSA for fitting, inherit from this mixin. The X that your _fit method
        receives will be aligned.
    - RequiresFixedLengthMixin :
        If the model requires fixed length sequences, inherit from this mixin. The model will check that
        the input sequences are fixed length.
    - CanRegressMixin :
        If the model outputs are also capable of being correlated to protein activity, inherit from this mixin.
        Predict will be available as a method.
    - RequiresWTDuringInferenceMixin :
        If the model, in order to get a score relative to wild type, requires the wild type sequence during inference,
        inherit from this mixin. Your _transform method will be expected to handle the normalization to wt scores.
    - PositionSpecificMixin :
        If the model can output per position scores, inherit from this mixin. You should ensure that your
        __init__ assigns a `positions` attribute and a `pool` attribute. In your _transform method, `positions` can
        be accessed to subset positions of the overal sequence to return. If `pool` is False, the output should be the same
        size as the number of positions, else you should implement some pooling like mean pooling over the specified positions.
    
    """
    _requires_msa_for_fit = False
    _requires_wt_during_inference = False
    _per_position_capable = False # be conservative for the default here
    _requires_fixed_length = False  # and here
    _can_regress = False

    def __init__(self, metadata_folder: str=None, wt: ProteinSequence=None):
        
        if metadata_folder is None:
            raise ValueError("metadata_folder must be provided.")
        if not os.path.exists(metadata_folder):
            os.makedirs(metadata_folder)
        self.metadata_folder = metadata_folder
        if not isinstance(wt, ProteinSequence) and wt is not None:
            wt = ProteinSequence(wt)
            if wt.has_gaps:
                raise ValueError("Wild type sequence cannot have gaps.")
        self.wt=wt

        # call check metadata
        self.check_metadata()

    def check_metadata(self):
        """Ensures that eveything this model class needs is in the metadata folder."""
        # raise a warning that does not stop execution
        warnings.warn("This model class did not implement check_metadata. If the model resuires anything other than raw sequences to be fit, this is unexpected.")

    @property   
    def wt(self):
        return self._wt
    
    @wt.setter
    def wt(self, wt):
        # check that it is a valid sequences
        if wt is not None:
            wt = ProteinSequence(wt)
            if wt.has_gaps:
                raise ValueError("Wild type sequence cannot have gaps.")
        self._wt = wt

    @abstractmethod
    def _fit(self, X, y=None):
        raise NotImplementedError("This method must be implemented in the child class.")
    
    def fit(self, X, y=None):
        """Fit the model.
        
        Params:
        - X: ProteinSequences or Iterable
            If Iterable, assumed to be a list of sequence strings.
        - y: np.ndarray or None
        """
        if not isinstance(X, ProteinSequences):
            X = ProteinSequences(list(X))

        # Check if the model requires an MSA for fitting
        # If so and the inputs are not aligned, align them
        if not X.aligned and self.requires_msa_for_fit:
            X = X.align_all()
            assert X.aligned, "Sequences should be aligned."

        # Check if the model requires fixed length sequences
        # If so, ensure that the sequences are fixed length
        if self.requires_fixed_length:
            assert X.aligned, "Sequences must be fixed length."
            if self.wt is not None:
                assert len(self.wt) == X.width, "Wild type sequence must be the same length as the sequences."
            self.length_ = X.width

        return self._fit(X, y)
    
    @abstractmethod
    def _transform(self, X):
        raise NotImplementedError("This method must be implemented in the child class.")
        
    def transform(self, X):
        """Transform the sequences.

        Params:
        - X: ProteinSequences or Iterable
            If Iterable, assumed to be a list of sequence strings.
        """
        check_is_fitted(self)
        if not isinstance(X, ProteinSequences):
            X = ProteinSequences(list(X))

        # Do fixed length checks if necessary
        # 1. Ensure input sequences are fixed length
        # 2. Ensure incoming sequences are the same size as was determined at fit
        if self.requires_fixed_length:
            if not X.aligned:
                raise ValueError("Sequences must be aligned.")
            if X.width != self.length_:
                raise ValueError("Sequences must be the same length as the training sequences.")
        
        # Behavior is dependent on whether the model requires the wild type sequence during inference
        # If so, it is expected that the model will handle the normalization in the _transform method.
        # else, the model will automatically normalize the output by the wild type sequence if present.
        if not self.requires_wt_during_inference and self.wt is not None:
            outs = self._transform(X)
            check_array(outs, ensure_2d=True)
            wt_outs = self._transform(ProteinSequences([self.wt]))
            check_array(wt_outs, ensure_2d=True)
            return outs - wt_outs
        else:
            outs = self._transform(X)
            check_array(outs, ensure_2d=True)
            return outs
    
    def predict(self, X):
        """Predict the sequences.

        Params:
        - X: ProteinSequences or Iterable
            If Iterable, assumed to be a list of sequence strings.
        """
        if not self.can_regress:
            raise ValueError("This model is not capable of regression.")
        return self.transform(X)

    @property
    def requires_msa_for_fit(self):
        """Whether the model requires an MSA for fitting.
        
        Note that this property is used for pipeline checks.
        """
        return self._requires_msa_for_fit
    
    @property
    def per_position_capable(self):
        """Whether the model can output per position scores.
        
        Note that this property is used for pipeline checks.
        """
        return self._per_position_capable
    
    @property
    def requires_fixed_length(self):
        return self._requires_fixed_length
    
    @property
    def requires_wt_during_inference(self):
        return self._requires_wt_during_inference

    @property
    def can_regress(self):
        return self._can_regress

    @staticmethod
    def _construct_necessary_metadata(model_directory: str, necessary_metadata: dict):
        """Construct the necessary metadata for a model.

        Operated on `necessary_metadata` to construct files in the model directory that are necessary for the model to run.

        This method should always accept the model_directory and necessary_metadata
        `from_basic_info` calls this method when constructing the class instead
        and necessary metadata as opposed to __init__ which expects metadata to be present already.

        """
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        warnings.warn("This model class did not implement _construct_necessary_metadata. If the model requires anything other than raw sequences to be fit, this is unexpected.")

    @classmethod
    def from_basic_info(
        cls,
        model_directory: str,
        necessary_metadata: dict={},
        wt: str=None,
        **kwargs
    ):
        """Construct the required metadata for a model from basic information and instantiate.
        
        Params:
        - model_directory: The directory to store the metadata in.
        - necessary_metadata: A dictionary of necessary metadata to construct.
        - kwargs: Additional arguments to pass to the model class.
        """
        cls._construct_necessary_metadata(model_directory, necessary_metadata)
        return cls(metadata_folder=model_directory, wt=wt, **kwargs)
    


#############################################
# MIXINS FOR MODELS
#############################################

class RequiresMSAMixin:
    """Mixin to ensure model recieves aligned sequences at fit.

    This mixin:
    1. Overrides the requires_msa_for_fit attribute to be True.
    """
    _requires_msa_for_fit = True

        
class RequiresFixedLengthMixin:
    """Mixin to ensure model recieves fixed length sequences at transform.
    
    This mixin:
    1. Overrides the requires_fixed_length attribute to be True.
    """
    _requires_fixed_length = True

class CanRegressMixin(RegressorMixin):
    """Mixin to ensure model can regress.
    
    This mixin:
    1. Overrides the can_regress attribute to be True.
    """
    _can_regress = True

class RequiresWTDuringInferenceMixin:
    """Mixin to ensure model requires wild type during inference.
    
    This mixin:
    1. Overrides the requires_wt_during_inference attribute to be True.
    """
    _requires_wt_during_inference = True

class PositionSpecificMixin:
    """Mixin for protein models that can output per position scores.
    
    This mixin:
    1. Overrides the per_position_capable attribute to be True.
    2. checks that positions and pool is an attribute
    3. Wraps the predict and transform methods to check that if positions were passed and not pooling, the output is the same length as the positions.
    """
    _per_position_capable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, 'positions'):
            raise ValueError("This model was specified as PositionSpecific, but does not have a positions attribute. Make sure `positions` is a parameter in the __init__ method.")
        if not hasattr(self, 'pool'):
            raise ValueError("This model was specified as PositionSpecific, but does not have a pool attribute. Make sure `pool` is a parameter in the __init__ method.")


    def transform(self, X):
        result = ProteinModelWrapper.transform(self, X)
        if self.positions is not None and not self.pool:
            dims = len(self.positions)
            if result.shape[1] != dims:
                raise ValueError("The output second dimension must have the same length as number of positions.")
        return result

    
    

