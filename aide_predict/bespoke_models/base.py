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

from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from aide_predict.utils.common import process_amino_acid_sequences, fixed_length_sequences

#############################################
# BASE CLASSES FOR DOWNSTREAM MODELS
#############################################
class ModelWrapper(BaseEstimator):
    """Base class for bespoke models.

    Ensure that this init is supered in the child class.

    Construction of a ModelWrapper from __init__ expects
    that the metadata is already present in the metadata folder, and
    we will do the appropriate checks. If you want to be able to construct
    the metadata necessary for the class algorithmically, you should implement
    the _construct_necessary_metadata method and use the from_basic_info classmethod
    to construct the class.
    
    Class attributes, please overload as necessary:
    - `requires_msa`: Whether the model requires an MSA as input. If so, please inherit from ModelWrapperRequiresMSA instead of this class,
      and it will automatically check for the presence of an MSA.
      This attribute also helps ensure that the user is allowed to turn the MSA step of the pipeline off.
       
    - `requires_wt_during_inference`: Whether the model requires the wild type sequence during inference
      if the intention is to get a score relative to wild type.
      If this is False, the model will automatically normalize the output by the wild type sequence if present
      and the user does not have to worry about it.
      If this is True, the model is expected to handle the normalization in the _predict and _transform methods.

    - `per_position_capable`: Whether the model can output per position scores.
      If so, the model should have a positions attribute.
      If your model has position specific capabilities, you should inherit from PositionSpecificMixin and
      all you have to do is ensure that your _predict and `_transform` methods are implemented to output the correct shape.
      as well as have a `positions` attribute.

    - `requires_fixed_length`: Whether the model requires a fixed length input.
      Besides being informative, this is used to check whether the user is allowed to
      pass variable length sequences to the model.

    Attributes:
    - `metadata_folder`: The folder where the metadata is stored.
    - `wt`: The wild type sequence.

    To create a subclass, please implement the following methods:
    - If your model has any parameters, implement `__init__` and call `super().__init__` with the
        metadata_folder and wild type sequence.
    - `fit`
    - `_predict`
    - `_transform` if the model can be within a pipeline as well as final predictor
    NOTE: The desired behavior of transform and predict is to normalize scores by wild type if present.
          For models where `requires_wt_during_inference` is False, the exposed methods will automatically
          call the hidden methods on the wt sequence and normalize, However, `requires_wt_during_inference` is true
          your implementations of `_predict` and `_transform` are expected to handle the normalization.
    - `check_metadata`: This method should check that all necessary metadata is present in the metadata folder
      if the model requires anything
    - `_construct_necessary_metadata`: This method scan be implemented to construct the necessary metadata for the model
      from arguments, instead of making the user manually set up the metadata folder.
    """
    _requires_msa = False
    _requires_wt_during_inference = False
    _per_position_capable = False # be conservative for the default here
    _requires_fixed_length = True  # and here

    def __init__(self, metadata_folder: str=None, wt: str=None):
        # Make sure all class variables are set
        if self.requires_msa is None:
            raise ValueError("requires_msa must be set for your subclass")
        if self.per_position_capable is None:
            raise ValueError("per_position_capable must be set for your subclass")
        if self.requires_fixed_length is None:
            raise ValueError("requires_fixed_length must be set for your subclass")
        
        if metadata_folder is None:
            raise ValueError("metadata_folder must be provided.")
        if not os.path.exists(metadata_folder):
            os.makedirs(metadata_folder)
        self.metadata_folder = metadata_folder
        self._wt=wt

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
            wt = next(process_amino_acid_sequences([wt]))
        self._wt = wt

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError("This method must be implemented in the child class.")
    
    @abstractmethod
    def _predict(self, X):
        raise NotImplementedError("This method must be implemented in the child class.")
    
    @abstractmethod
    def _transform(self, X):
        raise NotImplementedError("This method must be implemented in the child class.")
    
    def predict(self, X):
        """Predict the scores for the sequences.
        """
        X = list(process_amino_acid_sequences(X))
        if self.requires_fixed_length:
            if not fixed_length_sequences(X):
                raise ValueError("All sequences must have the same length for this model.")
            if self.wt is not None:
                assert len(self.wt) == len(X[0]), "Wild type sequence must have the same length as sequences for this model."
        if not self.requires_wt_during_inference and self.wt is not None:
            outs = self._predict(X)
            check_array(outs, ensure_2d=True)
            wt_outs = self._predict([self.wt])
            check_array(wt_outs, ensure_2d=True)
            return outs - wt_outs
        else:
            outs = self._predict(X)
            check_array(outs, ensure_2d=True)
            return outs
        
    def transform(self, X):
        """Transform the sequences.
        """
        X = list(process_amino_acid_sequences(X))
        if self.requires_fixed_length:
            assert all(len(x) == len(X[0]) for x in X), "All sequences must have the same length."
            if self.wt is not None:
                assert len(self.wt) == len(X[0]), "Wild type sequence must have the same length as sequences."
        if not self.requires_wt_during_inference and self.wt is not None:
            outs = self._transform(X)
            check_array(outs, ensure_2d=True)
            wt_outs = self._transform([self.wt])
            check_array(wt_outs, ensure_2d=True)
            return outs - wt_outs
        else:
            outs = self._transform(X)
            check_array(outs, ensure_2d=True)
            return outs
    
    @property
    def requires_msa(self):
        """Whether the model requires an MSA as input.
        
        Note that this property is used for pipeline checks.
        """
        return self._requires_msa
    
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

    @classmethod
    def help(cls):
        """Helper function to print out useful information about the model class and
        how to use it.
        """
        print(f"Help for {cls.__name__}")
        print(f"Requires MSA: {cls.requires_msa}")
        print(f"Per position capable: {cls.per_position_capable}")
        print(f"Requires fixed length: {cls.requires_fixed_length}")
        print(f"Input arguments: {inspect.signature(cls.__init__)}")
        if hasattr(cls, 'fit'):
            print(f"Fit arguments: {inspect.signature(cls.fit)}")
        if hasattr(cls, 'predict'):
            print(f"Predict arguments: {inspect.signature(cls.predict)}")
        if hasattr(cls, 'transform'):
            print(f"Transform arguments: {inspect.signature(cls.transform)}")

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


# This is simply give the user the option to not have to write metadata checks
# If all it requires is an MSA
class ModelWrapperRequiresMSA(ModelWrapper):
    """Base class for bespoke models that require an MSA.

    This class simply checks the metadata folder for the presence of an MSA
    such that the downstream classes do not have to
    """
    _requires_msa = True
    def check_metadata(self):
        if not os.path.exists(os.path.join(self.metadata_folder, 'alignment.a2m')):
            raise ValueError(f"alignment.a2m does not exist in {self.metadata_folder}")
    
    @staticmethod
    def _construct_necessary_metadata(model_directory: str, necessary_metadata: dict):
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        # move the alignment file to the correct name.
        if not 'alignment_location' in necessary_metadata:
            raise ValueError("alignment_location must be provided in necessary_metadata.")
        alignment_location = necessary_metadata['alignment_location']
        if not os.path.exists(alignment_location):
            raise ValueError(f"Alignment file does not exist at {alignment_location}")
        # copy the file
        shutil.copy(alignment_location, os.path.join(model_directory, 'alignment.a2m'))


#############################################
# MIXINS FOR MODELS
#############################################

class PositionSpecificMixin:
    """Mixin for models that can output per position scores.
    
    This mixin:
    1. Overrides the per_position_capable attribute to be True.
    2. checks that positions is an attribute
    3. Wraps the predict and transform methods to check that if positions were passed, the output is the same length as the positions.
    """
    _per_position_capable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, 'positions'):
            raise ValueError("This model was specified as PositionSpecific, but does not have a positions attribute. Make sure `positions` is a parameter in the __init__ method.")

        # Wrap the predict method
        self.predict = self._wrap_predict(self.predict)
        self.transform = self._wrap_predict(self.transform)

    def _wrap_predict(self, func):
        @wraps(func)
        def wrapper(X):
            # if positions is not None, check that the output second dimension has the same length as number of positions
            result = func(X)
            if self.positions is not None:
                dims = len(self.positions)
                if result.shape[1] != dims:
                    raise ValueError("The output second dimension must have the same length as number of positions.")
            return result
        return wrapper

    
    

