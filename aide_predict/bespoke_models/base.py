# aide_predict/bespoke_models/base.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Base classes for models to be wrapped into the API as sklearn estimators
'''
from dataclasses import dataclass
import os
from abc import abstractmethod
import inspect

from sklearn.base import BaseEstimator

@dataclass
class ModelWrapperArgs:
    """This class is meant to be inherited and expanded to represent the arguments for a given model.
    
    Desired behavior: 
    - A metadata folder must be passed, and the contents of that folder will be parsed
    to ensure that the necessary metadata is present.

    For example - many covartiation models expect MSAs, or language models expect weights.
    Here we define a protocol to point toward that metadata and check the format based on the bespoke purpose.
    """
    metadata_folder: str

    def __post_init__(self):
        if not os.path.exists(self.metadata_folder):
            raise ValueError(f"metadata_folder {self.metadata_folder} does not exist.")
        self.check_metadata()

    @abstractmethod
    def check_metadata(self):
        raise NotImplementedError("This method must be implemented in the child class.")
    
    @property
    def kwargs(self):
        """User defined parameters."""
        # only return no hidden attributes
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}
    
    def __call__(self):
        return self.kwargs

class ModelWrapper(BaseEstimator):
    _wrapper_args_class = None
    _requires_msa = None
    _per_position_capable = None
    _requires_fixed_length = None

    def __init__(self, **kwargs):
        # Make sure all class variables are set
        if self.wrapper_args_class is None:
            raise ValueError("wrapper_args_class must be set for your subclass")
        if self.requires_msa is None:
            raise ValueError("requires_msa must be set for your subclass")
        if self.per_position_capable is None:
            raise ValueError("per_position_capable must be set for your subclass")
        if self.requires_fixed_length is None:
            raise ValueError("requires_fixed_length must be set for your subclass")

        # Create an instance of the wrapper_args_class with the kwargs
        args = self.wrapper_args_class(**kwargs)

        # assign all of the parameters sklearn style
        for key, value in args.__dict__.items():
            setattr(self, key, value)

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError("This method must be implemented in the child class.")
    
    @property
    def wrapper_args_class(self):
        """The class that holds the arguments for the model.
        """
        return self._wrapper_args_class
    
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
        if 'positions' not in inspect.getargspec(self.transform).args and self._per_position_capable:
            raise ValueError("If per_position_capable is True, class method transform must accept a 'positions' argument.")

        return self._per_position_capable
    
    @property
    def requires_fixed_length(self):
        return self._requires_fixed_length
    
    

