# aide_predict/bespoke_models/base.py
'''
* Author: Evan Komp
* Created: 5/7/2024
* (c) Copyright by Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology

Base classes for models to be wrapped into the API
'''
from dataclasses import dataclass
import os
from abc import abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin

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

class ModelWrapper(TransformerMixin, BaseEstimator):
    wrapper_args_class = None

    def __init__(self, args: ModelWrapperArgs):
    
        # check that the args class is the correct type
        if self.wrapper_args_class is None or not isinstance(args, self.wrapper_args_class):
            raise ValueError(f"args must be of type {self.wrapper_args_class}")
        
        # assign all of the parameters sklearn style
        for key, value in args.__dict__.items():
            setattr(self, key, value)

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError("This method must be implemented in the child class.")
    
    @abstractmethod
    def transform(self, X):
        raise NotImplementedError("This method must be implemented in the child class.")
    
class ModelWrapperPreprocessor:
    """Abstract class that prepares metadata for a model based on """

    # TODO: Should the preprocessor occur in a seperate dvc stage, or should it be part of the model?
    # For example, if it is its own stage, we will more easily be able to atomize preprocessing steps,
    # but need to think about how eg. an MSA doesn't need to be repeated if jumping from Tranception to EVE, since they both require it.
    

