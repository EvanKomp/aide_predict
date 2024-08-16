# aide_predict/utils/checks.py
'''
* Author: Evan Komp
* Created: 6/13/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Common checks to ensure that different pipeline components are compatable.
'''
import os
import dvc.api

import inspect
from typing import Optional, List, Dict, Type

import aide_predict
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.bespoke_models.base import ProteinModelWrapper

def check_model_compatibility(
    training_sequences: Optional[ProteinSequences] = None,
    testing_sequences: Optional[ProteinSequences] = None,
    training_msa: Optional[ProteinSequences] = None,
    wt: Optional[ProteinSequence] = None
) -> Dict[str, List[str]]:
    """
    Check which models are compatible with the given data.

    Args:
        training_sequences (Optional[ProteinSequences]): Training protein sequences.
        testing_sequences (Optional[ProteinSequences]): Testing protein sequences.
        training_msa (Optional[ProteinSequences]): Training multiple sequence alignment.
        wt (Optional[ProteinSequence]): Wild-type protein sequence.

    Returns:
        Dict[str, List[str]]: A dictionary with two keys: 'compatible' and 'incompatible',
        each containing a list of compatible and incompatible model names respectively.
    """
    def load_models() -> List[Type[ProteinModelWrapper]]:
        models = []
        for name, obj in inspect.getmembers(aide_predict):
            if inspect.isclass(obj) and issubclass(obj, ProteinModelWrapper) and obj != ProteinModelWrapper:
                models.append(obj)
        return models

    def check_structures_available() -> bool:
        if wt and wt.structure is not None:
            return True
        for seq_set in [training_sequences, testing_sequences]:
            if seq_set:
                if any(seq.structure is not None for seq in seq_set):
                    return True
        return False

    def check_compatibility(model: Type[ProteinModelWrapper]) -> bool:
        if not model._available:
            return False
        if model._requires_fixed_length:
            if (training_sequences and not training_sequences.fixed_length) or \
               (testing_sequences and not testing_sequences.fixed_length):
                return False
        if model._requires_msa_for_fit and training_msa is None:
            return False
        if model._requires_wt_to_function and wt is None:
            return False
        if model._requires_structure and not check_structures_available():
            return False
        return True

    models = load_models()
    compatibility = {"compatible": [], "incompatible": []}

    for model in models:
        if check_compatibility(model):
            compatibility["compatible"].append(model.__name__)
        else:
            compatibility["incompatible"].append(model.__name__)

    return compatibility

def check_dvc_params():
    """Ensures the user provided the necessary data and did not ask for
    incompatible steps.

    Returns pre-run metrics about the pipeline.


    Checks done:
    - If the user asks for a model that requires an MSA, ensure the MSA step is on.
    - If the user is using specifying specific positions to score, ensure that any
      sequences to be evaluated are fixed length and all models are capable of position specific scoring.
      Currently incompatable with supervised models.
    - If the user gives either training or test data, and any have different legnths, ensure that
      models are capable of handling variable length sequences.
    - If the user asks for jackhmmer search, ensure that wt.fasta was provided
    - If the user asks for supervised and or/msa mode that adds training sequences, ensure that the training sequences are provided.
    - If the user asks for CV, ensure training data is provided.
    """
    params = dvc.api.params_show()
    if not 'data' in os.listdir() and 'dvc.yaml' in os.listdir():
        raise FileNotFoundError("This directory does not appear to be the aide_predict dvc pipeline. `check_dvc_params` must be run from the root of the dvc pipeline.")


def check_input_sequences_against_model_capability():
    # we need to catch if the user passed non fixed length sequences and the model is not capable of that
    # or if it is capable of not fixed length AS WELL AS position specific scoring, how can both be recitifed?
    # If the user passes an alignment and positions, that might work.     
    
    pass