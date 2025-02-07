# aide_predict/utils/checks.py
'''
* Author: Evan Komp
* Created: 6/13/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Common checks to ensure that different pipeline components are compatable.
'''
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

def get_supported_tools():
    from aide_predict.bespoke_models import TOOLS
    out_string = ""
    for tool in TOOLS:
        avail = tool._available
        if avail:
            message = 'AVAILABLE'
        else:
            message = tool._available.message
        out_string += tool.__name__ +f": {message}\n"
    print(out_string)
    return out_string