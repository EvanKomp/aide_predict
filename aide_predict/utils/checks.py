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
    wt: Optional[ProteinSequence] = None
) -> Dict[str, List[str]]:
    """
    Check which models are compatible with the given data.

    Args:
        training_sequences (Optional[ProteinSequences]): Training protein sequences.
        testing_sequences (Optional[ProteinSequences]): Testing protein sequences.
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
        """Check if any structure information is available in the provided data."""
        if wt and wt.structure is not None:
            return True
        for seq_set in [training_sequences, testing_sequences]:
            if seq_set:
                if any(seq.structure is not None for seq in seq_set):
                    return True
        return False
    
    def check_msa_for_fit_available() -> bool:
        """Check if an MSA is available for fitting."""
        # First check if training sequences are already aligned
        if training_sequences and training_sequences.aligned:
            return True
        # Then check if WT has an MSA
        if wt and wt.has_msa:
            return True
        return False
    
    def check_wt_msa_available() -> bool:
        """Check if wild-type has an associated MSA."""
        return wt is not None and wt.has_msa
    
    def check_msa_per_sequence_available() -> bool:
        """Check if each sequence has its own MSA."""
        if training_sequences:
            if all(seq.has_msa for seq in training_sequences):
                return True
        if testing_sequences:
            if all(seq.has_msa for seq in testing_sequences):
                return True
        return False

    def check_compatibility(model: Type[ProteinModelWrapper]) -> bool:
        """Check if the model is compatible with the provided data."""
        if not model._available:
            return False
            
        # Check fixed length requirement
        if model._requires_fixed_length:
            if (training_sequences and not training_sequences.fixed_length) or \
               (testing_sequences and not testing_sequences.fixed_length):
                return False
                
        # Check MSA-related requirements
        if model._requires_msa_for_fit and not check_msa_for_fit_available():
            return False
        if model._requires_wt_msa and not check_wt_msa_available():
            return False
        if model._requires_msa_per_sequence and not check_msa_per_sequence_available():
            return False
            
        # Check wild-type requirement
        if model._requires_wt_to_function and wt is None:
            return False
            
        # Check structure requirement
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