# aide_predict/utils/common.py
'''
* Author: Evan Komp
* Created: 6/11/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Common utility functions

'''
from aide_predict.utils.constants import AA_ALPHABET

from typing import Iterable
from types import SimpleNamespace


def process_amino_acid_sequences(sequences: Iterable[str]) -> Iterable[str]:
    """Processes a list of amino acid sequences to ensure that they are valid.
    Improper AAs raise.
    
    Params:
        sequences (Iterable[str]): A list of amino acid sequences.
    
    Returns:
        Iterable[str]: The same list of amino acid sequences, but capitalized.

    """
    for seq in sequences:
        if not set(seq).issubset(AA_ALPHABET):
            raise ValueError(f"Invalid amino acid in sequence: {seq}")
        yield seq.upper()
    
def fixed_length_sequences(sequences: Iterable[str]):
    """Check if all sequences are the same length, if not, raise.
    
    Params:
        sequences (Iterable[str]): A list of amino acid sequences.
    """
    sequences = process_amino_acid_sequences(sequences)
    lens = [len(seq) for seq in sequences]
    if not len(set(lens)) == 1:
        return False
    else:
        return True
    
def mutated_positions(sequences: Iterable[str]):
    """Find the positions of mutations in a list of sequences.
    
    Params:
        sequences (Iterable[str]): A list of amino acid sequences.
    
    Returns:
        list[int]: A list of positions that are mutated, zero indexed
    """
    sequences = list(process_amino_acid_sequences(sequences))
    if not fixed_length_sequences(sequences):
        raise ValueError("All sequences must be the same length.")
    positions = []
    for i in range(len(sequences[0])):
        if len(set(seq[i] for seq in sequences)) > 1:
            positions.append(i)
    return positions

def convert_dvc_params(dvc_params_dict: dict):
    """DVC Creates a nested dict with the parameters.
    
    We want an object that has nested attributes so that we can
    access parameters with dot notation.
    """
    def _dict_to_obj(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: _dict_to_obj(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [_dict_to_obj(x) for x in d]
        else:
            return d
    return _dict_to_obj(dvc_params_dict)
