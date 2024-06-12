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
    