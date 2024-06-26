# tests/utils/data_structures.py
'''
* Author: Evan Komp
* Created: 6/26/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

We should be able to handle on file and in memory protein sequences, both aligned and unaligned.
'''

import pytest

from aide_predict.utils.data_structures import ProteinCharacter, ProteinSequence, ProteinSequences, ProteinSequencesOnFile


def test_protein_character()
    

    # smoke test creation
    # bad character
    with pytest.raises(ValueError):
        ProteinCharacter("Z")

    # good character
    ProteinCharacter("A")

    # test properties
    assert ProteinCharacter("A").is_gap == False
    assert ProteinCharacter("-").is_gap == True
    assert ProteinCharacter("A").is_non_canonical == False
    assert ProteinCharacter("U").is_non_canonical == True
    assert ProteinCharacter("A").is_not_focus == False
    assert ProteinCharacter("a").is_not_focus == True
    assert ProteinCharacter("-").is_not_focus == True

def test_protein_sequence():
    # smoke test creation
    ProteinSequence("ACDEFGHIKLMNPQRSTVWY")

    # test properties
    assert ProteinSequence("ACDEFGHIKLMNPQRSTVWY")[0] == ProteinCharacter("A")
    assert str(ProteinSequence("ACDEFGHIKLMNPQRSTVWY")) == "ACDEFGHIKLMNPQRSTVWY"
    assert ProteinSequence("ACDEFGHIKLMNPQRSTVWY").has_gaps == False

    

