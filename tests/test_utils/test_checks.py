# tests/test_utils/test_checks.py
'''
* Author: Evan Komp
* Created: 8/16/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import os
import pytest
import inspect
from typing import List, Type
import aide_predict
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence, ProteinStructure
from aide_predict.bespoke_models.base import ProteinModelWrapper

# Import the updated check_model_compatibility function
from aide_predict.utils.checks import check_model_compatibility

# Note: The current implementation of check_model_compatibility doesn't have a training_msa parameter
# If we pass training_msa as an argument in tests, it won't be recognized by the function

# Mock models for testing
class MockFixedLengthModel(ProteinModelWrapper):
    _requires_fixed_length = True
    _available = True

class MockMSAModel(ProteinModelWrapper):
    _requires_msa_for_fit = True
    _available = True

class MockWTModel(ProteinModelWrapper):
    _requires_wt_to_function = True
    _available = True

class MockStructureModel(ProteinModelWrapper):
    _requires_structure = True
    _available = True

class MockWTMSAModel(ProteinModelWrapper):
    _requires_wt_to_function = True
    _requires_wt_msa = True
    _available = True

class MockMSAPerSequenceModel(ProteinModelWrapper):
    _requires_msa_per_sequence = True
    _can_handle_aligned_sequences = True
    _available = True

class MockUnavailableModel(ProteinModelWrapper):
    _available = False

# Mock function to replace inspect.getmembers
def mock_getmembers(module):
    return [
        ('MockFixedLengthModel', MockFixedLengthModel),
        ('MockMSAModel', MockMSAModel),
        ('MockWTModel', MockWTModel),
        ('MockStructureModel', MockStructureModel),
        ('MockWTMSAModel', MockWTMSAModel),
        ('MockMSAPerSequenceModel', MockMSAPerSequenceModel),
        ('MockUnavailableModel', MockUnavailableModel)
    ]

@pytest.fixture(autouse=True)
def mock_inspect_getmembers(monkeypatch):
    def patched_getmembers(module):
        if module == aide_predict:
            return mock_getmembers(module)
        return inspect.getmembers(module)
    monkeypatch.setattr(inspect, 'getmembers', patched_getmembers)

# Test data fixtures
@pytest.fixture
def fixed_length_sequences():
    return ProteinSequences([ProteinSequence("ACDEFG"), ProteinSequence("HIJKLM")])

@pytest.fixture
def variable_length_sequences():
    return ProteinSequences([ProteinSequence("ACDEFG"), ProteinSequence("HIJKLMNOP")])

@pytest.fixture
def msa_sequences():
    return ProteinSequences([ProteinSequence("ACDE-FG"), ProteinSequence("ACDEFGH")])

@pytest.fixture
def wt_sequence():
    return ProteinSequence("ACDEFGHIJKLM")

@pytest.fixture
def wt_with_msa(msa_sequences):
    wt = ProteinSequence("ACDEFGH")
    wt.msa = msa_sequences
    return wt

@pytest.fixture
def sequences_with_msas(msa_sequences):
    # Create sequences where each has its own MSA
    seq1 = ProteinSequence("ACDEFG")
    seq1.msa = msa_sequences
    seq2 = ProteinSequence("HIJKLM")
    seq2.msa = msa_sequences
    return ProteinSequences([seq1, seq2])

@pytest.fixture
def structure():
    return ProteinStructure(os.path.join("tests", "data", "ENVZ_ECOLI.pdb"))

# Tests for existing functionality
def test_fixed_length_compatibility(fixed_length_sequences, variable_length_sequences):
    result = check_model_compatibility(training_sequences=fixed_length_sequences)
    assert "MockFixedLengthModel" in result["compatible"]
    
    result = check_model_compatibility(training_sequences=variable_length_sequences)
    assert "MockFixedLengthModel" in result["incompatible"]

def test_msa_compatibility(msa_sequences):
    # The original test used training_msa but that's not in the function signature
    # Instead, we'll rely on aligned training sequences which should satisfy the MSA requirement
    
    result = check_model_compatibility(training_sequences=msa_sequences)
    assert "MockMSAModel" in result["compatible"], "Aligned sequences should be compatible with MSA models"
    
    # Test with non-aligned sequences
    result = check_model_compatibility(training_sequences=ProteinSequences([ProteinSequence("ACDEFG")]))
    assert "MockMSAModel" in result["incompatible"]

def test_wt_compatibility(wt_sequence):
    result = check_model_compatibility(wt=wt_sequence)
    assert "MockWTModel" in result["compatible"]
    
    result = check_model_compatibility()
    assert "MockWTModel" in result["incompatible"]

def test_structure_compatibility(structure, fixed_length_sequences):
    sequences_with_structure = ProteinSequences([
        ProteinSequence("ACDEFG", structure=structure),
        ProteinSequence("HIJKLM")
    ])
    result = check_model_compatibility(training_sequences=sequences_with_structure)
    assert "MockStructureModel" in result["compatible"]
    
    result = check_model_compatibility(training_sequences=fixed_length_sequences)
    assert "MockStructureModel" in result["incompatible"]

def test_unavailable_model():
    result = check_model_compatibility()
    assert "MockUnavailableModel" in result["incompatible"]

# Tests for new functionality
def test_wt_msa_compatibility(wt_with_msa, wt_sequence):
    result = check_model_compatibility(wt=wt_with_msa)
    assert "MockWTMSAModel" in result["compatible"], "WT with MSA should be compatible with WT MSA models"
    
    # Test with WT without MSA
    result = check_model_compatibility(wt=wt_sequence)
    assert "MockWTMSAModel" in result["incompatible"], "WT without MSA should be incompatible with WT MSA models"

def test_msa_per_sequence_compatibility(sequences_with_msas, fixed_length_sequences, wt_with_msa):
    result = check_model_compatibility(training_sequences=sequences_with_msas)
    assert "MockMSAPerSequenceModel" in result["compatible"], "Sequences with MSAs should be compatible with MSA per sequence models"
    
    # Test with sequences without MSAs
    result = check_model_compatibility(training_sequences=fixed_length_sequences)
    assert "MockMSAPerSequenceModel" in result["incompatible"], "Sequences without MSAs should be incompatible with MSA per sequence models"

    # check that wt msa can override it
    result = check_model_compatibility(wt=wt_with_msa)
    assert "MockMSAPerSequenceModel" in result["compatible"], "WT with MSA should be compatible with MSA per sequence models"

def test_msa_from_aligned_training(msa_sequences):
    result = check_model_compatibility(training_sequences=msa_sequences)
    assert "MockMSAModel" in result["compatible"], "Aligned training sequences should be compatible with MSA models"

def test_msa_from_wt(wt_with_msa):
    # Test that a model requiring MSA for fit is compatible when only WT has an MSA
    result = check_model_compatibility(wt=wt_with_msa)
    assert "MockMSAModel" in result["compatible"], "WT with MSA should satisfy MSA for fit requirement"

def test_all_compatible(fixed_length_sequences, msa_sequences, wt_with_msa, structure):
    # Create sequences with structure and MSAs
    seq1 = ProteinSequence("ACDEFG", structure=structure)
    seq1.msa = msa_sequences
    seq2 = ProteinSequence("HIJKLM", structure=structure)
    seq2.msa = msa_sequences
    sequences_with_all = ProteinSequences([seq1, seq2])
    
    result = check_model_compatibility(
        training_sequences=sequences_with_all,
        wt=wt_with_msa
    )
    
    # All mock models except the unavailable one should be compatible
    expected_compatible = {
        "MockFixedLengthModel", 
        "MockMSAModel", 
        "MockWTModel", 
        "MockStructureModel",
        "MockWTMSAModel",
        "MockMSAPerSequenceModel"
    }
    
    assert set(result["compatible"]) == expected_compatible
    assert "MockUnavailableModel" in result["incompatible"]