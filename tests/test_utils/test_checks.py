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

# Assuming the check_model_compatibility function is in a module named model_utils
from aide_predict.utils.checks import check_model_compatibility

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

class MockUnavailableModel(ProteinModelWrapper):
    _available = False

# Mock function to replace inspect.getmembers
def mock_getmembers(module):
    return [
        ('MockFixedLengthModel', MockFixedLengthModel),
        ('MockMSAModel', MockMSAModel),
        ('MockWTModel', MockWTModel),
        ('MockStructureModel', MockStructureModel),
        ('MockUnavailableModel', MockUnavailableModel)
    ]

@pytest.fixture(autouse=True)
def mock_inspect_getmembers(monkeypatch):
    def patched_getmembers(module):
        if module == aide_predict:
            return mock_getmembers(module)
        return inspect.getmembers(module)
    monkeypatch.setattr(inspect, 'getmembers', patched_getmembers)

# Test data
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
def structure():
    return ProteinStructure(os.path.join("tests", "data", "ENVZ_ECOLI.pdb"))

# Tests
def test_fixed_length_compatibility(fixed_length_sequences, variable_length_sequences):
    result = check_model_compatibility(training_sequences=fixed_length_sequences)
    assert "MockFixedLengthModel" in result["compatible"]
    
    result = check_model_compatibility(training_sequences=variable_length_sequences)
    assert "MockFixedLengthModel" in result["incompatible"]

def test_msa_compatibility(msa_sequences):
    result = check_model_compatibility(training_msa=msa_sequences)
    assert "MockMSAModel" in result["compatible"]
    
    result = check_model_compatibility(training_sequences=msa_sequences)
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

def test_all_compatible(fixed_length_sequences, msa_sequences, wt_sequence, structure):
    sequences_with_structure = ProteinSequences([
        ProteinSequence("ACDEFG", structure=structure),
        ProteinSequence("HIJKLM")
    ])
    result = check_model_compatibility(
        training_sequences=sequences_with_structure,
        training_msa=msa_sequences,
        wt=wt_sequence
    )
    assert set(result["compatible"]) == {
        "MockFixedLengthModel", "MockMSAModel", "MockWTModel", "MockStructureModel"
    }
    assert "MockUnavailableModel" in result["incompatible"]