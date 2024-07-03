# tests/utils/data_structures.py
'''
* Author: Evan Komp
* Created: 6/26/2024
* Company: Bottle Institute @ National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

We should be able to handle on file and in memory protein sequences, both aligned and unaligned.
'''

import pytest
import shutil
from io import StringIO
from tempfile import NamedTemporaryFile, mkdtemp
import os
import numpy as np

from aide_predict.utils.data_structures import ProteinCharacter, ProteinSequence, ProteinSequences, ProteinSequencesOnFile
from aide_predict.utils.constants import AA_SINGLE, GAP_CHARACTERS, NON_CONONICAL_AA_SINGLE

# Test data
VALID_AA = AA_SINGLE
INVALID_AA = "2"
GAP_CHAR = "-"
    

class TestProteinCharacter:
    def test_valid_initialization(self):
        for aa in VALID_AA:
            pc = ProteinCharacter(aa)
            assert pc == aa

    def test_invalid_initialization(self):
        with pytest.raises(ValueError):
            ProteinCharacter(INVALID_AA)

    def test_gap_character(self):
        pc = ProteinCharacter(GAP_CHAR)
        assert pc.is_gap

    def test_non_canonical(self):
        pc = ProteinCharacter("X")
        assert pc.is_non_canonical

    def test_not_focus(self):
        pc = ProteinCharacter("a")
        assert pc.is_not_focus

    def test_equality(self):
        pc1 = ProteinCharacter("A")
        pc2 = ProteinCharacter("A")
        assert pc1 == pc2
        assert pc1 == "A"

    def test_hash(self):
        pc = ProteinCharacter("A")
        assert hash(pc) == hash("A")

class TestProteinSequence:
    @pytest.fixture(scope="class")
    def temp_pdb_file(self, tmp_path_factory):
        # Create a temporary directory
        temp_dir = tmp_path_factory.mktemp("pdb_test")
        
        # Create a dummy PDB file
        pdb_path = temp_dir / "sample.pdb"
        pdb_path.write_text("DUMMY PDB CONTENT")
        
        return pdb_path

    @pytest.fixture
    def sample_sequence(self, temp_pdb_file):
        return ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="sample", structure=str(temp_pdb_file))
    
    def test_as_array(self, sample_sequence):
        arr = sample_sequence.as_array
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1, 20)
        assert ''.join(arr[0]) == str(sample_sequence)

    def test_initialization(self, sample_sequence, temp_pdb_file):
        assert str(sample_sequence) == "ACDEFGHIKLMNPQRSTVWY"
        assert sample_sequence.id == "sample"
        assert sample_sequence.structure == str(temp_pdb_file)

    def test_properties(self, sample_sequence):
        assert not sample_sequence.has_gaps
        assert not sample_sequence.has_non_canonical
        assert sample_sequence.num_gaps == 0
        assert sample_sequence.base_length == 20

        gapped = ProteinSequence("ACD-")
        assert gapped.has_gaps
        assert gapped.num_gaps == 1
        assert gapped.base_length == 3
        assert gapped.with_no_gaps() == ProteinSequence("ACD")


    def test_mutation(self, sample_sequence):
        mutated = sample_sequence.mutate(0, "M")
        assert str(mutated) == "MCDEFGHIKLMNPQRSTVWY"
        assert mutated.id is None  # id should not be passed to indicate mutation

    def test_mutated_positions(self, sample_sequence):
        other = ProteinSequence("MCDEFGHIKLMNPQRSTVWY")
        assert sample_sequence.mutated_positions(other) == [0]

    def test_get_protein_character(self, sample_sequence):
        assert isinstance(sample_sequence.get_protein_character(0), ProteinCharacter)
        assert sample_sequence.get_protein_character(0) == "A"

    def test_slice(self, sample_sequence):
        sliced = sample_sequence.slice_as_protein_sequence(0, 5)
        assert str(sliced) == "ACDEF"
        assert isinstance(sliced, ProteinSequence)

    def test_iter_protein_characters(self, sample_sequence):
        chars = list(sample_sequence.iter_protein_characters())
        assert all(isinstance(c, ProteinCharacter) for c in chars)
        assert "".join(str(c) for c in chars) == str(sample_sequence)

    def test_align(self, sample_sequence):
        # Create another sequence to align with
        other_sequence = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="other")
        
        # Introduce a gap in the middle of other_sequence
        other_sequence = ProteinSequence("ACDEFGHIKMNPQRSTVWY", id="other")
        
        # Align the sequences
        aligned_self, aligned_other = sample_sequence.align(other_sequence)
        
        # Check that the aligned sequences are ProteinSequence objects
        assert isinstance(aligned_self, ProteinSequence)
        assert isinstance(aligned_other, ProteinSequence)
        
        # Check that the aligned sequences have the same length
        assert len(aligned_self) == len(aligned_other)
        
        # Check that the original IDs are preserved
        assert aligned_self.id == "sample"
        assert aligned_other.id == "other"
        
        # Check that the alignment introduced gaps in the right places
        assert "-" in str(aligned_other)
        assert str(aligned_other) == "ACDEFGHIK-MNPQRSTVWY"
        
        # Check that the non-gap parts of the sequences match the originals
        assert aligned_self.with_no_gaps() == sample_sequence
        assert aligned_other.with_no_gaps() == other_sequence.with_no_gaps()
        
        # Check that the structures are preserved
        assert aligned_self.structure == sample_sequence.structure
        assert aligned_other.structure == other_sequence.structure

    def test_hash(self, sample_sequence):
        assert isinstance(hash(sample_sequence), int)

    def test_equality(self, sample_sequence):
        same_sequence = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="sample")
        different_sequence = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="different")
        assert sample_sequence == same_sequence
        assert sample_sequence != different_sequence

# ProteinSequences tests
class TestProteinSequences:
    @pytest.fixture
    def sample_sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDE", id="seq1"),
            ProteinSequence("ACDF", id="seq2"),
            ProteinSequence("ACD-", id="seq3")
        ])

    def test_properties(self, sample_sequences):
        assert sample_sequences.aligned
        assert not sample_sequences.fixed_length
        assert sample_sequences.width == 4
        assert sample_sequences.has_gaps

    def test_mutated_positions(self, sample_sequences):
        assert sample_sequences.mutated_positions == [3]

    def test_to_dict(self, sample_sequences):
        d = sample_sequences.to_dict()
        assert d == {"seq1": "ACDE", "seq2": "ACDF", "seq3": "ACD-"}

    def test_to_fasta(self, sample_sequences):
        with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            sample_sequences.to_fasta(temp_file.name)
            temp_file.seek(0)
            content = temp_file.read()
        assert content == ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACD-\n"
        os.unlink(temp_file.name)

    def test_from_fasta(self):
        fasta_content = ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACD-\n"
        with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(fasta_content)
            temp_file.flush()
            sequences = ProteinSequences.from_fasta(temp_file.name)
        assert len(sequences) == 3
        assert str(sequences[0]) == "ACDE"
        assert sequences[0].id == "seq1"
        os.unlink(temp_file.name)

    def test_with_no_gaps(self, sample_sequences):
        no_gaps = sample_sequences.with_no_gaps()
        assert isinstance(no_gaps, ProteinSequences)
        assert len(no_gaps) == 3
        assert str(no_gaps[2]) == "ACD"

    def test_get_alignment_mapping(self, sample_sequences):
        sample_sequences.append(ProteinSequence("A-DE", id="seq4"))
        mapping = sample_sequences.get_alignment_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) == 4
        assert mapping[3] == [0, 2, 3] # this should not inclide the gat at position 1

    def test_apply_alignment_mapping(self, sample_sequences):
        sample_sequences = sample_sequences.with_no_gaps()
        mapping = {
            0: [0,1, 2, 5],
            1: [0,1, 2, 5],
            2: [0,1, 5]
        }
        aligned = sample_sequences.apply_alignment_mapping(mapping)
        assert isinstance(aligned, ProteinSequences)
        assert len(aligned) == 3
        assert str(aligned[0]) == "ACD--E"
        assert str(aligned[1]) == "ACD--F"
        assert str(aligned[2]) == "AC---D"

    def test_as_array(self, sample_sequences):
        arr = sample_sequences.as_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 4)
        assert ''.join(arr[0]) == "ACDE"

    def test_iter_batches(self, sample_sequences):
        batches = list(sample_sequences.iter_batches(2))
        assert len(batches) == 2
        assert isinstance(batches[0], ProteinSequences)
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

# ProteinSequencesOnFile tests
class TestProteinSequencesOnFile:
    @pytest.fixture
    def sample_fasta_file(self):
        content = ">seq1\nACDE\n>seq2\nACDF\n>seq3\nACD-\n"
        with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            yield temp_file.name
        os.unlink(temp_file.name)

    def test_initialization(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        assert len(sequences) == 3
        assert sequences.aligned
        assert not sequences.fixed_length
        assert sequences.width == 4
        assert sequences.has_gaps

    def test_getitem(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        assert str(sequences[0]) == "ACDE"
        assert str(sequences["seq2"]) == "ACDF"
        with pytest.raises(IndexError):
            sequences[3]
        with pytest.raises(KeyError):
            sequences["nonexistent"]

    def test_iteration(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        seqs = list(sequences)
        assert len(seqs) == 3
        assert all(isinstance(seq, ProteinSequence) for seq in seqs)

    def test_mutated_positions(self, sample_fasta_file):
        sequences = ProteinSequencesOnFile(sample_fasta_file)
        assert sequences.mutated_positions == [3]

    def test_to_memory(self, sample_fasta_file):
        on_file = ProteinSequencesOnFile(sample_fasta_file)
        in_memory = on_file.to_memory()
        assert isinstance(in_memory, ProteinSequences)
        assert len(in_memory) == 3
        assert str(in_memory[0]) == "ACDE"

# Integration tests
def test_integration():
    # Create a ProteinSequence
    seq = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="test_seq")
    
    # Mutate it
    mutated_seq = seq.mutate(0, "M")
    
    # Create ProteinSequences
    sequences = ProteinSequences([seq, mutated_seq])
    
    # Check properties
    assert sequences.aligned
    assert sequences.fixed_length
    assert sequences.width == 20
    assert not sequences.has_gaps
    assert sequences.mutated_positions == [0]
    
    # Write to FASTA and read back
    with NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        sequences.to_fasta(temp_file.name)
        on_file = ProteinSequencesOnFile.from_fasta(temp_file.name)
    
    # Check if the data is preserved
    assert len(on_file) == 2
    assert str(on_file[0]) == str(seq)
    assert str(on_file[1]) == str(mutated_seq)
    
    os.unlink(temp_file.name)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])