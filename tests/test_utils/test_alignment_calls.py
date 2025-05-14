# tests/test_utils/test_alignment_calls.py
'''
* Author: Evan Komp
* Created: 7/2/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''

import pytest
import os
import tempfile
import subprocess
from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences, ProteinSequencesOnFile
from aide_predict.utils.alignment_calls import mafft_align

class TestMAFFTAlignment:
    @pytest.fixture
    def sample_sequences(self, tmp_path):
        # Create a mock structure
        pdb_path = tmp_path / "mock.pdb"
        with open(pdb_path, "w") as f:
            # put some actual pdb content here
            f.write("HEADER    MOCK PDB FILE\n")
            f.write("ATOM      1  N   ALA A   1      20.154  34.123  27.456  1.00 20.00\n")
            f.write("ATOM      2  CA  ALA A   1      21.123  35.456  28.789  1.00 20.00\n")
        
        # Create MSA for seq1
        msa = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="msa_seq1"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWF", id="msa_seq2")
        ])
        
        seq1 = ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq1", structure=str(pdb_path))
        seq1.msa = msa  # Add MSA to seq1
        
        return ProteinSequences([
            seq1,
            ProteinSequence("ACDEFGHIKLMNPQRSTVW", id="seq2"),
            ProteinSequence("ACDEFGHIKLMNPQRST", id="seq3")
        ])

    @pytest.fixture
    def sample_sequences_on_file(self, sample_sequences, tmp_path):
        file_path = tmp_path / "input.fasta"
        sample_sequences.to_fasta(str(file_path))
        return ProteinSequencesOnFile(str(file_path))

    @pytest.fixture
    def mock_mafft_add(self, monkeypatch):
        def mock_run(*args, **kwargs):
            output_file = args[0].split(">")[-1].strip()
            with open(output_file, 'w') as f:
                f.write(">seq1\nACDEFGHIKLMNPQRSTVWY-\n>seq2\nACDEFGHIKLMNPQRSTVW--\n>seq3\nACDEFGHIKLMNPQRST----\n>existing1\nACDEFGHIKLMNPQRSTVWY-\n>existing2\nACDEFGHIKLMNPQRSTVW--\n")
        monkeypatch.setattr(subprocess, "run", mock_run)

    @pytest.fixture
    def mock_mafft_new(self, monkeypatch):
        def mock_run(*args, **kwargs):
            output_file = args[0].split(">")[-1].strip()
            with open(output_file, 'w') as f:
                f.write(">seq1\nACDEFGHIKLMNPQRSTVWY-\n>seq2\nACDEFGHIKLMNPQRSTVW--\n>seq3\nACDEFGHIKLMNPQRST----\n")
        monkeypatch.setattr(subprocess, "run", mock_run)

    def test_mafft_align_new(self, sample_sequences, mock_mafft_new, tmp_path):
        output_file = str(tmp_path / "output.fasta")
        result = mafft_align(sample_sequences, output_fasta=output_file)
        
        assert isinstance(result, ProteinSequencesOnFile)
        assert os.path.exists(output_file)
        
        aligned_sequences = result.to_memory()
        assert len(aligned_sequences) == 3
        assert all(len(seq) == 21 for seq in aligned_sequences)
        assert aligned_sequences[0].id == "seq1"
        assert str(aligned_sequences[0]) == "ACDEFGHIKLMNPQRSTVWY-"

    def test_mafft_align_existing(self, sample_sequences, mock_mafft_add):
        existing_alignment = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY-", id="existing1"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVW--", id="existing2")
        ])
        
        result = mafft_align(sample_sequences, existing_alignment=existing_alignment, realign=False)
        
        assert isinstance(result, ProteinSequences)
        assert len(result) == 5  # 3 new sequences + 2 existing
        assert all(len(seq) == 21 for seq in result)

    def test_mafft_align_realign(self, sample_sequences, mock_mafft_add):
        existing_alignment = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY-", id="existing1"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVW--", id="existing2")
        ])
        
        result = mafft_align(sample_sequences, existing_alignment=existing_alignment, realign=True)
        
        assert isinstance(result, ProteinSequences)
        assert len(result) == 5  # All sequences realigned
        assert all(len(seq) == 21 for seq in result)

    def test_mafft_not_installed(self, sample_sequences, monkeypatch):
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("MAFFT not found")
        monkeypatch.setattr(subprocess, "run", mock_run)
        
        with pytest.raises(FileNotFoundError, match="MAFFT is not installed or not in PATH"):
            mafft_align(sample_sequences)

    def test_mafft_align_preserves_structure(self, sample_sequences, mock_mafft_new):
        # seq1 has structure in the fixture
        result = mafft_align(sample_sequences)
        
        # Check that structure is preserved for seq1
        assert result[0].id == "seq1"
        assert result[0].has_structure
        assert result[0].structure == sample_sequences[0].structure
        
        # Verify other sequences without structure remain without structure
        assert not result[1].has_structure
        assert not result[2].has_structure

    def test_mafft_align_preserves_msa(self, sample_sequences, mock_mafft_new):
        # seq1 has MSA in the fixture
        result = mafft_align(sample_sequences)
        
        # Check that MSA is preserved for seq1
        assert result[0].id == "seq1"
        assert result[0].has_msa
        assert result[0].msa == sample_sequences[0].msa
        
        # Verify other sequences without MSA remain without MSA
        assert not result[1].has_msa
        assert not result[2].has_msa

    def test_mafft_align_handles_sequences_without_ids(self, mock_mafft_new, sample_sequences):
        # Create sequences without IDs
        sequences = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY"),  # No ID
            ProteinSequence("ACDEFGHIKLMNPQRSTVW", id="seq2"),
            ProteinSequence("ACDEFGHIKLMNPQRST", id="seq3")
        ])
        
        # Add structure to the first sequence
        sequences[0].structure = sample_sequences[0].structure
        
        result = mafft_align(sequences)
        
        # Check structure is preserved using hash matching
        assert result[0].has_structure

    def test_mafft_with_gapped_sequences(self, sample_sequences_on_file):
        # Add gaps to the sequences
        gapped_sequences = ProteinSequences([
            ProteinSequence("ACDEF-GHIKLMNPQRSTVWY", id="seq1"),
            ProteinSequence("ACDEF-GHIKLMNPQRSTVW", id="seq2"),
            ProteinSequence("ACDEF-GHIKLMNPQRST", id="seq3")
        ])
        
        with pytest.raises(ValueError, match="Input sequences should not contain gaps"):
            mafft_align(gapped_sequences)
    
    def test_mafft_align_sequence_api_integrity(self, sample_sequences, mock_mafft_new):
        """Test that the aligned sequences maintain proper ProteinSequence API functionality"""
        result = mafft_align(sample_sequences)
        
        # Test basic methods and properties
        assert result.aligned
        assert result.width == 21
        assert result.has_gaps
        
        # Test sequence-specific operations
        seq = result[0]
        # Should keep original ungapped sequence when gaps removed
        assert str(seq.with_no_gaps()) == "ACDEFGHIKLMNPQRSTVWY"
        # Should handle character-by-character operations
        assert seq.get_protein_character(0) == 'A'
        