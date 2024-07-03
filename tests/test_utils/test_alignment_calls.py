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
    def sample_sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY", id="seq1"),
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
        