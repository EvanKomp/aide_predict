# tests/test_utils/test_msa.py
'''
* Author: Evan Komp
* Created: 8/16/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import pytest
import numpy as np
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.utils.msa import MSAProcessing

class TestMSAProcessing:
    @pytest.fixture
    def sample_msa(self):
        sequences = [
            ProteinSequence("MKYKVLPQGTVMKVLPK-TVMKVLPQGTV", id="seq1"),
            ProteinSequence("MKY-VLPQGTV-KVLPKGTVMKVLPQGTV", id="seq2"),
            ProteinSequence("MKYKVL-QGTVMKVLPKGTVMKVLPQGTV", id="seq3"),
            ProteinSequence("MKYKVLPQGTVMKVLPKGTVMKVLPQGTV", id="seq4"),
            ProteinSequence("MKY-VLP------VLPKGTVMKVLPQGTV", id="seq5"),
        ]
        return ProteinSequences(sequences)

    @pytest.fixture
    def msa_processor(self):
        return MSAProcessing(
            theta=0.2,
            threshold_sequence_frac_gaps=0.2,
            threshold_focus_cols_frac_gaps=0.3
        )

    def test_initialization(self, msa_processor):
        assert msa_processor.theta == 0.2
        assert msa_processor.threshold_sequence_frac_gaps == 0.2
        assert msa_processor.threshold_focus_cols_frac_gaps == 0.3

    def test_process_with_focus(self, sample_msa, msa_processor):
        processed_msa = msa_processor.process(sample_msa, focus_seq_id="seq1")
        assert len(processed_msa) == 4  # seq5 should be removed due to high gap fraction
        assert processed_msa.weights is not None
        assert len(processed_msa.weights) == 4

    def test_process_without_focus(self, sample_msa, msa_processor):
        processed_msa = msa_processor.process(sample_msa)
        assert len(processed_msa) == 4  # seq5 should be removed due to high gap fraction
        assert processed_msa.weights is not None
        assert len(processed_msa.weights) == 4
        # Check that all sequences are uppercase when no focus is provided
        assert all(seq.isupper() for seq in processed_msa)

    def test_preprocess_msa(self, sample_msa, msa_processor):
        msa_processor.focus_seq = sample_msa[0]
        preprocessed_msa = msa_processor._preprocess_msa(sample_msa)
        assert len(preprocessed_msa) == 4  # seq5 should be removed
        assert '-' not in str(preprocessed_msa[0])

    def test_get_focus_columns(self, sample_msa, msa_processor):
        focus_cols = msa_processor._get_focus_columns(sample_msa[0])
        assert len(focus_cols) == 29
        assert np.sum(focus_cols) == 28  # All but 1 columns should be focus columns for seq1

    def test_compute_weights(self, sample_msa, msa_processor):
        weights = msa_processor._compute_weights(sample_msa)
        assert len(weights) == 5
        assert sum(weights) < 5.0  # Weights should sum to less than 5.0 as some sequences are downweighted

    def test_no_focus_all_uppercase(self, sample_msa, msa_processor):
        processed_msa = msa_processor.process(sample_msa)
        assert all(seq.isupper() for seq in processed_msa)

    def test_with_focus_lowercase_non_focus(self, sample_msa, msa_processor):
        processed_msa = msa_processor.process(sample_msa, focus_seq_id="seq1")
        assert any(not char.isupper() for seq in processed_msa for char in str(seq))

    def test_remove_high_gap_sequences(self):
        
        sequences = [
            ProteinSequence("MKYKVLPQGTV", id="seq1"),
            ProteinSequence("MKY-V-PQG-V", id="seq2"),  # 30% gaps
            ProteinSequence("MKYKVLPQGTV", id="seq3"),
        ]
        msa = ProteinSequences(sequences)
        msa_processor = MSAProcessing(threshold_sequence_frac_gaps=0.2)
        processed_msa = msa_processor.process(msa)
        assert len(processed_msa) == 2
        assert "seq2" not in [seq.id for seq in processed_msa]

    def test_weight_computation_batch_size(self, sample_msa):
        msa_processor = MSAProcessing(weight_computation_batch_size=2)
        weights = msa_processor._compute_weights(sample_msa)
        assert len(weights) == 5

    def test_indeterminate_aa_removal(self):
        sequences = [
            ProteinSequence("MKYKVLPQGTV", id="seq1"),
            ProteinSequence("MKYXVLPQGTV", id="seq2"),  # Contains 'X'
            ProteinSequence("MKYKVLPQGTV", id="seq3"),
        ]
        msa = ProteinSequences(sequences)
        msa_processor = MSAProcessing(remove_sequences_with_indeterminate_aa_in_focus_cols=True)
        processed_msa = msa_processor.process(msa)
        assert len(processed_msa) == 2
        assert "seq2" not in [seq.id for seq in processed_msa]