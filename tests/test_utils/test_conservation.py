# tests/test_utils/test_conservation.py
'''
* Author: Evan Komp
* Created: 2/10/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''

import pytest
import numpy as np
from aide_predict.utils.conservation import ConservationAnalysis
from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences

class TestConservationAnalysis:
    @pytest.fixture
    def aligned_sequences(self):
        """Create a small alignment with known conservation patterns."""
        return ProteinSequences([
            ProteinSequence("AILV-KRDE", id="seq1"),  # Hydrophobic -> Charged
            ProteinSequence("VLIL-EKDR", id="seq2"),  # Hydrophobic -> Charged
            ProteinSequence("LIVI-DERK", id="seq3"),  # Hydrophobic -> Charged
            ProteinSequence("IVL-ARDEK", id="seq4"),  # Hydrophobic -> Charged
        ])

    @pytest.fixture
    def unaligned_sequences(self):
        """Create unaligned sequences to test validation."""
        return ProteinSequences([
            ProteinSequence("AILVKR", id="seq1"),
            ProteinSequence("VLILEKDR", id="seq2"),
        ])

    def test_initialization(self, aligned_sequences, unaligned_sequences):
        """Test initialization with valid and invalid inputs."""
        # Valid initialization
        analyzer = ConservationAnalysis(aligned_sequences)
        assert analyzer.sequences == aligned_sequences
        assert analyzer.ignore_gaps == True

        # Test with unaligned sequences
        with pytest.raises(ValueError, match="Input ProteinSequences must be aligned"):
            ConservationAnalysis(unaligned_sequences)


    def test_compute_conservation(self, aligned_sequences):
        """Test computation of conservation scores."""
        analyzer = ConservationAnalysis(aligned_sequences)
        scores = analyzer.compute_conservation()

        # Check that we got scores for all properties
        assert len(scores) == len(ConservationAnalysis.PROPERTIES)

        # Test that scores are between 0 and 1
        for prop, prop_scores in scores.items():
            assert np.all((prop_scores >= 0) & (prop_scores <= 1))

        # Test specific conservation patterns
        # First 4 positions should be highly hydrophobic
        assert np.mean(scores['Hydrophobic'][:4]) > 0.8
        # Last 4 positions should be highly charged
        assert np.mean(scores['Charged'][-4:]) > 0.8

        # Test gap handling
        assert not np.isnan(scores['Hydrophobic'][4])  # Gap position

    def test_compute_significance(self, aligned_sequences):
        """Test computation of statistical significance."""
        analyzer = ConservationAnalysis(aligned_sequences)
        significant_positions, p_values = analyzer.compute_significance(alpha=0.05)

        # Check output shapes
        assert len(significant_positions) == aligned_sequences.width
        assert all(len(pvals) == aligned_sequences.width for pvals in p_values.values())

        # Check p-values are valid
        for prop, pvals in p_values.items():
            assert np.all((pvals >= 0) & (pvals <= 1))

    def test_compare_alignments(self, aligned_sequences):
        """Test comparison between two alignments."""
        # Create a second alignment with different conservation patterns
        second_alignment = ProteinSequences([
            ProteinSequence("KRDE-AILV", id="seq1"),  # Reversed pattern
            ProteinSequence("EKDR-VLIL", id="seq2"),
            ProteinSequence("DERK-LIVI", id="seq3"),
        ])

        differences, p_values = ConservationAnalysis.compare_alignments(
            aligned_sequences, 
            second_alignment,
            ignore_gaps=True,
            alpha=0.05
        )

        # Check output shapes
        assert all(len(diff) == aligned_sequences.width for diff in differences.values())
        assert all(len(pvals) == aligned_sequences.width for pvals in p_values.values())

        # Test specific difference patterns
        # Hydrophobic conservation should be higher in first alignment at start
        assert np.mean(differences['Hydrophobic'][:4]) < 0.2
        # Charged conservation should be higher in second alignment at start
        assert np.mean(differences['Charged'][:4]) > 0.5

        # Test with mismatched alignments
        mismatched_alignment = ProteinSequences([
            ProteinSequence("KRDEAILV", id="seq1"),  # Different length
        ])
        with pytest.raises(ValueError, match="Alignments must have the same length"):
            ConservationAnalysis.compare_alignments(aligned_sequences, mismatched_alignment)

    def test_gap_handling(self, aligned_sequences):
        """Test how gaps are handled in conservation calculations."""
        # Test with and without gap ignoring
        analyzer_ignore_gaps = ConservationAnalysis(aligned_sequences, ignore_gaps=True)
        analyzer_with_gaps = ConservationAnalysis(aligned_sequences, ignore_gaps=False)

        scores_ignore = analyzer_ignore_gaps.compute_conservation()
        scores_with = analyzer_with_gaps.compute_conservation()

        # Scores should be different at gap positions
        assert not np.allclose(
            scores_ignore['Hydrophobic'][4], 
            scores_with['Hydrophobic'][4]
        )

        # Gap positions should have lower conservation when not ignored
        assert scores_with['Hydrophobic'][4] < scores_ignore['Hydrophobic'][4]

    def test_inverse_properties(self, aligned_sequences):
        """Test that inverse properties behave correctly."""
        analyzer = ConservationAnalysis(aligned_sequences)
        scores = analyzer.compute_conservation()

        # Test that property and its inverse sum to 1 (approximately)
        for prop in ['Hydrophobic', 'Polar', 'Small', 'Charged']:
            prop_sum = scores[prop] + scores[f'not_{prop}']
            assert np.allclose(prop_sum, 1.0)