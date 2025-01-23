# tests/test_not_base_models/test_badass.py
'''
* Author: Evan Komp
* Created: 1/23/2025
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import pytest
import numpy as np
import pandas as pd
from aide_predict.utils.badass import BADASSOptimizer, BADASSOptimizerParams
from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences
from unittest.mock import Mock, patch

class TestBADASSOptimizerParams:
    def test_default_initialization(self):
        params = BADASSOptimizerParams()
        assert params.seqs_per_iter == 500
        assert params.num_iter == 200
        assert params.sites_to_ignore == []  # Tests empty list conversion
        assert params.temperature == 1.5

    def test_custom_initialization(self):
        params = BADASSOptimizerParams(
            seqs_per_iter=100,
            num_iter=50,
            sites_to_ignore=[1, 2, 3],
            temperature=2.0
        )
        assert params.seqs_per_iter == 100
        assert params.num_iter == 50
        assert params.sites_to_ignore == [1, 2, 3]
        assert params.temperature == 2.0

    def test_to_dict(self):
        params = BADASSOptimizerParams(seqs_per_iter=100, temperature=2.0)
        param_dict = params.to_dict()
        assert param_dict['seqs_per_iter'] == 100
        assert param_dict['T'] == 2.0  # Tests correct key conversion
        assert 'sites_to_ignore' in param_dict
        assert isinstance(param_dict, dict)

class TestBADASSOptimizer:
    @pytest.fixture
    def reference_sequence(self):
        return ProteinSequence("MKLLVLGLPGAGKGT", id="wild_type")
    
    @pytest.fixture
    def mock_predictor(self):
        def predict(sequences):
            # Return predictable values instead of random
            return np.array([0.5] * len(sequences))
        return Mock(predict=Mock(side_effect=predict))

    @pytest.fixture
    def optimizer_params(self):
        return BADASSOptimizerParams(
            seqs_per_iter=10,
            num_iter=2,
            init_score_batch_size=5
        )

    @pytest.fixture
    def optimizer(self, mock_predictor, reference_sequence, optimizer_params):
        return BADASSOptimizer(
            predictor=mock_predictor,
            reference_sequence=reference_sequence,
            params=optimizer_params
        )

    def test_initialization(self, optimizer, reference_sequence, optimizer_params):
        assert isinstance(optimizer.reference_sequence, ProteinSequence)
        assert str(optimizer.reference_sequence) == str(reference_sequence)
        assert optimizer.params == optimizer_params
        assert hasattr(optimizer, '_optimizer')

    def test_wrapped_predictor(self, optimizer, mock_predictor):
        test_sequences = ["MKLLVLGLPGAGKGT", "AKLLVLGLPGAGKGT"]
        scores = optimizer._wrapped_predictor(test_sequences)
        assert isinstance(scores, list)
        assert len(scores) == len(test_sequences)
        assert all(s == 0.5 for s in scores)  # Check for expected values
        # Don't test number of calls since BADASS uses it internally

    @patch('aide_predict.utils.badass.GeneralProteinOptimizer')
    def test_optimize(self, mock_general_optimizer_class, optimizer):
        # Create mock instance
        mock_optimizer = Mock()
        mock_general_optimizer_class.return_value = mock_optimizer
        
        # Mock the optimization results
        mock_results = pd.DataFrame({
            'sequences': ['M1A-K2R', 'M1V'],
            'scores': [0.5, 0.6]
        })
        mock_stats = pd.DataFrame({
            'iteration': [1, 2],
            'mean_score': [0.5, 0.6]
        })
        mock_optimizer.optimize.return_value = (mock_results, mock_stats)
        
        # Replace optimizer's _optimizer with our mock
        optimizer._optimizer = mock_optimizer
        
        results_df, stats_df = optimizer.optimize()
        
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(stats_df, pd.DataFrame)
        assert 'full_sequence' in results_df.columns
        assert all(isinstance(seq, ProteinSequence) for seq in results_df['full_sequence'])

    def test_mutations_to_sequence(self, optimizer):
        # Test single mutation
        mut_seq = optimizer._mutations_to_sequence("M1A")
        assert mut_seq == "AKLLVLGLPGAGKGT"
        
        # Test multiple mutations
        mut_seq = optimizer._mutations_to_sequence("M1A-K2R")
        assert mut_seq == "ARLLVLGLPGAGKGT"
        
        # Test empty mutations
        mut_seq = optimizer._mutations_to_sequence("")
        assert mut_seq == str(optimizer.reference_sequence)

    def test_results_property(self, optimizer):
        # Test before optimization (no df attribute)
        assert optimizer.results == (None, None)
        
        # Test after optimization (mock df attributes)
        mock_df = pd.DataFrame({'test': [1]})
        mock_df_stats = pd.DataFrame({'stats': [1]})
        optimizer._optimizer.df = mock_df
        optimizer._optimizer.df_stats = mock_df_stats
        
        results, stats = optimizer.results
        assert results is mock_df
        assert stats is mock_df_stats

    def test_plot_and_save_methods(self, optimizer):
        # Create mock optimizer with required methods
        mock_optimizer = Mock()
        mock_optimizer.plot_scores = Mock()
        mock_optimizer.save_results = Mock()
        
        # Replace the real optimizer with our mock
        optimizer._optimizer = mock_optimizer
        
        # Test plot method
        optimizer.plot(save_figs=True)
        mock_optimizer.plot_scores.assert_called_once_with(save_figs=True)
        
        # Test save_results method 
        optimizer.save_results("test_file")
        mock_optimizer.save_results.assert_called_once_with(filename="test_file")
