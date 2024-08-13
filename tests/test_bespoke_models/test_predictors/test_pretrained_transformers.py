# tests/test_bespoke_models/test_predictors/test_pretrained_trasnformers.py
'''
* Author: Evan Komp
* Created: 7/11/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''

import pytest
import numpy as np
from typing import List
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.bespoke_models.predictors.pretrained_transformers import LikelihoodTransformerBase, MarginalMethod, model_device_context

class MinimalLikelihoodTransformer(LikelihoodTransformerBase):
    def _compute_log_likelihoods(self, X: ProteinSequences, mask_positions: List[List[int]] = None) -> List[np.ndarray]:
        # Simple mock implementation
        return [np.log(np.ones((len(seq), 20)) / 20) for seq in X]

    def _index_log_probs(self, log_probs: np.ndarray, sequences: ProteinSequences) -> np.ndarray:
        # Simple mock implementation
        return np.vstack([log_probs[np.arange(len(seq)), 0].reshape(1,-1) for seq in sequences])

    def _load_model(self):
        # Mock implementation
        self.model_ = "MockModel"

    def _cleanup_model(self):
        # Mock implementation
        del self.model_

class TestLikelihoodTransformerBase:
    @pytest.fixture
    def transformer(self):
        return MinimalLikelihoodTransformer(metadata_folder="test_folder", wt="ACDEFGHIKLMNPQRSTVWY")

    @pytest.fixture
    def sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDEFGHIGLMNPQRSTVWY"),
            ProteinSequence("ACDEAGHIKLMNPQRSTVWY"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY")
        ])

    def test_initialization(self, transformer):
        assert transformer.marginal_method == MarginalMethod.WILDTYPE
        assert transformer.batch_size == 2
        assert transformer.device == 'cpu'

    def test_validate_log_probs(self, transformer, sequences):
        valid_log_probs = [np.log(np.ones((20, 20)) / 20) for _ in range(3)]
        transformer._validate_log_probs(valid_log_probs, sequences)

        with pytest.raises(ValueError):
            invalid_log_probs = [np.ones((20, 20)) for _ in range(3)]
            transformer._validate_log_probs(invalid_log_probs, sequences)

    def test_validate_marginals(self, transformer, sequences):
        valid_marginals = [np.zeros((1, 20)) for _ in range(3)]
        transformer._validate_marginals(valid_marginals, sequences)

        with pytest.raises(ValueError):
            invalid_marginals = [np.zeros((1, 21)) for _ in range(3)]
            transformer._validate_marginals(invalid_marginals, sequences)

    @pytest.mark.parametrize("marginal_method", [
        MarginalMethod.MUTANT.value,
        MarginalMethod.WILDTYPE.value,
        MarginalMethod.MASKED.value
    ])
    def test_transform(self, transformer, sequences, marginal_method):
        transformer.marginal_method = marginal_method
        result = transformer._transform(sequences)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1)  # Assuming pooling is True by default

    def test_compute_mutant_marginal(self, transformer, sequences):
        with model_device_context(transformer, transformer._load_model, transformer._cleanup_model, transformer.device):
            result = transformer._compute_mutant_marginal(sequences)
        assert len(result) == 3
        assert all(r.shape == (1, 20) for r in result)

    def test_compute_wildtype_marginal(self, transformer, sequences):
        with model_device_context(transformer, transformer._load_model, transformer._cleanup_model, transformer.device):
            result = transformer._compute_wildtype_marginal(sequences)
        assert len(result) == 3
        assert all(r.shape == (1, 20) for r in result)

    def test_compute_masked_marginal(self, transformer, sequences):
        with model_device_context(transformer, transformer._load_model, transformer._cleanup_model, transformer.device):
            result = transformer._compute_masked_marginal(sequences)
        assert len(result) == 3
        assert all(r.shape == (1, 20) for r in result)

    def test_post_process_likelihoods(self, transformer, sequences):
        log_likelihoods = [np.zeros((1, 20)) for _ in range(3)]
        result = transformer._post_process_likelihoods(log_likelihoods, sequences)
        assert result.shape == (3, 1)

    def test_get_feature_names_out(self, transformer):
        transformer.model_ = True  # Mock the fitted model
        feature_names = transformer.get_feature_names_out()
        assert len(feature_names) == 1
        assert feature_names[0] == "MinimalLikelihoodTransformer_log_likelihood"

    def test_variable_length_sequences(self, transformer):
        var_sequences = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY"),
            ProteinSequence("ACDEFGHIKLMNPQRSTV"),
            ProteinSequence("ACDEFGHIKLMNPQRSTVWYGG")
        ])
        transformer.marginal_method = MarginalMethod.MUTANT.value
        with pytest.raises(ValueError):
            transformer._compute_wildtype_marginal(var_sequences)
        with pytest.raises(ValueError):
            transformer._compute_masked_marginal(var_sequences)
        
        # Mutant marginal should work with variable length sequences
        with model_device_context(transformer, transformer._load_model, transformer._cleanup_model, transformer.device):
            result = transformer._compute_mutant_marginal(var_sequences)
        assert len(result) == 3
        assert result[0].shape == (1, 20)
        assert result[1].shape == (1, 18)
        assert result[2].shape == (1, 22)

    def test_positions_filtering(self, transformer, sequences):
        transformer.positions = [0, 5, 10]
        transformer.pool = False
        log_likelihoods = [np.zeros((1, 20)) for _ in range(3)]
        result = transformer._post_process_likelihoods(log_likelihoods, sequences)
        assert result.shape == (3, 3)  # 3 sequences, 3 positions

    @pytest.mark.parametrize("pool", [True, False])
    def test_pooling(self, transformer, sequences, pool):
        transformer.pool = pool
        log_likelihoods = [np.zeros((1, 20)) for _ in range(3)]
        result = transformer._post_process_likelihoods(log_likelihoods, sequences)
        if pool:
            assert result.shape == (3, 1)
        else:
            assert result.shape == (3, 20)

    def test_load_and_cleanup_model(self, transformer):
        transformer._load_model()
        assert hasattr(transformer, 'model_')
        assert transformer.model_ == "MockModel"
        
        transformer._cleanup_model()
        assert not hasattr(transformer, 'model_')

    def test_model_device_context(self, transformer):
        with model_device_context(transformer, transformer._load_model, transformer._cleanup_model, transformer.device):
            assert hasattr(transformer, 'model_')
            assert transformer.model_ == "MockModel"
        
        assert not hasattr(transformer, 'model_')