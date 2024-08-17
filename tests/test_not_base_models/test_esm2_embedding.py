# tests/test_not_base_models/test_esm2_embedding.py
'''
* Author: Evan Komp
* Created: 7/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.bespoke_models.embedders.esm2 import ESM2Embedding

class TestESM2Embedding:

    @pytest.fixture(scope="class")
    def embedder(self):
        return ESM2Embedding(
            model_checkpoint="esm2_t6_8M_UR50D",  # Using a small model for faster tests
            layer=-1,
            batch_size=2
        )

    @pytest.fixture(scope="class")
    def sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWY"),
            ProteinSequence("LLLLLLLLLLLLLLLLLLLL")
        ])

    @pytest.fixture(scope="class")
    def aligned_sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDE-FGHIKLMNPQRSTVWY"),
            ProteinSequence("LLLL-LLLLLLLLLLLLLLLL")
        ])

    def test_initialization(self, embedder):
        assert embedder.model_checkpoint == 'esm2_t6_8M_UR50D'
        assert embedder.layer == -1
        assert embedder.positions is None
        assert not embedder.flatten
        assert not embedder.pool
        assert embedder.batch_size == 2

    def test_fit(self, embedder, sequences):
        embedder.fit(sequences)
        assert hasattr(embedder, 'fitted_')

    @pytest.mark.parametrize("positions,pool,flatten", [
        (None, False, False),
        ([0, 1, 2], False, False),
        (None, True, False),
        (None, False, True),
    ])
    def test_transform(self, embedder, sequences, positions, pool, flatten):
        embedder.positions = positions
        embedder.pool = pool
        embedder.flatten = flatten
        embedder.fit([])
        print(embedder.flatten)
        embeddings = embedder.transform(sequences)
        
        assert isinstance(embeddings, np.ndarray)
        if pool:
            assert embeddings.shape == (2, 320)  # ESM2 t6 8M model has 320 hidden dimensions
        elif positions:
            assert embeddings.shape == (2, len(positions), 320)
        else:
            if not flatten:
                assert embeddings.shape == (2, 20, 320)  # 20 amino acids in each sequence
            else:
                assert embeddings.shape == (2, 6400)

    def test_transform_aligned(self, embedder, aligned_sequences):
        embedder.positions = [0,4]
        embedder.pool = False
        embedder.flatten = False
        embedder.fit([])
        embeddings = embedder.transform(aligned_sequences)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 2, 320)  # 21 positions including the gap
        # check for zeros at the gap
        assert np.all(embeddings[:, 1] == 0)

    def test_transform_variable_length_error(self, embedder):
        sequences = ProteinSequences([ProteinSequence("ACGT"), ProteinSequence("ACGTAA")])
        embedder.positions = [0, 1]
        
        with pytest.raises(ValueError):
            embedder.transform(sequences)

    def test_get_feature_names_out_pooled(self, embedder, sequences):
        
        embedder.positions = None
        embedder.pool = True
        embedder.flatten = False
        embedder.fit(sequences)
        
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 320  # ESM2 t6 8M model has 320 hidden dimensions

    def test_get_feature_names_out_flattened(self, embedder, sequences):
        embedder.positions = [0, 1]
        embedder.pool = False
        embedder.flatten = True
        embedder.fit(sequences)
        
        
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 2 * 320
