# tests/test_not_base_models/test_msatrans_embedder.py
'''
* Author: Evan Komp
* Created: 7/8/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import pytest
import numpy as np
import torch
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.bespoke_models.embedders.msa_transformer import MSATransformerEmbedding

class TestMSATransformerEmbedding:
    @pytest.fixture
    def msa(self):
        return ProteinSequences([
            ProteinSequence("MKALVTGAGRGIGT"),
            ProteinSequence("LRALVTGAG-GIGT"),
            ProteinSequence("MKALV-GAGRGIGT")
        ])

    @pytest.fixture
    def embedder(self, tmp_path):
        return MSATransformerEmbedding(
            metadata_folder=str(tmp_path),
            layer=-1,
            positions=[0, 1, 2],
            flatten=False,
            pool=False,
            batch_size=2,
            device='cpu'
        )

    def test_fit(self, embedder, msa):
        fitted_embedder = embedder.fit(msa)
        assert fitted_embedder is embedder
        assert hasattr(fitted_embedder, 'model_')
        assert hasattr(fitted_embedder, 'alphabet_')
        assert fitted_embedder.msa_length_ == 14
        assert isinstance(fitted_embedder.original_msa_, ProteinSequences)

    def test_transform(self, embedder, msa):
        embedder.fit(msa)
        new_sequences = ProteinSequences([
            ProteinSequence("MKALVTGAGRGIGT"),
            ProteinSequence("LKALVTGAGRGIGT")
        ])
        embeddings = embedder.transform(new_sequences)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3, 768)  # 2 sequences, 3 positions, 768-dim embeddings

    def test_transform_with_pooling(self, embedder, msa):
        embedder.pool = True
        embedder.fit(msa)
        new_sequences = ProteinSequences([
            ProteinSequence("MKALVTGAGAGIGT"),
            ProteinSequence("LKALVTGAGAGIGT")
        ])
        embeddings = embedder.transform(new_sequences)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)  # 2 sequences, 768-dim embeddings

    def test_transform_with_flattening(self, embedder, msa):
        embedder.flatten = True
        embedder.fit(msa)
        new_sequences = ProteinSequences([
            ProteinSequence("MKALVTGAGAGIGT"),
            ProteinSequence("LKALVTGAGAGIGT")
        ])
        embeddings = embedder.transform(new_sequences)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3 * 768)  # 2 sequences, 3 positions * 768-dim embeddings

    def test_invalid_sequence_length(self, embedder, msa):
        embedder.fit(msa)
        invalid_sequences = ProteinSequences([
            ProteinSequence("MKALVTGAGAGIG"),  # One character short
            ProteinSequence("LKALVTGAGAGIGTT")  # One character long
        ])
        with pytest.raises(ValueError):
            embedder.transform(invalid_sequences)

    def test_get_feature_names_out(self, embedder, msa):
        embedder.fit(msa)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 3  # 3 positions
        assert all(name.startswith("pos") for name in feature_names)

    def test_get_feature_names_out_pooled(self, embedder, msa):
        embedder.pool = True
        embedder.fit(msa)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 768  # 768-dim embeddings
        assert all(name.startswith("MSA_emb") for name in feature_names)

    def test_get_feature_names_out_flattened(self, embedder, msa):
        embedder.flatten = True
        embedder.fit(msa)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 3 * 768  # 3 positions * 768-dim embeddings
        assert all(name.startswith("pos") for name in feature_names)