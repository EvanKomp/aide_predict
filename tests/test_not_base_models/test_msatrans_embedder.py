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
    def sample_sequence(self):
        """Create a sequence with its own MSA"""
        seq = ProteinSequence("MKALVTGAGRGIGT", id="test_seq")
        msa = ProteinSequences([
            ProteinSequence("MKALVTGAGRGIGT", id="test_seq"),
            ProteinSequence("LRALVTGAG-GIGT", id="homolog1"),
            ProteinSequence("MKALV-GAGRGIGT", id="homolog2")
        ])
        seq.msa = msa
        return seq
    
    @pytest.fixture
    def sample_sequences(self, sample_sequence):
        """Create a set of test sequences each with its own MSA"""
        seq1 = sample_sequence
        
        # Create second sequence with its own MSA
        seq2 = ProteinSequence("LKALVTGAGRGIGT", id="test_seq2")
        msa2 = ProteinSequences([
            ProteinSequence("LKALVTGAGRGIGT", id="test_seq2"),
            ProteinSequence("LKALVTGAG-GIGT", id="homolog1"),
            ProteinSequence("LKALV-GAGRGIGT", id="homolog2")
        ])
        seq2.msa = msa2
        
        return ProteinSequences([seq1, seq2])

    @pytest.fixture
    def embedder(self, tmp_path):
        return MSATransformerEmbedding(
            metadata_folder=str(tmp_path),
            layer=-1,
            positions=[0, 1, 2],
            flatten=False,
            pool=False,
            batch_size=2,
            device='cpu',
            n_msa_seqs=3,
            use_cache=False
        )

    def test_fit(self, embedder, sample_sequences):
        fitted_embedder = embedder.fit(sample_sequences)
        assert fitted_embedder is embedder
        assert hasattr(fitted_embedder, 'fitted_')
        assert hasattr(fitted_embedder, 'embedding_dim_')
        assert fitted_embedder.fitted_ == True
        assert fitted_embedder.embedding_dim_ == 768

    def test_transform(self, embedder, sample_sequences):
        embedder.fit(sample_sequences)
        embeddings = embedder.transform(sample_sequences)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3, 768)  # 2 sequences, 3 positions, 768-dim embeddings

    def test_transform_with_pooling(self, embedder, sample_sequences):
        embedder.pool = True
        embedder.fit(sample_sequences)
        embeddings = embedder.transform(sample_sequences)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)  # 2 sequences, 768-dim embeddings

    def test_transform_with_flattening(self, embedder, sample_sequences):
        embedder.flatten = True
        embedder.fit(sample_sequences)
        embeddings = embedder.transform(sample_sequences)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3 * 768)  # 2 sequences, 3 positions * 768-dim embeddings

    def test_missing_msa(self, embedder, sample_sequences):
        embedder.fit(sample_sequences)
        # Create a sequence without MSA
        seq_without_msa = ProteinSequence("MKALVTGAGRGIGT", id="no_msa")
        invalid_sequences = ProteinSequences([seq_without_msa])
        with pytest.raises(ValueError):
            embedder.transform(invalid_sequences)

    def test_mismatched_msa_width(self, embedder, sample_sequences):
        embedder.fit(sample_sequences)
        # Create a sequence with mismatched MSA width
        seq_with_bad_msa = ProteinSequence("MKALVTGAGRGIGT", id="bad_msa")
        msa = ProteinSequences([
            ProteinSequence("MKALVTGAGRGT", id="bad_msa"),
            ProteinSequence("LRALVTGAGIGT", id="homolog1"),  # Too short
        ])
        seq_with_bad_msa.msa = msa
        invalid_sequences = ProteinSequences([seq_with_bad_msa])
        with pytest.raises(ValueError):
            embedder.transform(invalid_sequences)

    def test_get_feature_names_out(self, embedder, sample_sequences):
        embedder.fit(sample_sequences)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 3  # 3 positions
        assert all(name.startswith("pos") for name in feature_names)

    def test_get_feature_names_out_pooled(self, embedder, sample_sequences):
        embedder.pool = True
        embedder.fit(sample_sequences)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 768  # 768-dim embeddings
        assert all(name.startswith("MSA_emb") for name in feature_names)

    def test_get_feature_names_out_flattened(self, embedder, sample_sequences):
        embedder.flatten = True
        embedder.fit(sample_sequences)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 3 * 768  # 3 positions * 768-dim embeddings
        assert all(name.startswith("pos") for name in feature_names)
    
    def test_msa_caching(self, embedder, sample_sequences, monkeypatch):
        """Test that MSAs are cached and reused"""
        # Track actual MSA sampling operations (not just method calls)
        sampling_counter = 0
        original_sample_method = np.random.seed
        
        def counting_sample(seed):
            nonlocal sampling_counter
            sampling_counter += 1
            return original_sample_method(seed)
            
        # Replace np.random.seed to track when actual sampling happens
        monkeypatch.setattr(np.random, 'seed', counting_sample)
        
        embedder.fit(sample_sequences)
        embedder._msa_cache = {}  # Clear any cache from init
        
        # First transform should perform sampling twice (once per sequence)
        embedder.transform(sample_sequences)
        assert sampling_counter == 2
        assert len(embedder._msa_cache) == 2  # Two MSAs should be cached
        
        # Reset counter
        sampling_counter = 0
        
        # Second transform should use cached MSAs without new sampling
        embedder.transform(sample_sequences)
        assert sampling_counter == 0  # No new sampling operations
        assert len(embedder._msa_cache) == 2  # Cache size unchanged