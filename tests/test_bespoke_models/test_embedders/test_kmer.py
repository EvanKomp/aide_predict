# tests/test_bespoke_models/test_embedders/test_kmer.py
'''
* Author: Evan Komp
* Created: 10/23/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import pytest
import numpy as np
from aide_predict.bespoke_models.embedders.kmer import KmerEmbedding
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence

@pytest.fixture
def simple_sequences():
    """Fixture providing simple protein sequences for testing."""
    seqs = [
        ProteinSequence("ACDEG", id="seq1"),
        ProteinSequence("ACDEF", id="seq2"),
        ProteinSequence("ACDEK", id="seq3")
    ]
    return ProteinSequences(seqs)

@pytest.fixture
def aligned_sequences():
    """Fixture providing aligned protein sequences for testing."""
    seqs = [
        ProteinSequence("ACD-EG", id="seq1"),
        ProteinSequence("ACD-EF", id="seq2"),
        ProteinSequence("ACD-EK", id="seq3")
    ]
    return ProteinSequences(seqs)

@pytest.fixture
def varied_length_sequences():
    """Fixture providing sequences of different lengths."""
    seqs = [
        ProteinSequence("ACDEG", id="seq1"),
        ProteinSequence("ACDEFGH", id="seq2"),
        ProteinSequence("ACDEKIJ", id="seq3")
    ]
    return ProteinSequences(seqs)

class TestKmerEmbedding:
    def test_initialization(self):
        """Test proper initialization of KmerEmbedding."""
        embedder = KmerEmbedding(k=3, normalize=True)
        assert embedder.k == 3
        assert embedder.normalize == True
        assert len(embedder._kmer_to_index) == 0

    def test_fitting_creates_kmer_mapping(self, simple_sequences):
        """Test that fitting creates proper kmer to index mapping."""
        embedder = KmerEmbedding(k=3)
        embedder.fit(simple_sequences)
        
        # Check that all possible 3-mers from sequences are in mapping
        expected_kmers = {'ACD', 'CDE', 'DEG', 'DEF', 'DEK'}
        assert set(embedder._kmer_to_index.keys()) == expected_kmers
        assert embedder.n_features_ == len(expected_kmers)
        assert embedder.fitted_

    def test_transform_shape(self, simple_sequences):
        """Test that transform returns correct shape."""
        embedder = KmerEmbedding(k=3)
        embedder.fit(simple_sequences)
        embeddings = embedder.transform(simple_sequences)
        
        assert embeddings.shape == (len(simple_sequences), embedder.n_features_)
        assert embeddings.dtype == np.float32

    def test_normalization(self, simple_sequences):
        """Test that normalization works correctly."""
        embedder = KmerEmbedding(k=3, normalize=True)
        embedder.fit(simple_sequences)
        embeddings = embedder.transform(simple_sequences)
        
        # Check that each row sums to 1
        row_sums = np.sum(embeddings, axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_no_normalization(self, simple_sequences):
        """Test without normalization."""
        embedder = KmerEmbedding(k=3, normalize=False)
        embedder.fit(simple_sequences)
        embeddings = embedder.transform(simple_sequences)
        
        # Each kmer should appear exactly once in each sequence
        assert np.all(embeddings <= 1)
        assert np.all(embeddings >= 0)

    def test_aligned_sequences(self, aligned_sequences):
        """Test handling of aligned sequences with gaps."""
        embedder = KmerEmbedding(k=3)
        embedder.fit(aligned_sequences)
        embeddings = embedder.transform(aligned_sequences)
        
        # Should produce same results as unaligned sequences
        expected_kmers = {'ACD', 'CDE', 'DEG', 'DEF', 'DEK'}
        assert set(embedder._kmer_to_index.keys()) == expected_kmers

    def test_varied_length_sequences(self, varied_length_sequences):
        """Test handling of sequences with different lengths."""
        embedder = KmerEmbedding(k=3)
        embedder.fit(varied_length_sequences)
        embeddings = embedder.transform(varied_length_sequences)
        
        # Should handle different lengths properly
        assert embeddings.shape[0] == len(varied_length_sequences)
        assert all(np.sum(embeddings[i]) > 0 for i in range(len(varied_length_sequences)))

    def test_feature_names(self, simple_sequences):
        """Test that feature names are generated correctly."""
        embedder = KmerEmbedding(k=3)
        embedder.fit(simple_sequences)
        feature_names = embedder.get_feature_names_out()
        
        assert len(feature_names) == embedder.n_features_
        assert all(name.startswith("kmer_") for name in feature_names)
        assert set(name[5:] for name in feature_names) == set(embedder._kmer_to_index.keys())

    def test_transform_new_sequences(self, simple_sequences):
        """Test transforming sequences not seen during fitting."""
        embedder = KmerEmbedding(k=3)
        embedder.fit(simple_sequences)
        
        new_seqs = ProteinSequences([ProteinSequence("ACDEM", id="new_seq")])
        embeddings = embedder.transform(new_seqs)
        
        assert embeddings.shape == (1, embedder.n_features_)

    def test_invalid_k(self):
        """Test handling of invalid k values."""
        with pytest.raises(ValueError):
            KmerEmbedding(k=0)
        with pytest.raises(ValueError):
            KmerEmbedding(k=-1)

    def test_empty_sequences(self):
        """Test handling of empty sequence list."""
        embedder = KmerEmbedding(k=3)
        empty_seqs = ProteinSequences([])
        
        with pytest.raises(ValueError):
            embedder.fit(empty_seqs)

    def test_sequence_shorter_than_k(self):
        """Test handling of sequences shorter than k."""
        embedder = KmerEmbedding(k=5)
        short_seqs = ProteinSequences([ProteinSequence("ACD", id="short")])
        
        with pytest.raises(ValueError):
            embedder.fit(short_seqs)

    def test_transform_before_fit(self, simple_sequences):
        """Test that transform raises error if called before fit."""
        embedder = KmerEmbedding(k=3)
        with pytest.raises(ValueError):
            embedder.transform(simple_sequences)
