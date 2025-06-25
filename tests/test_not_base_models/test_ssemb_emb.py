import pytest
import numpy as np
import os
from pathlib import Path
import tempfile
import time

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence, ProteinStructure
from aide_predict.bespoke_models.embedders.ssemb import SSEmbEmbedding

class TestSSEmbEmbedding:
    """Test the SSEmbEmbedding class with actual external calls."""
    
    @pytest.fixture
    def data_dir(self):
        """Path to test data directory"""
        return Path("tests/data")
    
    @pytest.fixture
    def envz_sequence(self, data_dir):
        """Create an ENVZ sequence with its structure and MSA"""
        pdb_path = data_dir / "ENVZ_ECOLI.pdb"
        msa_path = data_dir / "ENVZ_ECOLI_extreme_filtered.a2m"

        
        # Create sequence from structure
        seq = ProteinSequence.from_pdb(str(pdb_path), id="ENVZ_ECOLI")
        assert len(seq) == 60, f"ENVZ sequence should be 60 amino acids long, got {len(seq)}"
        
        # Add MSA
        msa = ProteinSequences.from_a3m(str(msa_path))
        
        # Confirm that seq is the first sequence in the MSA
        if str(seq) != str(msa[0]):
            # If not, create a new MSA with seq as the first sequence
            new_msa = [seq] + [s for s in msa if s.id != seq.id]
            msa = ProteinSequences(new_msa)
            
        seq.msa = msa
        
        return seq
    
    @pytest.fixture
    def gfp_sequence(self, data_dir):
        """Create a GFP sequence with its structure and MSA"""
        pdb_path = data_dir / "GFP_AEQVI.pdb"
        msa_path = data_dir / "GFP_AEQVI_full_04-29-2022_b08.a2m"
        
        # Create sequence from structure
        seq = ProteinSequence.from_pdb(str(pdb_path), id="GFP_AEQVI")
        
        # Add MSA
        msa = ProteinSequences.from_a3m(str(msa_path))
        
        # Confirm that seq is the first sequence in the MSA
        if str(seq) != str(msa[0]):
            # If not, create a new MSA with seq as the first sequence
            new_msa = [seq] + [s for s in msa if s.id != seq.id]
            msa = ProteinSequences(new_msa)
            
        seq.msa = msa
        
        return seq
    
    @pytest.fixture
    def sample_sequences(self, envz_sequence, gfp_sequence):
        """Create a set of test sequences each with its own MSA and structure"""
        return ProteinSequences([envz_sequence, gfp_sequence])

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def embedder(self, temp_dir):
        """Create an SSEmbEmbedding instance"""
        return SSEmbEmbedding(
            metadata_folder='./tmp/ssemb_emb',
            flatten=False,
            pool=True,  # Use pooling by default to handle different length sequences
            gpu_id=0,
            use_cache=False  # Enable caching for efficiency
        )

    def test_fit(self, embedder):
        """Test that the embedder fits correctly"""
        fitted_embedder = embedder.fit()  # No input needed for ExpectsNoFitMixin
        assert fitted_embedder is embedder
        assert hasattr(fitted_embedder, 'fitted_')
        assert fitted_embedder.fitted_ is True

    @pytest.mark.skipif(not os.environ.get('SSEMB_CONDA_ENV') or not os.environ.get('SSEMB_REPO'), 
                        reason="SSEmb environment variables not set")
    def test_transform_with_pooling(self, embedder, envz_sequence):
        """Test transformation with pooling enabled using a single sequence"""
        # First fit the embedder
        embedder.fit()
        
        # Transform the sequence
        embeddings = embedder.transform(ProteinSequences([envz_sequence]))
        
        # Check the shape and properties of the embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 256)  # 1 sequence, 256-dim embeddings (pooled)
        assert not np.isnan(embeddings).any(), "Embeddings contain NaN values"
        assert not np.isinf(embeddings).any(), "Embeddings contain infinite values"
        
        # Check that values are reasonable (not all zeros or ones)
        assert not np.allclose(embeddings, 0), "Embeddings are all close to zero"
        assert not np.allclose(embeddings, 1), "Embeddings are all close to one"

    @pytest.mark.skipif(not os.environ.get('SSEMB_CONDA_ENV') or not os.environ.get('SSEMB_REPO'), 
                        reason="SSEmb environment variables not set")
    def test_transform_with_positions(self, embedder, envz_sequence, temp_dir):
        """Test transformation with specific positions"""
        # Create embedder with specific positions
        pos_embedder = SSEmbEmbedding(
            metadata_folder='./tmp/ssemb_emb',
            positions=[0, 1, 2],  # First three positions
            pool=False,
            gpu_id=0,
            use_cache=False
        )
        
        pos_embedder.fit()
        embeddings = pos_embedder.transform(ProteinSequences([envz_sequence]))
        
        # Check position-specific embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 3, 256)  # 1 sequence, 3 positions, 256-dim embeddings
        assert not np.isnan(embeddings).any(), "Embeddings contain NaN values"
        
        # Verify that specific positions were extracted correctly
        # Each position should have a unique embedding pattern
        assert not np.allclose(embeddings[0, 0], embeddings[0, 1]), "Position 0 and 1 embeddings are identical"
        assert not np.allclose(embeddings[0, 1], embeddings[0, 2]), "Position 1 and 2 embeddings are identical"

    @pytest.mark.skipif(not os.environ.get('SSEMB_CONDA_ENV') or not os.environ.get('SSEMB_REPO'), 
                        reason="SSEmb environment variables not set")
    def test_transform_with_flattening(self, embedder, envz_sequence, temp_dir):
        """Test transformation with flattening and positions"""
        # Create embedder with flattening and positions
        flat_embedder = SSEmbEmbedding(
            metadata_folder='./tmp/ssemb_emb',
            positions=[0, 1, 2],  # First three positions
            flatten=True,
            pool=False,
            gpu_id=0,
            use_cache=False
        )
        
        flat_embedder.fit()
        embeddings = flat_embedder.transform(ProteinSequences([envz_sequence]))
        
        # Check flattened embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (1, 3 * 256)  # 1 sequence, 3 positions * 256-dim embeddings (flattened)
        assert not np.isnan(embeddings).any(), "Embeddings contain NaN values"

    @pytest.mark.skipif(not os.environ.get('SSEMB_CONDA_ENV') or not os.environ.get('SSEMB_REPO'), 
                        reason="SSEmb environment variables not set")
    def test_transform_multiple_sequences(self, embedder, sample_sequences):
        """Test transformation of multiple sequences with pooling"""
        # First fit the embedder
        embedder.fit()
        
        # Transform the sequences
        embeddings = embedder.transform(sample_sequences)
        
        # Check the shape and properties of the pooled embeddings
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 256)  # 2 sequences, 256-dim embeddings (pooled)
        assert not np.isnan(embeddings).any(), "Embeddings contain NaN values"
        
        # Check that the embeddings for different proteins are different
        assert not np.allclose(embeddings[0], embeddings[1]), "Different protein embeddings are identical"

    def test_missing_structure(self, embedder, envz_sequence):
        """Test handling of sequences without structure"""
        # Create a sequence with MSA but no structure
        seq_without_structure = ProteinSequence(str(envz_sequence), id="no_structure")
        seq_without_structure.msa = envz_sequence.msa
        
        invalid_sequences = ProteinSequences([seq_without_structure])
        
        # First fit the embedder
        embedder.fit()
        
        # Expect an error about missing structure
        with pytest.raises(ValueError):
            embedder.transform(invalid_sequences)

    def test_missing_msa(self, embedder, envz_sequence):
        """Test handling of sequences without MSA"""
        # Create a sequence with structure but no MSA
        seq_without_msa = ProteinSequence.from_pdb(
            str(envz_sequence.structure.pdb_file), 
            id="no_msa"
        )
        
        invalid_sequences = ProteinSequences([seq_without_msa])
        
        # First fit the embedder
        embedder.fit()
        
        # Expect an error about missing MSA
        with pytest.raises(ValueError):
            embedder.transform(invalid_sequences)

    def test_get_feature_names_out(self, embedder):
        """Test feature naming with pooling"""
        embedder.fit()
        feature_names = embedder.get_feature_names_out()
        
        assert len(feature_names) == 256  # 256-dim embeddings
        assert all(name.startswith("SSEmb_emb") for name in feature_names)

    def test_get_feature_names_out_with_positions(self, embedder):
        """Test feature naming with positions"""
        embedder.pool = False
        embedder.positions = [0, 1, 2]
        embedder.fit()
        feature_names = embedder.get_feature_names_out()
        
        assert len(feature_names) == 3  # 3 positions
        assert all(name.startswith("pos") for name in feature_names)

    def test_get_feature_names_out_flattened(self, embedder):
        """Test feature naming with flattening"""
        embedder.flatten = True
        embedder.pool = False
        embedder.positions = [0, 1, 2]
        embedder.fit()
        feature_names = embedder.get_feature_names_out()
        
        assert len(feature_names) == 3 * 256  # 3 positions * 256-dim embeddings
        assert all(name.startswith("pos") for name in feature_names)


