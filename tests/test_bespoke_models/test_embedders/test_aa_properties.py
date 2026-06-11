# tests/test_bespoke_models/test_embedders/test_aa_properties.py
'''
* Author: Evan Komp
* Created: 2026-06-10
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Unit tests for AAPropertiesEmbedding. Depends only on `aaindex` (a base
dependency in environment.yaml), so these run under default CI.
'''
import numpy as np
import pytest

from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.bespoke_models.embedders.aa_properties import (
    AAPropertiesEmbedding,
    _build_aa_property_lookup,
    DEFAULT_AAINDEX_PROPERTIES,
)

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


class TestBuildAAPropertyLookup:
    def test_lookup_contents(self):
        lookup, names = _build_aa_property_lookup(DEFAULT_AAINDEX_PROPERTIES)
        # All 20 canonical amino acids present.
        assert set(lookup.keys()) == set(CANONICAL)
        # 9 default properties + the aromatic boolean.
        assert len(names) == len(DEFAULT_AAINDEX_PROPERTIES) + 1
        assert names[-1] == "aromatic"
        # Each vector matches the number of property names.
        for aa in CANONICAL:
            assert lookup[aa].shape == (len(names),)
            assert lookup[aa].dtype == np.float32
        # Aromatic flag: F/W/Y -> 1.0, others -> 0.0.
        assert lookup["F"][-1] == 1.0
        assert lookup["W"][-1] == 1.0
        assert lookup["Y"][-1] == 1.0
        assert lookup["A"][-1] == 0.0

    def test_lookup_without_aromatic(self):
        lookup, names = _build_aa_property_lookup(
            DEFAULT_AAINDEX_PROPERTIES, include_aromatic=False
        )
        assert len(names) == len(DEFAULT_AAINDEX_PROPERTIES)
        assert "aromatic" not in names
        assert lookup["A"].shape == (len(DEFAULT_AAINDEX_PROPERTIES),)


class TestAAPropertiesEmbedding:
    @pytest.fixture
    def sample_sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDEF"),
            ProteinSequence("GHIKL"),
            ProteinSequence("MNPQR"),
        ])

    def test_initialization_defaults(self, tmp_path):
        emb = AAPropertiesEmbedding(metadata_folder=str(tmp_path))
        assert emb.include_aromatic is True
        assert emb.aaindex_properties == DEFAULT_AAINDEX_PROPERTIES
        assert emb.pool is False
        assert emb.flatten is False
        assert emb.positions is None

    def test_initialization_custom_properties(self, tmp_path):
        custom = [("KYTJ820101", "hydrophobicity")]
        emb = AAPropertiesEmbedding(
            metadata_folder=str(tmp_path), aaindex_properties=custom,
            include_aromatic=False, pool=True,
        )
        assert emb.aaindex_properties == custom
        assert emb.include_aromatic is False
        assert emb.pool is True

    def test_fit_builds_lookup(self, tmp_path, sample_sequences):
        emb = AAPropertiesEmbedding(metadata_folder=str(tmp_path))
        emb.fit(sample_sequences)
        assert emb.embedding_dim_ == len(DEFAULT_AAINDEX_PROPERTIES) + 1
        assert len(emb.property_names_) == emb.embedding_dim_
        assert set(emb.aa_property_lookup_.keys()) == set(CANONICAL)

    def test_transform_per_position(self, tmp_path, sample_sequences):
        emb = AAPropertiesEmbedding(metadata_folder=str(tmp_path), pool=False)
        emb.fit(sample_sequences)
        out = emb.transform(sample_sequences)
        assert out.shape == (3, 5, len(DEFAULT_AAINDEX_PROPERTIES) + 1)

    def test_transform_pooled(self, tmp_path, sample_sequences):
        emb = AAPropertiesEmbedding(metadata_folder=str(tmp_path), pool=True)
        emb.fit(sample_sequences)
        out = emb.transform(sample_sequences)
        assert out.shape == (3, len(DEFAULT_AAINDEX_PROPERTIES) + 1)

    def test_transform_flatten_with_positions(self, tmp_path, sample_sequences):
        dim = len(DEFAULT_AAINDEX_PROPERTIES) + 1
        emb = AAPropertiesEmbedding(
            metadata_folder=str(tmp_path), positions=[0, 2], flatten=True, pool=False,
        )
        emb.fit(sample_sequences)
        out = emb.transform(sample_sequences)
        assert out.shape == (3, 2 * dim)

    def test_include_aromatic_false_dim(self, tmp_path, sample_sequences):
        emb = AAPropertiesEmbedding(metadata_folder=str(tmp_path), include_aromatic=False)
        emb.fit(sample_sequences)
        assert emb.embedding_dim_ == len(DEFAULT_AAINDEX_PROPERTIES)

    def test_get_feature_names_pooled(self, tmp_path, sample_sequences):
        emb = AAPropertiesEmbedding(metadata_folder=str(tmp_path), pool=True)
        emb.fit(sample_sequences)
        names = emb.get_feature_names_out()
        assert len(names) == len(DEFAULT_AAINDEX_PROPERTIES) + 1
        assert all(n.startswith("AAProps_") for n in names)

    def test_get_feature_names_flatten(self, tmp_path, sample_sequences):
        dim = len(DEFAULT_AAINDEX_PROPERTIES) + 1
        emb = AAPropertiesEmbedding(
            metadata_folder=str(tmp_path), positions=[0, 2], flatten=True,
        )
        emb.fit(sample_sequences)
        names = emb.get_feature_names_out()
        assert len(names) == 2 * dim
        assert names[0].startswith("pos0_")

    def test_get_feature_names_before_fit_raises(self, tmp_path):
        emb = AAPropertiesEmbedding(metadata_folder=str(tmp_path), pool=True)
        with pytest.raises(ValueError, match="fitted"):
            emb.get_feature_names_out()

    def test_get_feature_names_non_pool_non_flatten_raises(self, tmp_path, sample_sequences):
        emb = AAPropertiesEmbedding(metadata_folder=str(tmp_path), pool=False, flatten=False)
        emb.fit(sample_sequences)
        with pytest.raises(ValueError):
            emb.get_feature_names_out()

    def test_aligned_gapped_input(self, tmp_path):
        # handle_aligned=True (default): gaps are stripped, scored, then remapped
        # back to alignment columns with gap_fill_value.
        dim = len(DEFAULT_AAINDEX_PROPERTIES) + 1
        emb = AAPropertiesEmbedding(metadata_folder=str(tmp_path), pool=False, gap_fill_value=0.0)
        seqs = ProteinSequences([
            ProteinSequence("AC-DE"),
            ProteinSequence("A-CDE"),
        ])
        emb.fit(seqs)
        out = emb.transform(seqs)
        # Output spans the full alignment width (5).
        assert out.shape == (2, 5, dim)
        # The gap column for each sequence is filled with gap_fill_value (0.0).
        np.testing.assert_array_equal(out[0, 2], np.zeros(dim, dtype=np.float32))
        np.testing.assert_array_equal(out[1, 1], np.zeros(dim, dtype=np.float32))
