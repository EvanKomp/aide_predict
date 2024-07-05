# tests/test_bespoke_models/test_embeddings/test_ohe.py
'''
* Author: Evan Komp
* Created: 7/5/2024
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT
'''
import pytest
import numpy as np
from aide_predict.utils.data_structures import ProteinSequences, ProteinSequence
from aide_predict.bespoke_models.embedders.ohe import OneHotProteinEmbedding, OneHotAlignedEmbedding
from aide_predict.utils.constants import AA_SINGLE, GAP_CHARACTERS

class TestOneHotProteinEmbedding:
    @pytest.fixture
    def sample_sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDEF"),
            ProteinSequence("GHIKL"),
            ProteinSequence("MNPQR")
        ])

    @pytest.fixture
    def embedder(self, tmp_path):
        return OneHotProteinEmbedding(metadata_folder=str(tmp_path), flatten=False)

    def test_initialization(self, embedder):
        assert embedder._vocab == list(AA_SINGLE)
        assert not hasattr(embedder, 'seq_length_')
        assert not hasattr(embedder, 'positions_len_')

    def test_fit(self, embedder, sample_sequences):
        embedder.fit(sample_sequences)
        assert embedder.seq_length_ == 5
        assert embedder.positions_len_ == 5
        assert embedder.encoder_ is not None

    def test_transform(self, embedder, sample_sequences):
        embedder.fit(sample_sequences)
        transformed = embedder.transform(sample_sequences)
        assert transformed.shape == (3, 5, 20)  # 3 sequences, 5 positions, 20 amino acids

    def test_position_specific_encoding(self, tmp_path, sample_sequences):
        embedder = OneHotProteinEmbedding(metadata_folder=str(tmp_path), positions=[0, 2, 4], flatten=False)
        embedder.fit(sample_sequences)
        transformed = embedder.transform(sample_sequences)
        assert transformed.shape == (3, 3, 20)  # 3 sequences, 3 positions, 20 amino acids

    def test_flattening(self, tmp_path, sample_sequences):
        embedder = OneHotProteinEmbedding(metadata_folder=str(tmp_path), flatten=True)
        embedder.fit(sample_sequences)
        transformed = embedder.transform(sample_sequences)
        assert transformed.shape == (3, 100)  # 3 sequences, 5 positions * 20 amino acids

    def test_get_feature_names_out(self, embedder, sample_sequences):
        embedder.fit(sample_sequences)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 5  # 5 positions
        assert feature_names[0] == "pos0"
        assert feature_names[-1] == "pos4"

    def test_get_feature_names_out_flattened(self, tmp_path, sample_sequences):
        embedder = OneHotProteinEmbedding(metadata_folder=str(tmp_path), flatten=True)
        embedder.fit(sample_sequences)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 100  # 5 positions * 20 amino acids
        assert feature_names[0].startswith("pos0_")
        assert feature_names[-1].startswith("pos4_")

    def test_invalid_input(self, embedder):
        with pytest.raises(ValueError):
            embedder.fit(ProteinSequences([ProteinSequence("ACE"), ProteinSequence("ACDEF")]))

        with pytest.raises(ValueError):
            embedder.fit(ProteinSequences([ProteinSequence("AC-EF"), ProteinSequence("ACDEF")]))

    def test_transform_mismatch(self, embedder, sample_sequences):
        embedder.fit(sample_sequences)
        with pytest.raises(ValueError, match="Input sequences must have length 5"):
            embedder.transform(ProteinSequences([ProteinSequence("ACDEFG")]))

    @pytest.mark.parametrize("flatten,expected_shape", [
        (True, (3, 100)),
        (False, (3, 5, 20)),
    ])
    def test_flatten_combinations(self, tmp_path, sample_sequences, flatten, expected_shape):
        embedder = OneHotProteinEmbedding(metadata_folder=str(tmp_path), flatten=flatten)
        embedder.fit(sample_sequences)
        transformed = embedder.transform(sample_sequences)
        assert transformed.shape == expected_shape

    def test_wild_type_handling(self, tmp_path, sample_sequences):
        wt = ProteinSequence("ACDEF")
        embedder = OneHotProteinEmbedding(metadata_folder=str(tmp_path), wt=wt)
        embedder.fit(sample_sequences)
        assert embedder.wt == wt

        with pytest.raises(ValueError, match="Wild type sequence cannot have gaps"):
            OneHotProteinEmbedding(metadata_folder=str(tmp_path), wt="AC-EF")


class TestOneHotAlignedEmbedding:
    @pytest.fixture
    def aligned_sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDEF-GH"),
            ProteinSequence("ACD-FGH-"),
            ProteinSequence("AC-EFGH-")
        ])

    @pytest.fixture
    def unaligned_sequences(self):
        return ProteinSequences([
            ProteinSequence("ACDEFGH"),
            ProteinSequence("ACDFGH"),
            ProteinSequence("ACEFGH")
        ])

    @pytest.fixture
    def embedder(self, tmp_path):
        return OneHotAlignedEmbedding(metadata_folder=str(tmp_path), flatten=False)

    def test_initialization(self, embedder):
        assert embedder._vocab == list(AA_SINGLE.union(GAP_CHARACTERS))
        assert not hasattr(embedder, 'alignment_width_')
        assert not hasattr(embedder, 'positions_len_')

    def test_fit(self, embedder, aligned_sequences):
        embedder.fit(aligned_sequences)
        assert embedder.alignment_width_ == 8
        assert embedder.positions_len_ == 8
        assert isinstance(embedder.original_alignment_, ProteinSequences)

    def test_transform_aligned(self, embedder, aligned_sequences):
        embedder.fit(aligned_sequences)
        transformed = embedder.transform(aligned_sequences)
        assert transformed.shape == (3, 8, 22)  # 3 sequences, 8 positions, 22 characters (20 amino acids + 2 gap characters)

    def test_transform_unaligned(self, embedder, aligned_sequences, unaligned_sequences):
        embedder.fit(aligned_sequences)
        with pytest.warns(UserWarning, match="Input sequences are not aligned"):
            transformed = embedder.transform(unaligned_sequences)
        assert transformed.shape == (3, 8, 22)

    def test_transform_misaligned(self, embedder, aligned_sequences):
        embedder.fit(aligned_sequences)
        misaligned = ProteinSequences([ProteinSequence("ACDEF-GHI")])  # One character too long
        with pytest.raises(ValueError, match="Aligned input sequences must have width 8"):
            embedder.transform(misaligned)

    def test_get_feature_names_out(self, embedder, aligned_sequences):
        embedder.fit(aligned_sequences)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 8  # 8 positions
        assert feature_names[0] == "pos0"
        assert feature_names[-1] == "pos7"

    def test_get_feature_names_out_flattened(self, tmp_path, aligned_sequences):
        embedder = OneHotAlignedEmbedding(metadata_folder=str(tmp_path), flatten=True)
        embedder.fit(aligned_sequences)
        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 8 * 22  # 8 positions * 22 characters
        assert feature_names[0].startswith("pos0_")
        assert feature_names[-1].startswith("pos7_")

    def test_specific_positions(self, tmp_path, aligned_sequences):
        embedder = OneHotAlignedEmbedding(metadata_folder=str(tmp_path), positions=[0, 2, 4], flatten=False)
        embedder.fit(aligned_sequences)
        transformed = embedder.transform(aligned_sequences)
        assert transformed.shape == (3, 3, 22)  # 3 sequences, 3 positions, 23 characters

    @pytest.mark.parametrize("flatten,expected_shape", [
        (True, (3, 8 * 22)),
        (False, (3, 8, 22)),
    ])
    def test_flatten_combinations(self, tmp_path, aligned_sequences, flatten, expected_shape):
        embedder = OneHotAlignedEmbedding(metadata_folder=str(tmp_path), flatten=flatten)
        embedder.fit(aligned_sequences)
        transformed = embedder.transform(aligned_sequences)
        assert transformed.shape == expected_shape


    def test_transform_before_fit(self, embedder, aligned_sequences):
        with pytest.raises(AttributeError):
            embedder.transform(aligned_sequences)

    def test_get_feature_names_before_fit(self, embedder):
        with pytest.raises(ValueError, match="Encoder has not been fitted yet"):
            embedder.get_feature_names_out()

    def test_wild_type_handling(self, tmp_path, aligned_sequences):
        wt = ProteinSequence("ACDEFGH")
        embedder = OneHotAlignedEmbedding(metadata_folder=str(tmp_path), wt=wt)
        embedder.fit(aligned_sequences)
        assert embedder.wt == wt

        with pytest.raises(ValueError, match="Wild type sequence cannot have gaps"):
            OneHotAlignedEmbedding(metadata_folder=str(tmp_path), wt="ACD-FGH")

