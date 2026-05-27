# tests/test_utils/test_scoring.py
'''
* Author: Evan Komp
* Created: 2026-05-26
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Unit tests for the scorer-agnostic primitives in aide_predict.utils.scoring.
'''
import numpy as np
import pandas as pd
import pytest

from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences
from aide_predict.utils.scoring import per_variant_mutation_info, zscore_by_aa_group


WT_STR = "ACDEFGHIKL"


@pytest.fixture
def wt():
    return ProteinSequence(WT_STR, id="wt")


class TestPerVariantMutationInfo:
    def test_single_point_variants(self, wt):
        variants = ProteinSequences([
            ProteinSequence("ACDEYGHIKL", id="F5Y"),   # single mutation at pos 5
            ProteinSequence("ACDEFGHIKP", id="L10P"),  # single mutation at pos 10
        ])
        df = per_variant_mutation_info(variants, wt)
        assert len(df) == 2
        assert df.iloc[0]["mutation"] == "F5Y"
        assert df.iloc[0]["position"] == 5
        assert df.iloc[0]["wt_aa"] == "F"
        assert df.iloc[0]["mut_aa"] == "Y"
        assert df.iloc[0]["n_mutations"] == 1
        assert df.iloc[1]["mutation"] == "L10P"
        assert df.iloc[1]["position"] == 10

    def test_multi_mutation_variant(self, wt):
        variants = ProteinSequences([
            ProteinSequence("ACDEYGHIKP", id="F5Y_L10P"),  # two mutations
        ])
        df = per_variant_mutation_info(variants, wt)
        assert df.iloc[0]["n_mutations"] == 2
        assert df.iloc[0]["mutation"] is None
        assert df.iloc[0]["position"] is None
        assert df.iloc[0]["wt_aa"] is None
        assert df.iloc[0]["mut_aa"] is None

    def test_identical_to_wt(self, wt):
        variants = ProteinSequences([ProteinSequence(WT_STR, id="wt_copy")])
        df = per_variant_mutation_info(variants, wt)
        assert df.iloc[0]["n_mutations"] == 0
        assert df.iloc[0]["mutation"] is None

    def test_variant_id_preserved(self, wt):
        variants = ProteinSequences([
            ProteinSequence("ACDEYGHIKL", id="some_id"),
            ProteinSequence("ACDEFGHIKP"),  # no id
        ])
        df = per_variant_mutation_info(variants, wt)
        assert df.iloc[0]["variant_id"] == "some_id"
        assert df.iloc[1]["variant_id"] is None

    def test_length_mismatch_raises(self, wt):
        variants = ProteinSequences([ProteinSequence("ACDEF", id="short")])
        with pytest.raises(ValueError, match="length"):
            per_variant_mutation_info(variants, wt)


class TestZScoreByAAGroup:
    @pytest.fixture
    def synthetic_df(self):
        # Build a df with two destination groups, each with 5 samples.
        # Group "→P": scores = [1, 2, 3, 4, 5], mean=3, std=sqrt(2.5) ≈ 1.5811
        # Group "→Q": scores = [10, 20, 30, 40, 50], mean=30, std=sqrt(250) ≈ 15.811
        rows = []
        for s in [1, 2, 3, 4, 5]:
            rows.append({"score": float(s), "wt_aa": "A", "mut_aa": "P"})
        for s in [10, 20, 30, 40, 50]:
            rows.append({"score": float(s), "wt_aa": "A", "mut_aa": "Q"})
        return pd.DataFrame(rows)

    def test_destination_residue_grouping(self, synthetic_df):
        df_out, stats = zscore_by_aa_group(
            synthetic_df, grouping="destination_residue", score_col="score"
        )
        # Two groups: ('P',) and ('Q',)
        assert set(stats.keys()) == {("P",), ("Q",)}
        # P group: mean=3, std≈1.5811
        np.testing.assert_allclose(stats[("P",)][0], 3.0)
        np.testing.assert_allclose(stats[("P",)][1], np.std([1, 2, 3, 4, 5], ddof=1))
        # Z scores: row 0 (score=1) → (1-3)/1.5811 ≈ -1.2649
        z_p_first = (1.0 - 3.0) / np.std([1, 2, 3, 4, 5], ddof=1)
        np.testing.assert_allclose(df_out.iloc[0]["z_score"], z_p_first)
        # Row 5 (score=10, group Q) → (10-30)/std_q
        z_q_first = (10.0 - 30.0) / np.std([10, 20, 30, 40, 50], ddof=1)
        np.testing.assert_allclose(df_out.iloc[5]["z_score"], z_q_first)

    def test_substitution_type_grouping(self, synthetic_df):
        df_out, stats = zscore_by_aa_group(
            synthetic_df, grouping="substitution_type", score_col="score"
        )
        # Two groups: ('A','P') and ('A','Q')
        assert set(stats.keys()) == {("A", "P"), ("A", "Q")}

    def test_min_group_size_filter(self):
        # Group with 4 samples → below threshold of 5, z-scores stay NaN.
        df = pd.DataFrame([
            {"score": 1.0, "wt_aa": "A", "mut_aa": "P"},
            {"score": 2.0, "wt_aa": "A", "mut_aa": "P"},
            {"score": 3.0, "wt_aa": "A", "mut_aa": "P"},
            {"score": 4.0, "wt_aa": "A", "mut_aa": "P"},
        ])
        df_out, stats = zscore_by_aa_group(df, score_col="score", min_group_size=5)
        assert ("P",) not in stats
        assert df_out["z_score"].isna().all()

    def test_custom_out_col(self, synthetic_df):
        df_out, _ = zscore_by_aa_group(
            synthetic_df, score_col="score", out_col="custom_z"
        )
        assert "custom_z" in df_out.columns
        assert "z_score" not in df_out.columns

    def test_fit_stats_override(self, synthetic_df):
        # Provide pre-computed stats and confirm they're used as-is.
        fit_stats = {("P",): (100.0, 1.0), ("Q",): (200.0, 1.0)}
        df_out, returned_stats = zscore_by_aa_group(
            synthetic_df, score_col="score", fit_stats=fit_stats
        )
        # P group score=1 → (1-100)/1 = -99
        np.testing.assert_allclose(df_out.iloc[0]["z_score"], -99.0)
        # Q group score=10 → (10-200)/1 = -190
        np.testing.assert_allclose(df_out.iloc[5]["z_score"], -190.0)
        assert returned_stats is fit_stats

    def test_missing_mut_aa_column_raises(self):
        df = pd.DataFrame({"score": [1.0, 2.0]})
        with pytest.raises(KeyError, match="mut_aa"):
            zscore_by_aa_group(df, score_col="score")

    def test_missing_wt_aa_for_substitution_type_raises(self):
        df = pd.DataFrame({"score": [1.0], "mut_aa": ["P"]})
        with pytest.raises(KeyError, match="wt_aa"):
            zscore_by_aa_group(df, grouping="substitution_type", score_col="score")

    def test_nan_mut_aa_rows_left_nan(self):
        # Multi-mutation rows (mut_aa = None) shouldn't get z-scored.
        rows = [{"score": float(s), "wt_aa": "A", "mut_aa": "P"} for s in range(1, 6)]
        rows.append({"score": 99.0, "wt_aa": None, "mut_aa": None})
        df = pd.DataFrame(rows)
        df_out, stats = zscore_by_aa_group(df, score_col="score")
        assert pd.isna(df_out.iloc[5]["z_score"])
        # The first 5 rows ARE z-scored (P group has 5 samples).
        assert df_out.iloc[:5]["z_score"].notna().all()
