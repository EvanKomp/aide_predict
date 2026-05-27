# tests/test_bespoke_models/test_composites/test_zscore.py
'''
* Author: Evan Komp
* Created: 2026-05-26

Unit tests for ZScoreRescaledScorer using a mock inner scorer. These do not
depend on fair-esm / transformers and run under default CI.
'''
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from aide_predict.bespoke_models.base import (
    ProteinModelWrapper,
    CanRegressMixin,
    RequiresFixedLengthMixin,
    ExpectsNoFitMixin,
    RequiresWTToFunctionMixin,
)
from aide_predict.bespoke_models.composites.zscore import ZScoreRescaledScorer
from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences


WT_STR = "ACDEFGHIKLMNPQRSTVWY"   # 20-mer one of each canonical AA


class _MockScorer(
    RequiresFixedLengthMixin,
    CanRegressMixin,
    ExpectsNoFitMixin,
    ProteinModelWrapper,
):
    """
    Deterministic per-variant scalar scorer. For each variant, returns a
    score equal to ``ord(mut_aa) * 0.1 + pos * 0.01`` so different (mut_aa)
    groups have systematically different score distributions — useful to
    verify the z-score rescaling logic.
    """

    def __init__(self, metadata_folder=None, wt=None, pool=True, **kwargs):
        super().__init__(metadata_folder=metadata_folder, wt=wt, **kwargs)
        self.pool = pool

    def _fit(self, X, y=None):
        self.fitted_ = True
        return self

    def _transform(self, X):
        out = []
        for seq in X:
            diffs = seq.mutated_positions(self.wt)
            if len(diffs) == 0:
                out.append(0.0)
            else:
                # base on first mutation; for multi-mutants we sum
                score = 0.0
                for p in diffs:
                    score += ord(str(seq[p])) * 0.1 + p * 0.01
                out.append(score)
        return np.array(out).reshape(-1, 1)


@pytest.fixture
def wt():
    return ProteinSequence(WT_STR, id="wt")


@pytest.fixture
def ssm(wt):
    """All 19*L single-point variants of WT."""
    return wt.saturation_mutagenesis()


@pytest.fixture
def composite(wt):
    inner = _MockScorer(wt=wt, pool=True)
    return ZScoreRescaledScorer(inner_scorer=inner, wt=wt)


class TestMixinFlags:
    def test_requires_wt(self):
        assert ZScoreRescaledScorer._requires_wt_to_function is True

    def test_requires_fixed_length(self):
        assert ZScoreRescaledScorer._requires_fixed_length is True

    def test_can_regress(self):
        assert ZScoreRescaledScorer._can_regress is True

    def test_fit_required(self):
        # Composite needs X at fit time to compute group stats, so it does NOT
        # set _expects_no_fit. (Removed ExpectsNoFitMixin from the inheritance.)
        assert ZScoreRescaledScorer._expects_no_fit is False


class TestInit:
    def test_pool_false_rejected(self, wt):
        inner = _MockScorer(wt=wt, pool=False)
        with pytest.raises(ValueError, match="pool=False"):
            ZScoreRescaledScorer(inner_scorer=inner, wt=wt)

    def test_invalid_grouping(self, wt):
        inner = _MockScorer(wt=wt, pool=True)
        with pytest.raises(ValueError, match="grouping must be"):
            ZScoreRescaledScorer(inner_scorer=inner, wt=wt, grouping="bogus")

    def test_invalid_multi_strategy(self, wt):
        inner = _MockScorer(wt=wt, pool=True)
        with pytest.raises(ValueError, match="multi_mutant_strategy"):
            ZScoreRescaledScorer(inner_scorer=inner, wt=wt, multi_mutant_strategy="explode")

    def test_inner_wt_synced(self, wt):
        inner = _MockScorer(pool=True)   # no wt
        composite = ZScoreRescaledScorer(inner_scorer=inner, wt=wt)
        # inner.wt should now be the composite's wt
        assert composite.inner_scorer.wt == wt


class TestFitTransform:
    def test_fit_transform_shape(self, composite, ssm):
        composite.fit(ssm)
        arr = composite.transform(ssm)
        # 20-residue WT × 19 substitutions = 380 variants
        assert arr.shape == (len(ssm), 2)

    def test_score_table_columns(self, composite, ssm):
        composite.fit(ssm)
        df = composite.score_table(ssm)
        expected = {
            "variant_idx", "variant_id", "n_mutations",
            "mutation", "position", "wt_aa", "mut_aa",
            "logratio", "z_logratio",
        }
        assert expected.issubset(set(df.columns))
        assert len(df) == len(ssm)

    def test_z_logratio_centered_per_group(self, composite, ssm):
        composite.fit(ssm)
        df = composite.score_table(ssm)
        # Group by destination AA → each group's z-scores should have mean ≈ 0
        # and std ≈ 1 (sample std, N-1).
        for dest_aa, sub in df.groupby("mut_aa"):
            if len(sub) >= 5:
                z = sub["z_logratio"].dropna()
                if len(z) > 1:
                    np.testing.assert_allclose(z.mean(), 0.0, atol=1e-6)
                    np.testing.assert_allclose(z.std(ddof=1), 1.0, atol=1e-6)

    def test_score_table_before_fit_raises(self, composite, ssm):
        with pytest.raises(RuntimeError, match="must be fit"):
            composite.score_table(ssm)

    def test_substitution_type_grouping(self, wt):
        inner = _MockScorer(wt=wt, pool=True)
        comp = ZScoreRescaledScorer(
            inner_scorer=inner, wt=wt, grouping="substitution_type"
        )
        ssm_local = wt.saturation_mutagenesis()
        comp.fit(ssm_local)
        df = comp.score_table(ssm_local)
        # Substitution-type groups: (wt_aa, mut_aa). WT has 20 distinct AAs and
        # 19 substitutions each → 20×19 = 380 unique pairs, each with 1 sample,
        # so each group is below min_group_size=5 → all z-scores NaN.
        assert df["z_logratio"].isna().all()

    def test_fit_one_transform_another(self, wt):
        """Stats learned on the SSM should be reused on a smaller variant set."""
        inner = _MockScorer(wt=wt, pool=True)
        comp = ZScoreRescaledScorer(inner_scorer=inner, wt=wt)
        ssm = wt.saturation_mutagenesis()
        comp.fit(ssm)
        stats_after_fit = dict(comp.group_stats_)

        small = ProteinSequences(ssm.data[:10])
        arr = comp.transform(small)
        assert arr.shape == (10, 2)
        # Stats should not have changed.
        assert comp.group_stats_ == stats_after_fit


class TestMultiMutantStrategy:
    def test_skip_default(self, wt):
        inner = _MockScorer(wt=wt, pool=True)
        comp = ZScoreRescaledScorer(inner_scorer=inner, wt=wt, multi_mutant_strategy="skip")
        # mix single + double mutants
        variants = ProteinSequences([
            ProteinSequence("ACDEFGHIKLMNPQRSTVWA"),  # Y20A  → ord('A')=65
            ProteinSequence("ACDEFGHIKLMNPQRSTVWP"),  # Y20P
            ProteinSequence("ACDEFGHIKLMNPQRSTVWG"),  # Y20G
            ProteinSequence("ACDEFGHIKLMNPQRSTVWS"),  # Y20S
            ProteinSequence("ACDEFGHIKLMNPQRSTVWT"),  # Y20T
            ProteinSequence("AADEFGHIKLMNPQRSTVWA"),  # double mutant C2A,Y20A
        ])
        comp.fit(variants)
        df = comp.score_table(variants)
        # Double mutant row's z_logratio should be NaN.
        assert pd.isna(df.iloc[5]["z_logratio"])
        # But its logratio is finite (inner scorer scored it).
        assert np.isfinite(df.iloc[5]["logratio"])

    def test_raise(self, wt):
        inner = _MockScorer(wt=wt, pool=True)
        comp = ZScoreRescaledScorer(inner_scorer=inner, wt=wt, multi_mutant_strategy="raise")
        variants = ProteinSequences([
            ProteinSequence("AADEFGHIKLMNPQRSTVWA"),  # double mutant
        ])
        with pytest.raises(ValueError, match="multi-mutation"):
            comp.fit(variants)


class TestEdgeCases:
    def test_group_below_min_size(self, wt):
        inner = _MockScorer(wt=wt, pool=True)
        comp = ZScoreRescaledScorer(inner_scorer=inner, wt=wt, min_group_size=5)
        # Only 3 destination-residue groups, each with 3 samples → all below min size.
        variants = ProteinSequences([
            ProteinSequence("CCDEFGHIKLMNPQRSTVWY"),   # A1C
            ProteinSequence("ACCEFGHIKLMNPQRSTVWY"),   # D3C
            ProteinSequence("ACDEFCHIKLMNPQRSTVWY"),   # G6C
        ])
        comp.fit(variants)
        df = comp.score_table(variants)
        assert df["z_logratio"].isna().all()

    def test_wt_identical_variant(self, wt):
        inner = _MockScorer(wt=wt, pool=True)
        comp = ZScoreRescaledScorer(inner_scorer=inner, wt=wt)
        variants = ProteinSequences([
            ProteinSequence(WT_STR),  # identical to WT
        ])
        # No fit; just transform after a separate fit on real SSM:
        ssm = wt.saturation_mutagenesis()
        comp.fit(ssm)
        df = comp.score_table(variants)
        assert df.iloc[0]["n_mutations"] == 0
        # z_logratio NaN, logratio = inner's prediction (0.0 for our mock)
        assert pd.isna(df.iloc[0]["z_logratio"])
