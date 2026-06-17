# tests/test_not_base_models/test_zscore_composite.py
'''
* Author: Evan Komp
* Created: 2026-05-26

Integration tests for ZScoreRescaledScorer wrapping the real ESM-IF wrapper.
Excluded from default CI (@pytest.mark.optional) because it loads fair-esm.
'''
import os

import numpy as np
import pandas as pd
import pytest
import torch

from aide_predict.utils.data_structures import (
    ProteinSequence,
    ProteinSequences,
    ProteinStructure,
)
from aide_predict.bespoke_models.composites.zscore import ZScoreRescaledScorer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENVZ_WT = "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR"


@pytest.mark.optional
def test_zscore_wrapping_esm_if_envz_ssm():
    """End-to-end: wrap ESM-IF wildtype_marginal with z-score rescaling on the
    ENVZ_ECOLI SSM. Check shape, columns, and that within-group z-scores have
    the expected mean ≈ 0 / std ≈ 1 behavior."""
    from aide_predict.bespoke_models.predictors.esm_if import ESMIFLikelihoodWrapper

    structure = ProteinStructure(os.path.join("tests", "data", "ENVZ_ECOLI.pdb"))
    wt = ProteinSequence(ENVZ_WT, structure=structure, id="ENVZ_WT")

    inner = ESMIFLikelihoodWrapper(
        marginal_method="wildtype_marginal",
        wt=wt,
        pool=True,
        device=DEVICE,
    )
    composite = ZScoreRescaledScorer(
        inner_scorer=inner,
        grouping="destination_residue",
        wt=wt,
    )

    ssm = wt.saturation_mutagenesis()
    assert len(ssm) == len(ENVZ_WT) * 19   # 60 * 19 = 1140

    composite.fit(ssm)
    arr = composite.transform(ssm)
    assert arr.shape == (len(ssm), 2)
    assert np.isfinite(arr[:, 0]).all()   # raw log-ratios should all be finite

    df = composite.score_table(ssm)
    expected_cols = {
        "variant_idx", "variant_id", "n_mutations",
        "mutation", "position", "wt_aa", "mut_aa",
        "logratio", "z_logratio",
    }
    assert expected_cols.issubset(df.columns)
    assert (df["n_mutations"] == 1).all()
    # Every group in an SSM has 60 samples (one per position) so all groups
    # exceed min_group_size=5 and every variant gets a z_logratio.
    assert df["z_logratio"].notna().all()

    # Within each destination-AA group, z-scores should be centered and unit-std.
    for dest_aa, sub in df.groupby("mut_aa"):
        z = sub["z_logratio"]
        np.testing.assert_allclose(z.mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(z.std(ddof=1), 1.0, atol=1e-6)


@pytest.mark.optional
def test_zscore_fit_one_transform_subset():
    """Fit on the full SSM, transform a subset — stats must persist and be applied."""
    from aide_predict.bespoke_models.predictors.esm_if import ESMIFLikelihoodWrapper

    structure = ProteinStructure(os.path.join("tests", "data", "ENVZ_ECOLI.pdb"))
    wt = ProteinSequence(ENVZ_WT, structure=structure, id="ENVZ_WT")

    composite = ZScoreRescaledScorer(
        inner_scorer=ESMIFLikelihoodWrapper(
            marginal_method="wildtype_marginal", wt=wt, pool=True, device=DEVICE,
        ),
        wt=wt,
    )

    ssm = wt.saturation_mutagenesis()
    composite.fit(ssm)
    learned_stats = dict(composite.group_stats_)

    subset = ProteinSequences(ssm.data[:20])
    arr = composite.transform(subset)
    assert arr.shape == (20, 2)
    assert np.isfinite(arr).all()

    # Stats must not have been recomputed during transform.
    assert composite.group_stats_ == learned_stats
