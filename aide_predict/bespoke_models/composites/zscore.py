# aide_predict/bespoke_models/composites/zscore.py
'''
* Author: Evan Komp
* Created: 2026-05-26
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

ZScoreRescaledScorer — a composition wrapper that runs any per-variant-scalar
scorer (e.g. an ESM-IF / ESM2 / SaProt wrapper in pool=True mode) on a set of
variants and rescales the per-variant log-ratios within an amino-acid group
(destination residue or substitution type), matching MULTI-evolve's z-score
rescaling step.

Designed for SSM workflows: fit on ``wt.saturation_mutagenesis()``, then
transform any subset (or the SSM itself) to get both the raw log-ratio and
its AA-group z-score per variant.
'''
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Literal

import numpy as np
import pandas as pd

from aide_predict.bespoke_models.base import (
    ProteinModelWrapper,
    RequiresWTToFunctionMixin,
    RequiresWTDuringInferenceMixin,
    RequiresFixedLengthMixin,
    CanRegressMixin,
    ShouldRefitOnSequencesMixin,
)
from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences
from aide_predict.utils.scoring import per_variant_mutation_info, zscore_by_aa_group


class ZScoreRescaledScorer(
    RequiresWTToFunctionMixin,
    RequiresWTDuringInferenceMixin,
    RequiresFixedLengthMixin,
    CanRegressMixin,
    ShouldRefitOnSequencesMixin,
    ProteinModelWrapper,
):
    """
    Composition wrapper: runs an inner per-variant-scalar scorer and adds
    AA-group z-score rescaling.

    Mechanics:
    - ``inner_scorer`` is any ProteinModelWrapper whose ``transform(X)``
      returns one scalar per variant (e.g. any wildtype_marginal-mode
      LikelihoodTransformerBase subclass with ``pool=True``).
    - At ``fit``, the inner scorer is fit on X, then its scores are paired
      with each variant's (wt_aa, mut_aa) identity to compute per-group
      (mean, std). Group keys are stored on the wrapper.
    - At ``transform``, the inner is rerun on the new X; per-variant scalars
      are paired with mutation identity; the stored group stats produce a
      z-score per variant. Output shape ``(N, 2)`` with columns
      ``[logratio, z_logratio]``.
    - ``score_table(X)`` returns the full per-variant DataFrame for the
      human-readable view (variant_idx, variant_id, mutation, position,
      wt_aa, mut_aa, logratio, z_logratio, n_mutations).

    Multi-mutation variants are handled per ``multi_mutant_strategy`` —
    ``"skip"`` (default) gives them NaN z-scores while leaving their
    log-ratios untouched; ``"raise"`` errors if any are present.

    Note: groups with fewer than ``min_group_size`` samples at fit time
    contribute no stats; variants in those groups get NaN z-scores at
    transform. This matches MULTI-evolve's threshold of 5.
    """

    def __init__(
        self,
        inner_scorer: ProteinModelWrapper,
        grouping: Literal["destination_residue", "substitution_type"] = "destination_residue",
        min_group_size: int = 5,
        multi_mutant_strategy: Literal["skip", "raise"] = "skip",
        metadata_folder: Optional[str] = None,
        wt: Optional[Union[str, ProteinSequence]] = None,
        **kwargs,
    ):
        if hasattr(inner_scorer, "pool") and inner_scorer.pool is False:
            raise ValueError(
                "ZScoreRescaledScorer expects the inner scorer to produce one "
                "scalar per variant; got inner_scorer.pool=False (per-position "
                "output). Set inner_scorer.pool=True."
            )
        if grouping not in ("destination_residue", "substitution_type"):
            raise ValueError(
                f"grouping must be 'destination_residue' or 'substitution_type', got {grouping!r}"
            )
        if multi_mutant_strategy not in ("skip", "raise"):
            raise ValueError(
                f"multi_mutant_strategy must be 'skip' or 'raise', got {multi_mutant_strategy!r}"
            )

        super().__init__(metadata_folder=metadata_folder, wt=wt, **kwargs)

        self.inner_scorer = inner_scorer
        self.grouping = grouping
        self.min_group_size = min_group_size
        self.multi_mutant_strategy = multi_mutant_strategy

        # Sync WT into the inner scorer if it doesn't have one yet.
        if self.wt is not None and getattr(inner_scorer, "wt", None) is None:
            inner_scorer.wt = self.wt

    def _fit(self, X: ProteinSequences, y: Optional[np.ndarray] = None) -> "ZScoreRescaledScorer":
        self.inner_scorer.fit(X, y)
        scalar_scores = self._call_inner(X)
        df = self._build_table(X, scalar_scores)
        self._check_multi_mutants(df)
        _, self.group_stats_ = zscore_by_aa_group(
            df,
            grouping=self.grouping,
            score_col="logratio",
            out_col="z_logratio",
            min_group_size=self.min_group_size,
        )
        return self

    def _transform(self, X: ProteinSequences) -> np.ndarray:
        df = self.score_table(X)
        return df[["logratio", "z_logratio"]].to_numpy()

    def score_table(self, X: ProteinSequences) -> pd.DataFrame:
        """
        Return the full per-variant DataFrame with columns
        ``[variant_idx, variant_id, n_mutations, mutation, position, wt_aa,
        mut_aa, logratio, z_logratio]``.

        Uses the group stats stored at ``fit`` time to compute z-scores;
        variants belonging to a group that wasn't seen during fit get NaN
        for ``z_logratio``.
        """
        if not hasattr(self, "group_stats_"):
            raise RuntimeError("ZScoreRescaledScorer must be fit before score_table is called.")
        scalar_scores = self._call_inner(X)
        df = self._build_table(X, scalar_scores)
        self._check_multi_mutants(df)
        df, _ = zscore_by_aa_group(
            df,
            grouping=self.grouping,
            score_col="logratio",
            out_col="z_logratio",
            min_group_size=self.min_group_size,
            fit_stats=self.group_stats_,
        )
        return df

    def _call_inner(self, X: ProteinSequences) -> np.ndarray:
        out = np.asarray(self.inner_scorer.transform(X))
        if out.ndim == 2 and out.shape[1] == 1:
            out = out.ravel()
        if out.ndim != 1:
            raise ValueError(
                "Inner scorer returned shape "
                f"{out.shape}; ZScoreRescaledScorer expects one scalar per variant. "
                "Configure the inner scorer with pool=True."
            )
        if out.shape[0] != len(X):
            raise ValueError(
                f"Inner scorer returned {out.shape[0]} scores for {len(X)} variants."
            )
        return out

    def _build_table(self, X: ProteinSequences, scores: np.ndarray) -> pd.DataFrame:
        info = per_variant_mutation_info(X, self.wt)
        info["logratio"] = scores
        return info

    def _check_multi_mutants(self, df: pd.DataFrame) -> None:
        if self.multi_mutant_strategy == "raise":
            multi = df[df["n_mutations"] > 1]
            if len(multi) > 0:
                raise ValueError(
                    f"{len(multi)} multi-mutation variant(s) present with "
                    "multi_mutant_strategy='raise'. Set multi_mutant_strategy='skip' "
                    "to tolerate them (their z-scores will be NaN)."
                )
