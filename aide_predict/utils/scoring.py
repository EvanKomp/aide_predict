# aide_predict/utils/scoring.py
'''
* Author: Evan Komp
* Created: 2026-05-26
* Company: National Renewable Energy Lab, Bioeneergy Science and Technology
* License: MIT

Scorer-agnostic primitives for per-variant mutation accounting and
amino-acid-grouped z-score rescaling.

These two functions together let any per-variant scalar score (e.g. the
output of a wildtype_marginal scorer in pool=True mode) be rescaled against
the distribution of scores within a destination-AA or substitution-type
group — the rescaling the MULTI-evolve ensemble paper uses on top of
ESM-1v/ESM2/ESM-IF outputs.
'''
from typing import Dict, List, Optional, Tuple
from typing_extensions import Literal

import numpy as np
import pandas as pd

from aide_predict.utils.data_structures import ProteinSequence, ProteinSequences


def per_variant_mutation_info(
    sequences: ProteinSequences,
    wt: ProteinSequence,
) -> pd.DataFrame:
    """
    Extract single-point mutation identity for each variant in ``sequences``.

    Returns a DataFrame with one row per input variant containing:

    - ``variant_idx`` (int): row index in ``sequences``.
    - ``variant_id`` (Optional[str]): ``sequences[v].id`` (None if unset).
    - ``n_mutations`` (int): number of positions differing from WT.
    - ``mutation`` (Optional[str]): e.g. ``"A123L"`` (1-indexed), only set when
      ``n_mutations == 1``; ``None`` otherwise.
    - ``position`` (Optional[int]): 1-indexed mutated position, only set when
      ``n_mutations == 1``.
    - ``wt_aa`` (Optional[str]): WT amino acid at the mutated position, only
      set when ``n_mutations == 1``.
    - ``mut_aa`` (Optional[str]): variant amino acid at the mutated position,
      only set when ``n_mutations == 1``.

    Multi-mutation variants still get a row (with the per-mutation columns
    None) so callers can choose to drop, skip, or error on them — the
    decision is left to the consumer (typically ``ZScoreRescaledScorer``).

    Args:
        sequences: ProteinSequences of variants to inspect.
        wt: Wild-type sequence to compare against. Must match the length of
            each variant.

    Returns:
        pd.DataFrame with the columns described above, one row per variant.
    """
    rows = []
    for v_idx, seq in enumerate(sequences):
        if len(seq) != len(wt):
            raise ValueError(
                f"Variant {v_idx} has length {len(seq)} but WT has length {len(wt)}. "
                "per_variant_mutation_info requires same-length sequences."
            )
        diffs = seq.mutated_positions(wt)  # 0-indexed
        row = {
            "variant_idx": v_idx,
            "variant_id": seq.id,
            "n_mutations": len(diffs),
            "mutation": None,
            "position": None,
            "wt_aa": None,
            "mut_aa": None,
        }
        if len(diffs) == 1:
            p = diffs[0]
            row["position"] = p + 1
            row["wt_aa"] = str(wt[p])
            row["mut_aa"] = str(seq[p])
            row["mutation"] = f"{row['wt_aa']}{row['position']}{row['mut_aa']}"
        rows.append(row)
    return pd.DataFrame(rows)


def zscore_by_aa_group(
    df: pd.DataFrame,
    grouping: Literal["destination_residue", "substitution_type"] = "destination_residue",
    score_col: str = "score",
    out_col: Optional[str] = None,
    min_group_size: int = 5,
    fit_stats: Optional[Dict[Tuple[str, ...], Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, ...], Tuple[float, float]]]:
    """
    Add an AA-group z-score column to ``df`` (matches MULTI-evolve's rescaling).

    Groups are keyed by:
      - ``"destination_residue"`` (default) → ``(mut_aa,)`` — bins all '→P'
        mutations together.
      - ``"substitution_type"`` → ``(wt_aa, mut_aa)`` — bins all 'A→P'
        mutations together.

    Per-group statistics are ``(mean, std)`` with pandas' default sample
    standard deviation (N-1 denominator). Groups with fewer than
    ``min_group_size`` samples are not z-scored — output values stay NaN.
    Rows missing the grouping columns (e.g. multi-mutation variants without
    a defined ``mut_aa``) are also NaN'd.

    Args:
        df: Input DataFrame. Must contain ``score_col``, ``mut_aa``, and
            (if ``grouping=="substitution_type"``) ``wt_aa``.
        grouping: Group definition.
        score_col: Column to z-score.
        out_col: Output column name. Defaults to ``f"z_{score_col}"``.
        min_group_size: Groups with fewer samples produce NaN z-scores
            (matches MULTI-evolve's threshold of 5).
        fit_stats: If provided, skip stat-computation and apply these
            ``(mean, std)`` per group instead. Enables the sklearn-style
            fit/transform split where stats are learned on a calibration
            set (typically the full SSM) and applied to new variants.

    Returns:
        Tuple of (DataFrame with the new column added, dict of group_key →
        (mean, std)). The returned stats dict is whatever ``fit_stats`` was
        passed in, or — if ``fit_stats is None`` — the freshly computed
        stats so the caller can persist them for later transform calls.
    """
    if score_col not in df.columns:
        raise KeyError(f"score_col '{score_col}' not in df.columns ({list(df.columns)})")
    if "mut_aa" not in df.columns:
        raise KeyError("zscore_by_aa_group requires a 'mut_aa' column on df")
    if grouping == "substitution_type" and "wt_aa" not in df.columns:
        raise KeyError("zscore_by_aa_group with grouping='substitution_type' requires a 'wt_aa' column")

    if out_col is None:
        out_col = f"z_{score_col}"

    # Build per-row hashable group keys. Using a plain list avoids
    # pandas.DataFrame.apply / Series-of-tuple equality quirks that surface
    # when comparing tuple keys against a stored fit_stats dict at transform
    # time.
    mut_col = df["mut_aa"].tolist()
    if grouping == "destination_residue":
        keys: List[Optional[Tuple[str, ...]]] = [
            (m,) if (m is not None and not (isinstance(m, float) and np.isnan(m))) else None
            for m in mut_col
        ]
    else:
        wt_col = df["wt_aa"].tolist()
        keys = [
            (w, m) if (
                w is not None and m is not None
                and not (isinstance(w, float) and np.isnan(w))
                and not (isinstance(m, float) and np.isnan(m))
            ) else None
            for w, m in zip(wt_col, mut_col)
        ]

    if fit_stats is None:
        # Compute group stats by aggregating row indices per key.
        key_to_rows: Dict[Tuple[str, ...], List[int]] = {}
        for i, k in enumerate(keys):
            if k is None:
                continue
            key_to_rows.setdefault(k, []).append(i)

        stats: Dict[Tuple[str, ...], Tuple[float, float]] = {}
        scores = df[score_col].to_numpy()
        for k, idxs in key_to_rows.items():
            vals = scores[idxs]
            vals = vals[~np.isnan(vals)] if vals.dtype.kind == "f" else vals
            if len(vals) < min_group_size:
                continue
            std = float(np.std(vals, ddof=1))
            if not np.isfinite(std) or std == 0:
                continue
            stats[k] = (float(np.mean(vals)), std)
    else:
        stats = fit_stats

    df = df.copy()
    z_out = np.full(len(df), np.nan)
    scores = df[score_col].to_numpy()
    for i, k in enumerate(keys):
        if k is None or k not in stats:
            continue
        s = scores[i]
        if isinstance(s, float) and np.isnan(s):
            continue
        mean, std = stats[k]
        z_out[i] = (s - mean) / std
    df[out_col] = z_out
    return df, stats
