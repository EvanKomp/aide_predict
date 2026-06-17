---
title: Ensemble Variant Nomination
---

# Ensemble Variant Nomination (MULTI-evolve first round)

## Overview

This guide reproduces the first round of the [MULTI-evolve](https://www.science.org/doi/10.1126/science.aea1820) zero-shot nomination workflow using AIDE primitives. The idea is to score a full saturation-mutagenesis (SSM) library along two complementary tracks and combine them:

- a **structure track** — `ESMIFLikelihoodWrapper` (ESM-IF1), conditioned on the backbone;
- a **sequence track** — an ensemble of sequence PLMs (`ESM2LikelihoodWrapper` loading the ESM-1v / ESM-2 checkpoints), averaged.

Each track is scored as a per-mutation log-ratio, then **rescaled within an amino-acid group** (`zscore_by_aa_group`) so that mutations to, say, proline are compared against other mutations to proline rather than against the whole library. Finally we take a **position-exclusive top-N** under four rankings (fold-change and z-score for each track) and **union** them — at most one mutation per residue per method.

All of the pieces below are public API. A ready-to-run driver that implements this exact workflow (including multimer handling and plots) lives at `scripts/multi_evolve_ssm_scan.py`.

## 1. Build the WT and its SSM library

```python
from aide_predict import ProteinSequence, ProteinSequences

# from_pdb extracts the chain sequence and attaches the ProteinStructure.
wt = ProteinSequence.from_pdb("my_enzyme.pdb", chain="A", id="my_enzyme")

# All single-point variants (19 * L). MULTI-evolve drops Met1 by convention
# (mutated_positions is 0-indexed, so position 0 is residue 1).
ssm = wt.saturation_mutagenesis()
ssm = ProteinSequences([s for s in ssm if s.mutated_positions(wt)[0] != 0])
```

## 2. Structure track — ESM-IF1

```python
from aide_predict import ESMIFLikelihoodWrapper

esm_if = ESMIFLikelihoodWrapper(
    wt=wt,
    marginal_method="wildtype_marginal",  # masked_marginal is refused (autoregressive)
    pool=True,                            # one scalar (log-ratio) per variant
    device="cuda",
)
esm_if.fit()
esm_if_logratio = esm_if.predict(ssm).ravel()
```

## 3. Sequence track — PLM ensemble

The full MULTI-evolve sequence track is five ESM-1v checkpoints plus `esm2_t36_3B_UR50D`. Load each one, score, then free it before the next so peak GPU memory stays bounded. Two checkpoints are shown here for brevity:

```python
import numpy as np
from aide_predict import ESM2LikelihoodWrapper

PLM_CHECKPOINTS = ["esm1v_t33_650M_UR90S_1", "esm2_t36_3B_UR50D"]
plm_logratios = []
for ckpt in PLM_CHECKPOINTS:
    plm = ESM2LikelihoodWrapper(
        wt=wt, model_checkpoint=ckpt,
        marginal_method="wildtype_marginal", pool=True, device="cuda",
    )
    plm.fit()
    plm_logratios.append(plm.predict(ssm).ravel())
    del plm  # free before loading the next checkpoint

plm_logratios = np.vstack(plm_logratios)             # (n_plms, n_variants)
mean_plm_logratio = plm_logratios.mean(axis=0)        # Track A score
plm_pass_count = (plm_logratios > 0).sum(axis=0)      # consensus tier for Seq-FC
```

## 4. Assemble the table and z-score each track

`per_variant_mutation_info` gives the `(wt_aa, mut_aa, position, mutation)` scaffold; `zscore_by_aa_group` adds the AA-grouped z-score column and returns the per-group `(mean, std)` stats it learned.

```python
from aide_predict.utils.scoring import per_variant_mutation_info, zscore_by_aa_group

df = per_variant_mutation_info(ssm, wt)
df["esm_if_logratio"] = esm_if_logratio
df["mean_plm_logratio"] = mean_plm_logratio
df["plm_pass_count"] = plm_pass_count

# Rescale within destination-residue groups (the MULTI-evolve default).
df, _ = zscore_by_aa_group(df, grouping="destination_residue",
                           score_col="esm_if_logratio", out_col="z_esm_if_logratio",
                           min_group_size=5)
df, _ = zscore_by_aa_group(df, grouping="destination_residue",
                           score_col="mean_plm_logratio", out_col="z_mean_plm_logratio",
                           min_group_size=5)
```

## 5. Four-method position-exclusive top-N, then union

```python
import pandas as pd

def position_exclusive_topn(df, sort_cols, ascendings, n):
    """Top-n rows, at most one per residue position, skipping NaN sort keys."""
    used, picked = set(), []
    for _, row in df.sort_values(sort_cols, ascending=ascendings).iterrows():
        if any(pd.isna(row[c]) for c in sort_cols) or row["position"] in used:
            continue
        picked.append(row); used.add(row["position"])
        if len(picked) >= n:
            break
    return pd.DataFrame(picked)

N = 24
methods = {
    "seq_fc":    (["plm_pass_count", "mean_plm_logratio"], [False, False]),  # consensus + magnitude
    "struct_fc": (["esm_if_logratio"], [False]),
    "seq_z":     (["z_mean_plm_logratio"], [False]),
    "struct_z":  (["z_esm_if_logratio"], [False]),
}
picks = {m: position_exclusive_topn(df, cols, asc, N) for m, (cols, asc) in methods.items()}

# Union: one row per nominated mutation, with a per-method indicator column.
nominations = pd.concat(picks.values()).drop_duplicates("mutation")[["mutation"]].copy()
for m, sub in picks.items():
    nominations[f"{m}_picked"] = nominations["mutation"].isin(sub["mutation"]).astype(int)
nominations["n_methods_picked"] = nominations.filter(like="_picked").sum(axis=1)
nominations = nominations.sort_values("n_methods_picked", ascending=False)
```

`nominations` now holds the unioned variant short-list; `n_methods_picked` (1–4) tells you how many of the four rankings agreed on each mutation.

## Reusable composite: `ZScoreRescaledScorer`

The z-score step (steps 2–4 for a single scorer) is also packaged as a scikit-learn-style transformer you can wrap around *any* per-variant scorer. It fits the AA-group stats on a calibration set and applies them at transform time:

```python
from aide_predict import ESMIFLikelihoodWrapper, ZScoreRescaledScorer

inner = ESMIFLikelihoodWrapper(wt=wt, marginal_method="wildtype_marginal", pool=True)
scorer = ZScoreRescaledScorer(inner_scorer=inner, grouping="destination_residue",
                              min_group_size=5, wt=wt)

scorer.fit(ssm)
arr = scorer.transform(ssm)        # (N, 2): columns [logratio, z_logratio]
table = scorer.score_table(ssm)    # rich DataFrame: mutation, position, wt_aa,
                                   # mut_aa, logratio, z_logratio, ...
```

`grouping` switches between `"destination_residue"` (bin by `mut_aa`) and `"substitution_type"` (bin by the `(wt_aa, mut_aa)` pair); `multi_mutant_strategy` controls whether multi-mutation variants are tolerated (`"skip"`, giving them a NaN z-score) or rejected (`"raise"`).

## Multichain complexes

ESM-IF1 can condition the target chain on the rest of a complex. If `my_enzyme.pdb` were a multimer, point the WT at the chain you want to score and let the others become structural context:

```python
wt = ProteinSequence.from_pdb("complex.pdb", chain="A")
wt = wt.with_target_chain("A", auto_context=True)   # other chains -> ESM-IF context
```

For a homo-/hetero-multimer you typically loop the target chain over each protein chain and run the scan once per chain. See [Data Structures](data_structures.md) for `set_target_chain`, `with_target_chain`, and `context_chains`.
