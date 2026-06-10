"""
Walkthrough: ESM-IF SSM scores with AA-grouped z-score rescaling.

Wraps an ESM-IF wildtype_marginal scorer with ZScoreRescaledScorer, runs the
SSM of a small protein (ENVZ_ECOLI from tests/data, 60 residues), and prints
the per-mutation table with both raw log-ratios and destination-AA z-scores.

Designed to be stepped through in a debugger — single linear function.

Run::

    python scripts/zscore_ssm_demo.py

Output::

    scripts/zscore_ssm_demo.csv          (per-mutation table)
    scripts/zscore_ssm_demo_plots/*.png  (distribution + SSM plots)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

os.environ.setdefault("KEEP_MODEL_ON_DEVICE", "1")

from aide_predict.utils.data_structures import (
    ProteinSequence,
    ProteinSequences,
    ProteinStructure,
)
from aide_predict.bespoke_models.predictors.esm_if import ESMIFLikelihoodWrapper
from aide_predict.bespoke_models.composites.zscore import ZScoreRescaledScorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PDB = REPO_ROOT / "tests" / "data" / "ENVZ_ECOLI.pdb"
WT_SEQ = "LADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEECNAIIEQFIDYLR"
OUT_CSV = HERE / "zscore_ssm_demo.csv"
PLOT_DIR = HERE / "zscore_ssm_demo_plots"

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def _save(fig, name):
    PLOT_DIR.mkdir(exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def make_plots(df, wt_seq):
    """Render distribution + SSM-style plots of the rescaled SSM."""
    sns.set_theme(context="notebook", style="whitegrid")

    # 1. Overlaid histograms of raw vs z-scored values.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["logratio"], bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="black", lw=0.8, ls="--")
    axes[0].set(title="Raw log-ratio", xlabel="logratio", ylabel="count")
    axes[1].hist(df["z_logratio"], bins=40, color="darkorange", edgecolor="white")
    axes[1].axvline(0, color="black", lw=0.8, ls="--")
    axes[1].set(title="Z-rescaled log-ratio (by destination AA)",
                xlabel="z_logratio", ylabel="count")
    fig.suptitle("SSM score distributions: before vs after z-score rescaling")
    fig.tight_layout()
    _save(fig, "01_distribution_raw_vs_zscore.png")

    # 2. Per-destination-AA violins — raw scale shows why rescaling is needed
    # (AA-specific offsets/spreads), z scale collapses them to a common frame.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    sns.violinplot(data=df, x="mut_aa", y="logratio", order=AA_ORDER,
                   inner="quartile", ax=axes[0], color="steelblue")
    axes[0].axhline(0, color="black", lw=0.6, ls="--")
    axes[0].set(title="Raw log-ratio by destination AA", xlabel="", ylabel="logratio")
    sns.violinplot(data=df, x="mut_aa", y="z_logratio", order=AA_ORDER,
                   inner="quartile", ax=axes[1], color="darkorange")
    axes[1].axhline(0, color="black", lw=0.6, ls="--")
    axes[1].set(title="Z-rescaled log-ratio by destination AA",
                xlabel="destination AA", ylabel="z_logratio")
    fig.tight_layout()
    _save(fig, "02_per_destination_aa_violins.png")

    # 3. Scatter of raw vs z, colored by destination AA — visualizes how each
    # AA group's offset+scale gets removed.
    fig, ax = plt.subplots(figsize=(7, 6))
    palette = sns.color_palette("husl", n_colors=len(AA_ORDER))
    for aa, color in zip(AA_ORDER, palette):
        sub = df[df["mut_aa"] == aa]
        ax.scatter(sub["logratio"], sub["z_logratio"], s=14, alpha=0.7,
                   color=color, label=aa)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set(title="Raw vs z-rescaled log-ratio (color = destination AA)",
           xlabel="logratio", ylabel="z_logratio")
    ax.legend(ncol=2, fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    _save(fig, "03_raw_vs_zscore_scatter.png")

    # 4. Classic SSM heatmaps (position x destination AA). Two panels — raw
    # then z — share the same layout so they're directly comparable.
    pivot_raw = df.pivot(index="mut_aa", columns="position", values="logratio") \
                  .reindex(AA_ORDER)
    pivot_z = df.pivot(index="mut_aa", columns="position", values="z_logratio") \
                .reindex(AA_ORDER)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, 0.18 * len(wt_seq)), 9),
                             sharex=True)
    vmax_raw = np.nanmax(np.abs(pivot_raw.values))
    sns.heatmap(pivot_raw, ax=axes[0], cmap="RdBu_r", center=0,
                vmin=-vmax_raw, vmax=vmax_raw, yticklabels=AA_ORDER,
                cbar_kws={"label": "logratio"})
    axes[0].set(title="Raw log-ratio SSM", ylabel="destination AA", xlabel="")
    vmax_z = np.nanmax(np.abs(pivot_z.values))
    sns.heatmap(pivot_z, ax=axes[1], cmap="RdBu_r", center=0,
                vmin=-vmax_z, vmax=vmax_z, yticklabels=AA_ORDER,
                cbar_kws={"label": "z_logratio"})
    axes[1].set(title="Z-rescaled SSM (by destination AA)",
                ylabel="destination AA", xlabel="position")
    for ax in axes:
        ax.set_yticklabels(AA_ORDER, rotation=0, fontsize=8)

    # Tick every 5 residues with the WT identity, to keep the x-axis readable.
    positions = sorted(df["position"].unique())
    tick_idx = [i for i, p in enumerate(positions) if (p - 1) % 5 == 0]
    tick_labels = [f"{wt_seq[positions[i] - 1]}{positions[i]}" for i in tick_idx]
    for ax in axes:
        ax.set_xticks([i + 0.5 for i in tick_idx])
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    fig.tight_layout()
    _save(fig, "04_ssm_heatmaps.png")

    # 5. Per-position summary: best (max z) and mean z. Highlights "hotspots"
    # — positions that tolerate or prefer substitutions on the rescaled scale.
    per_pos = df.groupby("position")["z_logratio"].agg(["max", "mean", "min"])
    fig, ax = plt.subplots(figsize=(max(12, 0.18 * len(wt_seq)), 4))
    ax.plot(per_pos.index, per_pos["max"], label="max z", color="darkorange")
    ax.plot(per_pos.index, per_pos["mean"], label="mean z", color="steelblue")
    ax.plot(per_pos.index, per_pos["min"], label="min z", color="firebrick")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set(title="Per-position z_logratio summary",
           xlabel="position", ylabel="z_logratio")
    ax.legend()
    fig.tight_layout()
    _save(fig, "05_per_position_summary.png")


def main():
    # 1. Build WT with structure attached.
    structure = ProteinStructure(str(PDB))
    wt = ProteinSequence(WT_SEQ, structure=structure, id="ENVZ_WT")
    print(f"WT length: {len(wt)}  (chains: {structure.get_all_chain_ids()})")

    # 2. Build the inner scorer (ESM-IF wildtype_marginal, pool=True → one
    # scalar per variant). Any per-variant-scalar scorer works here; ESM-IF
    # was chosen because it's tested.
    inner = ESMIFLikelihoodWrapper(
        marginal_method="wildtype_marginal",
        wt=wt,
        pool=True,
        device=DEVICE,
        metadata_folder=str(HERE / "tmp_esm_if"),
    )

    # 3. Wrap with the z-score rescaler. Default grouping is by destination
    # residue (→P bins). Switch via grouping='substitution_type' for A→P bins.
    composite = ZScoreRescaledScorer(
        inner_scorer=inner,
        grouping="destination_residue",
        wt=wt,
    )

    # 4. Generate the full SSM and score it in one fit_transform-like call.
    # composite.fit(ssm)         → runs inner.transform(ssm) once, learns
    #                              group stats from the SSM distribution.
    # composite.score_table(ssm) → returns the rich DataFrame.
    ssm = wt.saturation_mutagenesis()
    print(f"Generated {len(ssm)} SSM variants ({len(WT_SEQ)} positions x 19 AAs)")

    composite.fit(ssm)
    df = composite.score_table(ssm)
    print(f"\nGroup stats keys (destination AAs): "
          f"{sorted(g[0] for g in composite.group_stats_.keys())}")

    # 5. Save and summarize.
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")

    print("\nTop 5 by z_logratio (most favored under destination-AA rescaling):")
    print(df.sort_values("z_logratio", ascending=False)
            .head()[["mutation", "logratio", "z_logratio"]]
            .to_string(index=False))

    print("\nTop 5 by raw logratio (most favored under absolute log-ratio):")
    print(df.sort_values("logratio", ascending=False)
            .head()[["mutation", "logratio", "z_logratio"]]
            .to_string(index=False))

    # Sanity: within each destination group the z-scores have mean ≈ 0, std ≈ 1.
    print("\nPer-destination-AA z-score sanity (should be ~0 / ~1):")
    summary = df.groupby("mut_aa")["z_logratio"].agg(["mean", "std", "count"])
    print(summary.round(4).to_string())

    # 6. Plots.
    print(f"\nWriting plots to {PLOT_DIR}/ ...")
    make_plots(df, WT_SEQ)


if __name__ == "__main__":
    main()
