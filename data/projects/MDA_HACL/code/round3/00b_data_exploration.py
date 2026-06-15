#!/usr/bin/env python
"""
00b_data_exploration.py — Exploratory data analysis for MDA HACL Round 3
========================================================================

Reads the cleaned outputs from 00_data_wrangling.py and generates
diagnostic plots to understand activity patterns across mutations,
positions, and substrates.

Outputs figures to results/00_data_exploration/.

Usage:
    python 00b_data_exploration.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "round3" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "00_data_exploration"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
ACTIVE_PALETTE = sns.color_palette("Set2", 6)
SUBSTRATE_ORDER_ACTIVE = [
    "Formaldehyde", "Acetaldehyde", "Acetone",
    "Glycoaldehyde", "Phenylacetaldehyde", "Pyruvate",
]
SUBSTRATE_ORDER_ALL = SUBSTRATE_ORDER_ACTIVE + [
    "Methylglyoxal", "4-Hydroxybutan-2-one", "Glyoxylic acid",
]
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
SUPP_MARKER = " *"  # suffix for supplemental position labels


def load_data():
    """Load all processed data files."""
    form_df = pd.read_csv(PROCESSED_DIR / "formaldehyde_ssm.csv")
    multi_df = pd.read_csv(PROCESSED_DIR / "multi_substrate_ssm.csv")
    with open(PROCESSED_DIR / "substrate_metadata.json") as f:
        metadata = json.load(f)
    return form_df, multi_df, metadata


def get_supplemental_positions(multi_df):
    """Return set of 0-indexed positions that came from supplemental data."""
    if "is_supplemental" not in multi_df.columns:
        return set()
    return set(multi_df.loc[multi_df["is_supplemental"], "position"].unique())


def _style_supp_ticklabels(ax, supp_positions, axis="x"):
    """Bold + color tick labels for supplemental positions."""
    getter = ax.get_xticklabels if axis == "x" else ax.get_yticklabels
    for label in getter():
        text = label.get_text()
        # Extract numeric position from labels like "A555" or "555"
        digits = "".join(c for c in text if c.isdigit())
        if digits and int(digits) in supp_positions:
            label.set_color("#d62728")
            label.set_fontweight("bold")


# =========================================================================
# Plot functions
# =========================================================================

def plot_activity_distributions(multi_df, out_dir):
    """Violin + strip plots of fold-change and log_fc per substrate."""
    active_df = multi_df[multi_df["is_active_substrate"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fold-change (linear scale)
    ax = axes[0]
    sns.violinplot(
        data=active_df, x="substrate", y="fold_change",
        order=SUBSTRATE_ORDER_ACTIVE, palette=ACTIVE_PALETTE,
        inner=None, cut=0, ax=ax, alpha=0.6,
    )
    sns.stripplot(
        data=active_df, x="substrate", y="fold_change",
        order=SUBSTRATE_ORDER_ACTIVE, color="k", size=1.5, alpha=0.3, ax=ax,
    )
    ax.axhline(1.0, color="red", ls="--", lw=0.8, label="WT level")
    ax.set_xlabel("")
    ax.set_ylabel("Fold-change")
    ax.set_title("Activity distribution (linear)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)

    # Log fold-change
    ax = axes[1]
    sns.violinplot(
        data=active_df, x="substrate", y="log_fc",
        order=SUBSTRATE_ORDER_ACTIVE, palette=ACTIVE_PALETTE,
        inner=None, cut=0, ax=ax, alpha=0.6,
    )
    sns.stripplot(
        data=active_df, x="substrate", y="log_fc",
        order=SUBSTRATE_ORDER_ACTIVE, color="k", size=1.5, alpha=0.3, ax=ax,
    )
    ax.axhline(0, color="red", ls="--", lw=0.8, label="WT level")
    ax.set_xlabel("")
    ax.set_ylabel("log₁₀(FC + ε)")
    ax.set_title("Activity distribution (log-transformed)")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)

    fig.suptitle("Multi-substrate activity distributions (active substrates only)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "activity_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved activity_distributions.png")


def plot_zero_fraction(multi_df, out_dir):
    """Bar chart of zero-activity fraction per substrate."""
    summary = []
    for substrate in SUBSTRATE_ORDER_ALL:
        sub = multi_df[multi_df["substrate"] == substrate]
        if len(sub) == 0:
            continue
        n_zero = (sub["fold_change"] == 0).sum()
        n_above_wt = (sub["fold_change"] > 1).sum()
        n_total = len(sub)
        summary.append({
            "substrate": substrate,
            "Zero activity": n_zero / n_total,
            "Sub-WT (0 < FC ≤ 1)": ((sub["fold_change"] > 0) & (sub["fold_change"] <= 1)).sum() / n_total,
            "Above WT (FC > 1)": n_above_wt / n_total,
        })
    summary_df = pd.DataFrame(summary).set_index("substrate")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    summary_df.plot.barh(
        stacked=True, ax=ax,
        color=["#d62728", "#ff9896", "#2ca02c"],
    )
    ax.set_xlabel("Fraction of mutations")
    ax.set_ylabel("")
    ax.set_title("Mutation activity breakdown by substrate")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "zero_fraction_by_substrate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved zero_fraction_by_substrate.png")


def _build_heatmap_matrix(df, value_col="fold_change"):
    """Pivot a substrate's data into a position x AA matrix for heatmapping."""
    # Need 1-indexed position labels for readability
    df = df.copy()
    df["pos_label"] = df["wt_aa"] + df["position"].astype(str)

    pivot = df.pivot_table(
        index="mut_aa", columns="pos_label", values=value_col, aggfunc="first",
    )
    # Reorder rows to standard AA order (only those present)
    row_order = [aa for aa in AA_ORDER if aa in pivot.index]
    pivot = pivot.reindex(row_order)
    return pivot


def plot_mutation_heatmaps_multi(multi_df, out_dir):
    """Heatmap of fold-change per (position, mutant AA) for each active substrate.

    Supplemental positions (from all_data2.xlsx) are highlighted with red bold labels.
    """
    supp_positions = get_supplemental_positions(multi_df)
    active_substrates = [s for s in SUBSTRATE_ORDER_ACTIVE if s != "Pyruvate"]
    n_pos = multi_df["position"].nunique()
    n = len(active_substrates)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, substrate in enumerate(active_substrates):
        sub = multi_df[multi_df["substrate"] == substrate]
        pivot = _build_heatmap_matrix(sub, "log_fc")
        ax = axes[i]
        sns.heatmap(
            pivot, ax=ax, cmap="RdYlGn", center=0,
            vmin=-2.5, vmax=1.2,
            linewidths=0.3, linecolor="white",
            cbar_kws={"label": "log₁₀(FC+ε)", "shrink": 0.7},
        )
        ax.set_title(substrate, fontweight="bold")
        ax.set_xlabel("Position")
        ax.set_ylabel("Mutant AA" if i % 3 == 0 else "")
        _style_supp_ticklabels(ax, supp_positions, axis="x")

    # Pyruvate in last slot
    sub = multi_df[multi_df["substrate"] == "Pyruvate"]
    pivot = _build_heatmap_matrix(sub, "log_fc")
    ax = axes[5]
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", center=0,
        vmin=-2.5, vmax=1.2,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "log₁₀(FC+ε)", "shrink": 0.7},
    )
    ax.set_title("Pyruvate (ref: V83P)", fontweight="bold")
    ax.set_xlabel("Position")
    ax.set_ylabel("")
    _style_supp_ticklabels(ax, supp_positions, axis="x")

    supp_note = (f"  (red labels = {len(supp_positions)} new supplemental positions)"
                 if supp_positions else "")
    fig.suptitle(
        f"Mutation activity heatmaps — {n_pos} positions{supp_note}",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "mutation_heatmaps_multi.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved mutation_heatmaps_multi.png")


def plot_formaldehyde_heatmap(form_df, out_dir):
    """Full 64-position heatmap for formaldehyde."""
    df = form_df.copy()
    df["pos_label"] = df["wt_aa"] + (df["position"]).astype(str)

    # Sort positions numerically
    positions_sorted = sorted(df["position"].unique())
    pos_labels_sorted = []
    for p in positions_sorted:
        row = df[df["position"] == p].iloc[0]
        pos_labels_sorted.append(row["pos_label"])

    pivot = df.pivot_table(
        index="mut_aa", columns="pos_label", values="log_fc", aggfunc="first",
    )
    row_order = [aa for aa in AA_ORDER if aa in pivot.index]
    pivot = pivot.reindex(index=row_order, columns=pos_labels_sorted)

    fig, ax = plt.subplots(figsize=(28, 6))
    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn", center=0,
        vmin=-2.5, vmax=1.0,
        linewidths=0.2, linecolor="white",
        cbar_kws={"label": "log₁₀(FC+ε)", "shrink": 0.5},
    )
    ax.set_title("Formaldehyde SSM — all 64 positions", fontsize=13, fontweight="bold")
    ax.set_xlabel("Position (WT_AA + 0-indexed)")
    ax.set_ylabel("Mutant AA")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "formaldehyde_heatmap_64pos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved formaldehyde_heatmap_64pos.png")


def plot_cross_substrate_correlation(multi_df, out_dir):
    """Pairwise Spearman correlation of activities across substrates at shared positions."""
    active_df = multi_df[multi_df["is_active_substrate"]].copy()

    # Pivot to wide: rows = mutation_string, cols = substrate, values = fold_change
    wide = active_df.pivot_table(
        index="mutation_string", columns="substrate", values="fold_change",
    )
    wide = wide[SUBSTRATE_ORDER_ACTIVE]

    corr = wide.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-0.3, vmax=1, center=0.5,
        square=True, linewidths=0.5,
    )
    n_pos = multi_df["position"].nunique()
    ax.set_title(f"Cross-substrate Spearman correlation\n({n_pos} shared positions)")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_substrate_correlation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cross_substrate_correlation.png")


def plot_cross_substrate_scatter(multi_df, out_dir):
    """Pairwise scatter plots for the 3 most correlated substrate pairs vs formaldehyde."""
    active_df = multi_df[multi_df["is_active_substrate"]].copy()
    wide = active_df.pivot_table(
        index="mutation_string", columns="substrate", values="log_fc",
    )

    pairs = [
        ("Formaldehyde", "Acetaldehyde"),
        ("Formaldehyde", "Acetone"),
        ("Acetaldehyde", "Phenylacetaldehyde"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (s1, s2) in zip(axes, pairs):
        data = wide[[s1, s2]].dropna()
        ax.scatter(data[s1], data[s2], s=15, alpha=0.6, edgecolors="k", linewidths=0.3)
        ax.set_xlabel(f"{s1}\nlog₁₀(FC+ε)")
        ax.set_ylabel(f"{s2}\nlog₁₀(FC+ε)")
        rho = data[s1].corr(data[s2], method="spearman")
        ax.set_title(f"ρ = {rho:.2f}")

        # Identity line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "r--", lw=0.8, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

    fig.suptitle("Cross-substrate activity scatter (log-space)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_substrate_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cross_substrate_scatter.png")


def plot_position_profiles(multi_df, out_dir):
    """Mean activity per position for each active substrate — grouped bar chart.

    Supplemental positions are highlighted with red bold x-axis labels.
    """
    supp_positions = get_supplemental_positions(multi_df)
    active_df = multi_df[multi_df["is_active_substrate"]].copy()

    pos_means = active_df.groupby(["substrate", "position"])["fold_change"].mean().reset_index()
    pos_means["position"] = pos_means["position"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=pos_means, x="position", y="fold_change", hue="substrate",
        hue_order=SUBSTRATE_ORDER_ACTIVE, palette=ACTIVE_PALETTE, ax=ax,
    )
    ax.axhline(1.0, color="red", ls="--", lw=0.8, label="WT level")
    ax.set_xlabel("Position (0-indexed)")
    ax.set_ylabel("Mean fold-change")
    ax.set_title("Mean activity per position across substrates")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    _style_supp_ticklabels(ax, supp_positions, axis="x")
    fig.tight_layout()
    fig.savefig(out_dir / "position_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved position_profiles.png")


def plot_beneficial_overlap(multi_df, out_dir):
    """Heatmap: which mutations are beneficial (FC > 1) on how many substrates?"""
    active_df = multi_df[
        (multi_df["is_active_substrate"]) & (multi_df["substrate"] != "Pyruvate")
    ].copy()

    # Binary beneficial flag per substrate
    beneficial = active_df.pivot_table(
        index="mutation_string", columns="substrate",
        values="fold_change", aggfunc="first",
    )
    beneficial = (beneficial > 1).astype(int)

    # Sort by total number of substrates where beneficial
    beneficial["n_beneficial"] = beneficial.sum(axis=1)
    beneficial = beneficial.sort_values("n_beneficial", ascending=False)

    # Top 30 most broadly beneficial
    top = beneficial.head(30).drop(columns=["n_beneficial"])
    substrates = [s for s in SUBSTRATE_ORDER_ACTIVE if s in top.columns and s != "Pyruvate"]
    top = top[substrates]

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        top, ax=ax, cmap="YlGn", vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Beneficial (FC > 1)", "ticks": [0, 1]},
    )
    ax.set_title("Top 30 most broadly beneficial mutations\n(excluding Pyruvate)")
    ax.set_ylabel("Mutation")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_dir / "beneficial_overlap_top30.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved beneficial_overlap_top30.png")


def plot_formaldehyde_hist(form_df, out_dir):
    """Histogram of formaldehyde fold-change distribution (64 pos)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Linear
    ax = axes[0]
    ax.hist(form_df["fold_change"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(1.0, color="red", ls="--", lw=1, label="WT level")
    ax.set_xlabel("Fold-change")
    ax.set_ylabel("Count")
    ax.set_title("Formaldehyde activity (linear)")
    ax.legend(fontsize=8)

    # Log
    ax = axes[1]
    ax.hist(form_df["log_fc"], bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=1, label="WT level")
    ax.set_xlabel("log₁₀(FC + ε)")
    ax.set_ylabel("Count")
    ax.set_title("Formaldehyde activity (log-transformed)")
    ax.legend(fontsize=8)

    fig.suptitle(f"Formaldehyde SSM — {len(form_df)} mutations at 64 positions", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "formaldehyde_histogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved formaldehyde_histogram.png")


def plot_pyruvate_detail(multi_df, out_dir):
    """Detail plot for pyruvate — the unusual substrate."""
    pyr = multi_df[multi_df["substrate"] == "Pyruvate"].copy()
    pyr_active = pyr[pyr["fold_change"] > 0].sort_values("fold_change", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram
    ax = axes[0]
    ax.hist(pyr["fold_change"], bins=30, color="orange", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Fold-change (relative to V83P)")
    ax.set_ylabel("Count")
    ax.set_title(f"Pyruvate: {len(pyr)} mutations, {len(pyr_active)} active")

    # Bar chart of active mutations
    ax = axes[1]
    if len(pyr_active) > 0:
        ax.barh(
            pyr_active["mutation_string"],
            pyr_active["fold_change"],
            color="orange", edgecolor="k", linewidth=0.5,
        )
        ax.set_xlabel("Fold-change (relative to V83P)")
        ax.set_title("Pyruvate: active mutations")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No active mutations", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)

    fig.suptitle("Pyruvate detail (normalized to V83P, not WT)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "pyruvate_detail.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved pyruvate_detail.png")


def print_summary_table(multi_df, form_df, metadata):
    """Print a text summary table."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"\nFormaldehyde SSM (full): {len(form_df)} mutations, "
          f"{form_df['position'].nunique()} positions")
    print(f"  FC range: [{form_df['fold_change'].min():.3f}, {form_df['fold_change'].max():.3f}]")
    print(f"  Beneficial (FC>1): {(form_df['fold_change'] > 1).sum()}")
    print(f"  Dead (FC=0): {(form_df['fold_change'] == 0).sum()}")

    supp_positions = get_supplemental_positions(multi_df)
    n_orig = multi_df["position"].nunique() - len(supp_positions)
    print(f"\nMulti-substrate SSM: {len(multi_df)} total rows, "
          f"{multi_df['substrate'].nunique()} substrates, "
          f"{multi_df['position'].nunique()} positions "
          f"({n_orig} original + {len(supp_positions)} supplemental)")
    if supp_positions:
        print(f"  Supplemental positions (0-indexed): "
              f"{sorted(supp_positions)}")

    has_supp = "is_supplemental" in multi_df.columns
    print(f"\n{'Substrate':<25s} {'N':>5s} {'Supp':>5s} {'Active':>7s} {'Benef':>6s} "
          f"{'Dead':>5s} {'MaxFC':>7s} {'SMILES'}")
    print("-" * 95)
    for substrate in SUBSTRATE_ORDER_ALL:
        sub = multi_df[multi_df["substrate"] == substrate]
        if len(sub) == 0:
            continue
        m = metadata.get(substrate, {})
        n = len(sub)
        n_supp = int(sub["is_supplemental"].sum()) if has_supp else 0
        n_active = (sub["fold_change"] > 0).sum()
        n_beneficial = (sub["fold_change"] > 1).sum()
        n_dead = (sub["fold_change"] == 0).sum()
        max_fc = sub["fold_change"].max()
        smiles = m.get("smiles", "?")
        print(f"{substrate:<25s} {n:5d} {n_supp:5d} {n_active:7d} {n_beneficial:6d} "
              f"{n_dead:5d} {max_fc:7.2f} {smiles}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("00b_data_exploration.py — MDA HACL Round 3")
    print("=" * 60)

    # Load data
    form_df, multi_df, metadata = load_data()
    print(f"Loaded formaldehyde_ssm.csv: {len(form_df)} rows")
    print(f"Loaded multi_substrate_ssm.csv: {len(multi_df)} rows")
    print(f"Loaded substrate_metadata.json: {len(metadata)} substrates")

    # Summary table
    print_summary_table(multi_df, form_df, metadata)

    # Create output directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\nSaving figures to {RESULTS_DIR}")

    # Generate all plots
    plot_formaldehyde_hist(form_df, RESULTS_DIR)
    plot_formaldehyde_heatmap(form_df, RESULTS_DIR)
    plot_activity_distributions(multi_df, RESULTS_DIR)
    plot_zero_fraction(multi_df, RESULTS_DIR)
    plot_mutation_heatmaps_multi(multi_df, RESULTS_DIR)
    plot_cross_substrate_correlation(multi_df, RESULTS_DIR)
    plot_cross_substrate_scatter(multi_df, RESULTS_DIR)
    plot_position_profiles(multi_df, RESULTS_DIR)
    plot_beneficial_overlap(multi_df, RESULTS_DIR)
    plot_pyruvate_detail(multi_df, RESULTS_DIR)

    print(f"\nDone! {len(list(RESULTS_DIR.glob('*.png')))} figures saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
