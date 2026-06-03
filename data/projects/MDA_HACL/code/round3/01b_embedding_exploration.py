#!/usr/bin/env python
"""
01b_embedding_exploration.py — Diagnostic plots for precomputed embeddings
==========================================================================

Generates 8 figures in results/01_embeddings/ to visualize and validate
the embeddings produced by 01_embeddings.py.

Usage:
    python 01b_embedding_exploration.py
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

import yaml

def load_config() -> dict:
    config_path = SCRIPT_DIR / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Plot 1: ESM2 WT residue PCA
# ---------------------------------------------------------------------------

def plot_esm2_wt_pca(wt_emb: np.ndarray, wt_seq: str, output_dir: Path):
    """PCA of WT per-residue embeddings colored by position and AA identity."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(wt_emb)  # (565, 2)
    var_explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Colored by position
    ax = axes[0]
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=np.arange(len(wt_seq)), cmap="viridis", s=10, alpha=0.7,
    )
    plt.colorbar(sc, ax=ax, label="Residue index (0-indexed)")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_title("ESM2 WT residue embeddings — colored by position")

    # (b) Colored by AA identity
    ax = axes[1]
    aa_list = sorted(set(wt_seq))
    aa_to_int = {aa: i for i, aa in enumerate(aa_list)}
    aa_ints = np.array([aa_to_int[aa] for aa in wt_seq])
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=aa_ints, cmap="tab20", s=10, alpha=0.7,
    )
    # Legend with AA labels
    handles = []
    for aa in aa_list:
        idx = aa_to_int[aa]
        color = plt.cm.tab20(idx / max(len(aa_list) - 1, 1))
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=6, label=aa))
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
              ncol=2, fontsize=7, title="AA")
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_title("ESM2 WT residue embeddings — colored by AA")

    fig.suptitle("ESM2 WT Per-Residue Embeddings (PCA)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "esm2_wt_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved esm2_wt_pca.png")


# ---------------------------------------------------------------------------
# Plot 2: ESM2 mutation effect (L2 distance WT vs mutant)
# ---------------------------------------------------------------------------

def plot_esm2_mutation_effect(
    wt_emb: np.ndarray,
    mutant_embs: dict,
    form_df: pd.DataFrame,
    output_dir: Path,
):
    """Distribution of L2 distance between ESM_wt and ESM_mut at each position."""
    records = []
    for _, row in form_df.iterrows():
        mut_str = row["mutation_string"]
        pos = int(row["position"])
        if mut_str not in mutant_embs:
            continue
        wt_vec = wt_emb[pos]
        mut_vec = mutant_embs[mut_str]
        dist = np.linalg.norm(mut_vec - wt_vec)
        is_synonymous = row["wt_aa"] == row["mut_aa"]
        records.append({
            "position": pos,
            "mutation_string": mut_str,
            "l2_distance": dist,
            "is_synonymous": is_synonymous,
            "fold_change": row["fold_change"],
        })

    dist_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # (a) Boxplot by position (top 20 positions by variance)
    ax = axes[0]
    pos_var = dist_df.groupby("position")["l2_distance"].std().sort_values(ascending=False)
    top_positions = pos_var.head(20).index
    subset = dist_df[dist_df["position"].isin(top_positions)]
    sns.boxplot(data=subset, x="position", y="l2_distance", ax=ax,
                order=sorted(top_positions), fliersize=2)
    ax.set_xlabel("Position (0-indexed)")
    ax.set_ylabel("L2 distance (ESM mut − ESM wt)")
    ax.set_title("Mutation perturbation by position (top 20 most variable)")
    ax.tick_params(axis='x', rotation=45)

    # (b) Scatter: L2 distance vs fold change
    ax = axes[1]
    non_syn = dist_df[~dist_df["is_synonymous"]]
    ax.scatter(non_syn["l2_distance"], non_syn["fold_change"], s=5, alpha=0.3)
    rho, pval = stats.spearmanr(non_syn["l2_distance"], non_syn["fold_change"])
    ax.set_xlabel("L2 distance (ESM mut − ESM wt)")
    ax.set_ylabel("Fold change")
    ax.set_title(f"Mutation perturbation vs activity (Spearman ρ={rho:.3f})")

    fig.suptitle("ESM2 Mutation Effect Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "esm2_mutation_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved esm2_mutation_effect.png")


# ---------------------------------------------------------------------------
# Plot 3: SaProt vs formaldehyde activity
# ---------------------------------------------------------------------------

def plot_saprot_vs_activity(
    saprot_scores: dict,
    form_df: pd.DataFrame,
    output_dir: Path,
):
    """Scatter: SaProt zero-shot score vs formaldehyde log_fc."""
    records = []
    for _, row in form_df.iterrows():
        mut_str = row["mutation_string"]
        if mut_str in saprot_scores:
            records.append({
                "saprot_score": saprot_scores[mut_str],
                "log_fc": row["log_fc"],
                "fold_change": row["fold_change"],
                "is_synonymous": row["wt_aa"] == row["mut_aa"],
            })

    df = pd.DataFrame(records)
    non_syn = df[~df["is_synonymous"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) SaProt vs log_fc
    ax = axes[0]
    ax.scatter(non_syn["saprot_score"], non_syn["log_fc"], s=8, alpha=0.4)
    rho, pval = stats.spearmanr(non_syn["saprot_score"], non_syn["log_fc"])
    # Fit line
    z = np.polyfit(non_syn["saprot_score"], non_syn["log_fc"], 1)
    x_range = np.linspace(non_syn["saprot_score"].min(), non_syn["saprot_score"].max(), 100)
    ax.plot(x_range, np.polyval(z, x_range), "r--", alpha=0.7, label=f"Linear fit")
    ax.set_xlabel("SaProt zero-shot score")
    ax.set_ylabel("log₁₀(FC + ε)")
    ax.set_title(f"SaProt vs Formaldehyde Activity\n(Spearman ρ={rho:.3f}, p={pval:.1e})")
    ax.legend()

    # (b) SaProt vs raw FC
    ax = axes[1]
    ax.scatter(non_syn["saprot_score"], non_syn["fold_change"], s=8, alpha=0.4)
    rho2, pval2 = stats.spearmanr(non_syn["saprot_score"], non_syn["fold_change"])
    ax.set_xlabel("SaProt zero-shot score")
    ax.set_ylabel("Fold change")
    ax.set_title(f"SaProt vs Raw FC (Spearman ρ={rho2:.3f})")

    fig.suptitle("SaProt Zero-Shot Predictive Power", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "saprot_vs_activity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved saprot_vs_activity.png")


# ---------------------------------------------------------------------------
# Plot 4: SaProt score distribution
# ---------------------------------------------------------------------------

def plot_saprot_distribution(
    saprot_scores: dict,
    form_df: pd.DataFrame,
    output_dir: Path,
):
    """Histogram of SaProt scores split by active vs dead mutations."""
    records = []
    for _, row in form_df.iterrows():
        mut_str = row["mutation_string"]
        if mut_str in saprot_scores and row["wt_aa"] != row["mut_aa"]:
            records.append({
                "saprot_score": saprot_scores[mut_str],
                "category": "Active (FC > 0.1)" if row["fold_change"] > 0.1
                            else "Dead/Low (FC ≤ 0.1)",
            })

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cat, color in [("Active (FC > 0.1)", "steelblue"), ("Dead/Low (FC ≤ 0.1)", "coral")]:
        subset = df[df["category"] == cat]
        ax.hist(subset["saprot_score"], bins=40, alpha=0.6, label=f"{cat} (n={len(subset)})",
                color=color, density=True)

    ax.set_xlabel("SaProt zero-shot score")
    ax.set_ylabel("Density")
    ax.set_title("SaProt Score Distribution: Active vs Dead Mutations")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "saprot_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved saprot_distribution.png")


# ---------------------------------------------------------------------------
# Plot 5: Substrate embedding PCA
# ---------------------------------------------------------------------------

def plot_substrate_embedding_pca(
    morgan: np.ndarray,
    maccs: np.ndarray,
    mordred: np.ndarray,
    substrate_names: list,
    output_dir: Path,
    molformer: Optional[np.ndarray] = None,
):
    """PCA of substrate embeddings for each type."""
    panels = [
        (morgan, f"Morgan ({morgan.shape[1]} bits)"),
        (maccs, f"MACCS ({maccs.shape[1]} bits)"),
        (mordred, f"Mordred ({mordred.shape[1]} features)"),
    ]
    if molformer is not None:
        panels.append((molformer, f"MoLFormer ({molformer.shape[1]}-dim)"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (emb, title) in zip(axes, panels):
        n_components = min(2, emb.shape[0], emb.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(emb)

        ax.scatter(coords[:, 0], coords[:, 1] if n_components > 1 else np.zeros(len(coords)),
                   s=80, zorder=5)
        for i, name in enumerate(substrate_names):
            y = coords[i, 1] if n_components > 1 else 0
            ax.annotate(name, (coords[i, 0], y), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        if n_components > 1:
            ax.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax.set_title(title)

    fig.suptitle("Substrate Embedding PCA", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "substrate_embedding_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved substrate_embedding_pca.png")


# ---------------------------------------------------------------------------
# Plot 6: Substrate pairwise distance heatmaps
# ---------------------------------------------------------------------------

def plot_substrate_pairwise_dist(
    morgan: np.ndarray,
    maccs: np.ndarray,
    mordred: np.ndarray,
    substrate_names: list,
    output_dir: Path,
    molformer: Optional[np.ndarray] = None,
):
    """Heatmap of pairwise distances between substrate embeddings."""
    panels = [
        (morgan, "Morgan"),
        (maccs, "MACCS"),
        (mordred, "Mordred"),
    ]
    if molformer is not None:
        panels.append((molformer, "MoLFormer"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (emb, title) in zip(axes, panels):
        dist = pairwise_distances(emb, metric="euclidean")
        # Normalize for visualization
        if dist.max() > 0:
            dist_norm = dist / dist.max()
        else:
            dist_norm = dist

        sns.heatmap(
            dist_norm, ax=ax, xticklabels=substrate_names, yticklabels=substrate_names,
            cmap="YlOrRd", annot=True, fmt=".2f", square=True,
            cbar_kws={"label": "Normalized distance"},
        )
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    fig.suptitle("Substrate Pairwise Distances", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "substrate_pairwise_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved substrate_pairwise_dist.png")


# ---------------------------------------------------------------------------
# Plot 7: Feature dimension summary
# ---------------------------------------------------------------------------

def plot_feature_dim_summary(
    esm_dim: int,
    morgan_dim: int,
    maccs_dim: int,
    mordred_dim: int,
    output_dir: Path,
    molformer_dim: Optional[int] = None,
):
    """Bar chart of feature dimensions for each feature group."""
    features = {
        "ESM_wt": esm_dim,
        "ESM_mut": esm_dim,
        "SaProt_zs": 1,
        "FC_ref": 1,
        "Morgan": morgan_dim,
        "MACCS": maccs_dim,
        "Mordred": mordred_dim,
    }
    colors = ["#4C72B0", "#4C72B0", "#55A868", "#DD8452",
              "#C44E52", "#C44E52", "#C44E52"]
    if molformer_dim is not None:
        features["MoLFormer"] = molformer_dim
        colors.append("#8172B2")

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(features.keys())
    dims = list(features.values())

    bars = ax.bar(names, dims, color=colors, edgecolor="white", linewidth=0.5)
    for bar, dim in zip(bars, dims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(dim), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Feature dimensions")
    ax.set_title("Feature Group Dimensions for BNN2")
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(dims) * 2)

    # Category labels
    ax.axhline(y=0.7, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_dim_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved feature_dim_summary.png")


# ---------------------------------------------------------------------------
# Plot 8: ESM2 per-position variance
# ---------------------------------------------------------------------------

def plot_esm2_position_variance(
    wt_emb: np.ndarray,
    mutant_embs: dict,
    form_df: pd.DataFrame,
    output_dir: Path,
):
    """Per-position variance of mutant ESM2 embeddings overlaid with mean FC."""
    # Group mutant embeddings by position
    pos_embeddings = {}
    for _, row in form_df.iterrows():
        mut_str = row["mutation_string"]
        pos = int(row["position"])
        if mut_str in mutant_embs:
            if pos not in pos_embeddings:
                pos_embeddings[pos] = []
            pos_embeddings[pos].append(mutant_embs[mut_str])

    positions = sorted(pos_embeddings.keys())
    mean_variance = []
    for pos in positions:
        embs = np.stack(pos_embeddings[pos])
        # Mean variance across embedding dimensions
        mean_variance.append(embs.var(axis=0).mean())

    # Mean FC per position
    mean_fc = form_df.groupby("position")["fold_change"].mean()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    color1 = "steelblue"
    color2 = "coral"

    ax1.bar(range(len(positions)), mean_variance, color=color1, alpha=0.7, label="ESM2 variance")
    ax1.set_xlabel("Position index")
    ax1.set_ylabel("Mean embedding variance", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Set x-tick labels to position indices (every 5th)
    ax1.set_xticks(range(0, len(positions), 5))
    ax1.set_xticklabels([positions[i] for i in range(0, len(positions), 5)], fontsize=7)

    ax2 = ax1.twinx()
    fc_values = [mean_fc.get(pos, 0) for pos in positions]
    ax2.plot(range(len(positions)), fc_values, color=color2, linewidth=2,
             marker="o", markersize=3, label="Mean FC")
    ax2.set_ylabel("Mean fold change", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    rho, _ = stats.spearmanr(mean_variance, fc_values)
    ax1.set_title(f"Per-Position ESM2 Embedding Variance vs Mean Activity "
                  f"(Spearman ρ={rho:.3f})")

    fig.tight_layout()
    fig.savefig(output_dir / "esm2_position_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved esm2_position_variance.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    emb_dir = processed_dir / "embeddings"

    output_dir = PROJECT_ROOT / "results" / "01_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    form_df = pd.read_csv(processed_dir / "formaldehyde_ssm.csv")
    wt_fasta = processed_dir / "wt_sequence.fasta"
    lines = wt_fasta.read_text().strip().split("\n")
    wt_seq = "".join(line.strip() for line in lines if not line.startswith(">"))

    # Load embeddings
    print("\nLoading embeddings...")

    wt_emb = np.load(emb_dir / "esm2_wt_residues.npy")
    print(f"  ESM2 WT: {wt_emb.shape}")

    mutant_data = np.load(emb_dir / "esm2_mutant_residues.npz")
    mutant_embs = {k: mutant_data[k] for k in mutant_data.files}
    print(f"  ESM2 mutants: {len(mutant_embs)} mutations")

    saprot_scores = None
    saprot_path = emb_dir / "saprot_scores.json"
    if saprot_path.exists():
        with open(saprot_path) as f:
            saprot_scores = json.load(f)
        print(f"  SaProt: {len(saprot_scores)} scores")

    morgan = np.load(emb_dir / "substrate_morgan.npy")
    maccs = np.load(emb_dir / "substrate_maccs.npy")
    mordred = np.load(emb_dir / "substrate_mordred.npy")
    molformer = None
    molformer_path = emb_dir / "substrate_molformer.npy"
    if molformer_path.exists():
        molformer = np.load(molformer_path)
    with open(emb_dir / "substrate_names.json") as f:
        substrate_names = json.load(f)
    sub_info = f"Morgan {morgan.shape}, MACCS {maccs.shape}, Mordred {mordred.shape}"
    if molformer is not None:
        sub_info += f", MoLFormer {molformer.shape}"
    print(f"  Substrates: {sub_info}")

    # Generate plots
    print("\nGenerating plots...")

    # 1. ESM2 WT PCA
    plot_esm2_wt_pca(wt_emb, wt_seq, output_dir)

    # 2. ESM2 mutation effect
    plot_esm2_mutation_effect(wt_emb, mutant_embs, form_df, output_dir)

    # 3 & 4. SaProt plots (if available)
    if saprot_scores:
        plot_saprot_vs_activity(saprot_scores, form_df, output_dir)
        plot_saprot_distribution(saprot_scores, form_df, output_dir)
    else:
        print("  Skipping SaProt plots (no scores available)")

    # 5. Substrate PCA
    plot_substrate_embedding_pca(morgan, maccs, mordred, substrate_names, output_dir,
                                 molformer=molformer)

    # 6. Substrate pairwise distances
    plot_substrate_pairwise_dist(morgan, maccs, mordred, substrate_names, output_dir,
                                 molformer=molformer)

    # 7. Feature dimension summary
    esm_dim = int(wt_emb.shape[1])
    molformer_dim = int(molformer.shape[1]) if molformer is not None else None
    plot_feature_dim_summary(esm_dim, morgan.shape[1], maccs.shape[1],
                             mordred.shape[1], output_dir,
                             molformer_dim=molformer_dim)

    # 8. ESM2 position variance
    plot_esm2_position_variance(wt_emb, mutant_embs, form_df, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
