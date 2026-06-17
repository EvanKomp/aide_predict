#!/usr/bin/env python
"""new_05d_bnn2_deploy_select.py — deploy the locked-in BNN2 + null models to
SELECT single-point mutants to test against genuinely NEW substrates.

The new substrates are SMILES-only candidates NOT in the dataset (zero data) —
e.g. data/round3/new_substrates.json + data/round3/extra_substrates_smiles.json.
Their substrate embeddings are computed from SMILES on the fly. The 200 SSM
single-point mutants (10 library positions) are the design space.

Selectors (per substrate):
  - 4 NULL modes (formaldehyde, nearest, avg_all, distance_weighted).
  - 4 BNN endpoints (global NDCG, global top-3, per-position top-3,
    per-position NDCG), each at its LOCKED-IN-CV winning reference mode + that
    mode's recorded best_beta (read from scoring/metrics.json, NOT the
    optimistic hyperopt best_reference_mode), PLUS an "explore" variant at
    beta + --explore-beta-bump (default 1.0) to surface higher-uncertainty picks.
  UCB score = y_pred + beta * acq_std, acq_sigma=within_epi_ale.

References = "all" by default (config bnn2.pairwise.ref_substrates), matching how
the models were trained/CV'd: the 9 existing substrates (incl. the 3 inactive
ones at the -2 floor) serve as references; the new substrates auto-drop (no
fc_ref). null(nearest) is flagged degenerate when its nearest of the 9 is
floor-dominated.

Outputs (under --output-dir): selections.csv, all_predictions.csv,
mutation_consensus.csv, and figures/ (agreement heatmap, selector overlap,
per-position consensus, score distribution, BNN-vs-null overlap).

Reuses 06_predict.py inference primitives + 05_bnn2_common.py +
new_05b_bnn2_score.aggregate_and_null_per_fold so deployment matches the CV
pipeline exactly.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from importlib.util import spec_from_file_location, module_from_spec

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # data/projects/MDA_HACL
sys.path.insert(0, str(SCRIPT_DIR.parent))  # for `from bnns import ...`

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("deploy_select")


def _load_module(name: str, filename: str):
    spec = spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_common = _load_module("bnn2_common", "05_bnn2_common.py")
_predict = _load_module("bnn2_predict", "06_predict.py")
_score = _load_module("bnn2_score", "new_05b_bnn2_score.py")

# Reused primitives
load_config = _common.load_config
load_all_embeddings = _common.load_all_embeddings
load_substrate_metadata = _common.load_substrate_metadata
load_multi_substrate_data = _common.load_multi_substrate_data
build_other_features = _common.build_other_features
build_bnn2_model = _common.build_bnn2_model
add_ref_distances = _common.add_ref_distances
compute_pairwise_distances = _common.compute_pairwise_distances
compute_null_for_mode = _common.compute_null_for_mode
FORMALDEHYDE_SUBSTRATE = _common.FORMALDEHYDE_SUBSTRATE

parse_mutation_string = _predict.parse_mutation_string
load_trained_model = _predict.load_trained_model
load_preprocessing_pipelines = _predict.load_preprocessing_pipelines
apply_saved_preprocessing = _predict.apply_saved_preprocessing
build_fc_lookup = _predict.build_fc_lookup
build_inference_df = _predict.build_inference_df
parse_substrate_smiles = _predict.parse_substrate_smiles
compute_substrate_maccs = _predict.compute_substrate_maccs
compute_substrate_molformer = _predict.compute_substrate_molformer
compute_substrate_mordred = _predict.compute_substrate_mordred
compute_substrate_morgan = _predict.compute_substrate_morgan

aggregate_and_null_per_fold = _score.aggregate_and_null_per_fold

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
MODES = ["formaldehyde", "nearest", "avg_all", "distance_weighted"]
FLOOR_THRESHOLD = -1.99  # log_fc at/below this counts as "floor" (inactive)

# endpoint dir name -> (short label, target ranking-metric key in bnn_at_best_beta)
ENDPOINTS = {
    "ndcg": ("ndcg", "ndcg"),
    "top3_recovery": ("top3", "top3_recovery"),
    "per_position_top3_recovery_mean": ("pp_top3", "per_position_top3_recovery_mean"),
    "per_position_ndcg_mean": ("pp_ndcg", "per_position_ndcg_mean"),
}

_PAIRWISE_META_COLS = [
    "mutation_string", "position", "wt_aa", "mut_aa",
    "substrate", "ref_substrate",
    "fold_change", "log_fc", "fc_ref", "log_fc_ref",
    "is_active_substrate", "ref_type",
]


# ════════════════════════════════════════════════════════════════════════════
# New-substrate SMILES embedding injection (multi-type, single pass)
# ════════════════════════════════════════════════════════════════════════════
def inject_new_substrates_multi(embeddings: dict, substrate_meta: dict,
                                new_smiles: dict, types: List[str],
                                config: dict) -> List[str]:
    """Compute fingerprints of EVERY requested embedding type for the truly-new
    substrates and inject them (plus names/metadata) once. (06_predict's
    inject_new_substrates skips on the 2nd type because the names already exist,
    so we roll our own to cover maccs/molformer/mordred together.)"""
    existing = set(embeddings["substrate_names"])
    truly_new = {k: v for k, v in new_smiles.items() if k not in existing}
    if not truly_new:
        logger.info("No new substrates to inject (all already present).")
        return []
    names = list(truly_new.keys())
    emb_cfg = config.get("embeddings", {})

    for t in types:
        key = f"substrate_{t}"
        if t == "morgan":
            fps = compute_substrate_morgan(
                truly_new, radius=emb_cfg.get("morgan_radius", 2),
                n_bits=emb_cfg.get("morgan_bits", 2048))
        elif t == "maccs":
            fps = compute_substrate_maccs(truly_new)
        elif t == "mordred":
            fnp = (PROJECT_ROOT / config["data"]["output_dir"]
                   / "embeddings" / "mordred_feature_names.json")
            existing_names = json.load(open(fnp)) if fnp.exists() else None
            fps, _ = compute_substrate_mordred(truly_new, existing_feature_names=existing_names)
        elif t == "molformer":
            fps = compute_substrate_molformer(truly_new)
        else:
            raise ValueError(f"unsupported embedding type {t}")
        if key not in embeddings:
            raise KeyError(f"embeddings missing {key}")
        embeddings[key] = np.concatenate(
            [embeddings[key], fps.astype(np.float32)], axis=0)
        logger.info("  injected %d %s embeddings (dim=%d)", len(names), t, fps.shape[1])

    for n in names:
        embeddings["substrate_names"].append(n)
        substrate_meta[n] = {"name": n, "smiles": truly_new[n],
                             "is_active": False, "ref_type": "wt"}
    logger.info("Injected %d new substrates: %s", len(names), names)
    return names


# ════════════════════════════════════════════════════════════════════════════
# Endpoint config from the LOCKED-IN substrate-split CV
# ════════════════════════════════════════════════════════════════════════════
def pick_cv_winner(run_dir: Path, tkey: str) -> dict:
    """Pick the reference mode that scored highest on the locked-in CV for this
    endpoint's target metric, at its recorded best_beta. Returns dict with
    mode, beta, score, runner_up, gap, near_tie."""
    with open(run_dir / "scoring" / "metrics.json") as f:
        mj = json.load(f)
    modes = mj.get("modes", {})
    scored = []
    for m in MODES:  # MODES order makes ties resolve toward avg_all before dw
        if m not in modes:
            continue
        mm = modes[m]["metrics"]
        val = (mm.get("bnn_at_best_beta") or {}).get(tkey)
        if val is None:
            continue
        scored.append((m, float(val), float(mm.get("best_beta", 0.0) or 0.0)))
    if not scored:
        raise ValueError(f"no bnn_at_best_beta[{tkey}] in {run_dir}")
    # strict argmax; ties keep the earlier (avg_all before distance_weighted)
    best = scored[0]
    for s in scored[1:]:
        if s[1] > best[1]:
            best = s
    runners = sorted([s for s in scored if s[0] != best[0]], key=lambda s: -s[1])
    gap = best[1] - runners[0][1] if runners else float("inf")
    return {"mode": best[0], "beta": best[2], "score": best[1],
            "runner_up": runners[0][0] if runners else None,
            "gap": gap, "near_tie": gap < 0.005}


# ════════════════════════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════════════════════════
def _hidden_dims_from_state(sd) -> List[int]:
    """Recover BNN2 head hidden-layer widths from the checkpoint (ground truth),
    in layer order. The saved hyperparams.json hidden_dims can be stale vs the
    actual final_model.pt for some locked-in endpoints (ndcg, pp_ndcg)."""
    layers = []
    for k in sd:
        m = re.match(r"bnn2_hidden\.(\d+)\.weight_mu$", k)
        if m:
            layers.append((int(m.group(1)), int(sd[k].shape[0])))
    return [w for _, w in sorted(layers)]


def load_model_from_checkpoint(models_dir: Path, device: str):
    """Load the final model, reconstructing architecture from the checkpoint
    weights (NOT hyperparams.json, which can be stale). Endpoints here are all
    x_aa=off, so no BNN1 backbone is needed."""
    with open(models_dir / "model_metadata.json") as f:
        metadata = json.load(f)
    with open(models_dir.parent / "hyperparams.json") as f:
        params = json.load(f)
    ckpt = torch.load(models_dir / "final_model.pt", map_location=device,
                      weights_only=False)
    sd = ckpt["model_state_dict"]
    ckpt_hd = _hidden_dims_from_state(sd)
    if ckpt_hd and ckpt_hd != params.get("hidden_dims"):
        logger.warning("  %s: checkpoint hidden_dims=%s != hyperparams.json %s "
                       "— using the checkpoint (model weights are ground truth)",
                       models_dir.parent.name, ckpt_hd, params.get("hidden_dims"))
    params = dict(params)
    params["hidden_dims"] = ckpt_hd
    if (params.get("features") or {}).get("x_aa", False):
        raise NotImplementedError("deploy script assumes x_aa=off (no BNN1)")
    model = build_bnn2_model(None, 0, 0, metadata["other_feature_dim"], params, device)
    model.load_state_dict(sd)
    logger.info("  loaded %s (hidden_dims=%s, other_feature_dim=%d)",
                models_dir.parent.name, ckpt_hd, metadata["other_feature_dim"])
    return model, params, metadata


def _predict_absolute(model, X, log_fc_ref, n_samples, prediction_floor,
                      device, batch=4096):
    """Batched MC inference returning absolute (delta + log_fc_ref) mean + stds."""
    means, epis, ales, tots = [], [], [], []
    for s in range(0, len(X), batch):
        e = min(s + batch, len(X))
        xb = torch.tensor(X[s:e], dtype=torch.float32, device=device)
        # fc_ref must be (batch, 1): the model does fc_ref.unsqueeze(0) and adds
        # to mu_samples of shape (n_samples, batch, 1). A bare (batch,) vector
        # would broadcast to (n_samples, batch, batch).
        fb = torch.tensor(log_fc_ref[s:e], dtype=torch.float32, device=device).unsqueeze(-1)
        est = model.predict_with_uncertainty(
            xb, n_samples=n_samples, hurdle_config=None, fc_ref=fb,
            prediction_floor=prediction_floor)
        means.append(est.mean.cpu().numpy().squeeze(-1))
        epis.append(est.epistemic_std.cpu().numpy().squeeze(-1))
        ales.append(est.aleatoric_std.cpu().numpy().squeeze(-1))
        tots.append(est.total_std.cpu().numpy().squeeze(-1))
    return (np.concatenate(means), np.concatenate(epis),
            np.concatenate(ales), np.concatenate(tots))


def aggregate_endpoint(run_dir: Path, winner: dict, mutations: List[str],
                       new_subs: List[str], embeddings: dict, substrate_meta: dict,
                       config: dict, fc_lookup: dict, train_lookup: pd.DataFrame,
                       null_cfg: dict, device: str) -> pd.DataFrame:
    """Run one endpoint's final model on (new-sub mutant × refs), reconstruct
    absolute predictions, and aggregate under the winner reference mode.
    Returns the per-(mutation, substrate) aggregated frame with y_pred/acq_std/
    null_pred."""
    models_dir = run_dir / "models"
    model, params, metadata = load_model_from_checkpoint(models_dir, device)
    n_samples = int(params.get("n_inference_samples", 20))
    sub_emb = params["substrate_embedding_type"]
    epsilon = config.get("data", {}).get("epsilon", 0.01)
    _, _, other_pipelines = load_preprocessing_pipelines(models_dir)

    esm2_wt_len = embeddings["esm2_wt"].shape[0]
    esm2_mut_keys = set(embeddings["esm2_mut"].keys())
    pos_off = config["data"]["position_offset"]

    df_pw, report = build_inference_df(
        mutations, new_subs, substrate_meta, fc_lookup, config,
        pos_off, esm2_wt_len, esm2_mut_keys)
    if len(df_pw) == 0:
        logger.warning("  %s: no pairwise rows (coverage: %s)", run_dir.name, report)
        return pd.DataFrame()
    nrefs = list(report["ref_counts"].values())
    logger.info("  %s: %d preds, %d pairwise rows, refs/pred min=%d max=%d",
                run_dir.name, report["n_predictions"], report["n_pairwise_rows"],
                min(nrefs), max(nrefs))
    if report["skipped_no_refs"]:
        logger.warning("  %s: %d (mut,sub) had no reference data",
                       run_dir.name, len(report["skipped_no_refs"]))

    # Align feature toggles to the trained checkpoint (hyperparams can be stale).
    trained_groups = set(metadata.get("other_feature_groups", []))
    feats = dict(params.get("features") or {})
    for g in ("fc_ref", "x_target_substrate", "x_ref_substrate",
              "ref_distance", "saprot_zs", "esm_wt", "esm_mut"):
        feats[g] = g in trained_groups
    params["features"] = feats

    groups = build_other_features(df_pw, embeddings, params, substrate_meta)
    X = apply_saved_preprocessing(groups, other_pipelines)
    if X.shape[1] != metadata["other_feature_dim"]:
        raise ValueError(f"{run_dir.name}: feature dim {X.shape[1]} != "
                         f"model {metadata['other_feature_dim']}")

    log_fc_ref = np.log10(df_pw["fc_ref"].values.astype(np.float64) + epsilon).astype(np.float32)
    df_pw = df_pw.copy()
    df_pw["log_fc_ref"] = log_fc_ref
    y_pred, epi, ale, tot = _predict_absolute(
        model, X, log_fc_ref, n_samples, params.get("prediction_floor"), device)

    # _ref_distance feature for distance_weighted aggregation
    ref_dist = compute_pairwise_distances(embeddings[f"substrate_{sub_emb}"], "cosine")
    add_ref_distances(df_pw, ref_dist, embeddings["substrate_names"])

    meta_cols = [c for c in _PAIRWISE_META_COLS if c in df_pw.columns]
    pw = df_pw[meta_cols].copy()
    pw["_ref_distance"] = df_pw["_ref_distance"].values
    pw["y_pred"] = y_pred
    pw["epi_std"] = epi
    pw["ale_std"] = ale
    pw["tot_std"] = tot
    pw["fold"] = 0

    agg = aggregate_and_null_per_fold(
        mode=winner["mode"], pairwise_df=pw, train_lookup_df=train_lookup,
        embeddings=embeddings,
        substrate_embedding_type=null_cfg["emb"], distance_metric=null_cfg["metric"],
        distance_weight_temperature=null_cfg["tau"], acq_sigma="within_epi_ale")
    return agg


# ════════════════════════════════════════════════════════════════════════════
# Selection
# ════════════════════════════════════════════════════════════════════════════
def _parse_pos_aa(ms: str):
    return int(ms[1:-1]), ms[-1]


def _topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    out = df.sort_values("score", ascending=False).head(k).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def run_bnn_selectors(run_dirs: Dict[str, Path], mutations, new_subs, embeddings,
                      substrate_meta, config, fc_lookup, train_lookup, null_cfg,
                      device, top_k, bump):
    sel, allp, winners = [], [], {}
    for ep, (short, tkey) in ENDPOINTS.items():
        rd = run_dirs[ep]
        winner = pick_cv_winner(rd, tkey)
        winners[ep] = winner
        tie = " [NEAR-TIE w/ %s, Δ=%.4f]" % (winner["runner_up"], winner["gap"]) \
            if winner["near_tie"] else ""
        logger.info("Endpoint %s -> mode=%s beta=%g (CV %s=%.4f)%s",
                    ep, winner["mode"], winner["beta"], tkey, winner["score"], tie)
        agg = aggregate_endpoint(rd, winner, mutations, new_subs, embeddings,
                                 substrate_meta, config, fc_lookup, train_lookup,
                                 null_cfg, device)
        if agg.empty:
            continue
        beta = winner["beta"]
        for variant, b in [("base", beta), ("explore", beta + bump)]:
            if variant == "explore" and bump == 0:
                continue
            label = f"BNN({short})_{winner['mode']}_b{b:g}"
            a = agg.copy()
            a["score"] = a["y_pred"] + b * a["acq_std"]
            a["model"] = label
            a["model_type"] = "bnn"
            a["reference_mode"] = winner["mode"]
            a["beta"] = b
            a["beta_variant"] = variant
            a["target_metric"] = short
            a["degenerate"] = False
            for sub, g in a.groupby("substrate"):
                g = g.copy()
                g["rank_within_substrate"] = g["score"].rank(ascending=False, method="first").astype(int)
                allp.append(g)
                sel.append(_topk(g, top_k))
    return sel, allp, winners


def run_null_selectors(mutations, new_subs, data, ref_subs, embeddings, null_cfg,
                       degenerate_map, top_k):
    train_lookup = data[data["substrate"].isin(ref_subs)][
        ["mutation_string", "substrate", "log_fc"]].copy()
    train_lookup["fold"] = 0
    # one row per candidate mutant for each new substrate
    base_rows = []
    for ms in mutations:
        try:
            wt, pos0, mut = parse_mutation_string(ms, 0)  # pos only for display
        except Exception:
            continue
        base_rows.append((ms, ms[0], int(ms[1:-1]), ms[-1]))
    sel, allp = [], []
    for sub in new_subs:
        agg = pd.DataFrame([{"mutation_string": ms, "substrate": sub,
                             "position": p, "wt_aa": w, "mut_aa": mu}
                            for (ms, w, p, mu) in base_rows])
        for mode in MODES:
            null_pred = compute_null_for_mode(
                mode=mode, agg_df=agg, df_train=train_lookup, embeddings=embeddings,
                substrate_embedding_type=null_cfg["emb"], distance_metric=null_cfg["metric"],
                distance_weight_temperature=null_cfg["tau"],
                formaldehyde_substrate=FORMALDEHYDE_SUBSTRATE)
            degen = bool(degenerate_map.get(sub, False)) if mode == "nearest" else False
            a = agg.copy()
            a["score"] = null_pred
            a["y_pred"] = null_pred
            a["acq_std"] = np.nan
            a["null_pred"] = null_pred
            a["model"] = f"null_{mode}"
            # NB: "null" alone is in pandas' default na_values and would read back
            # as NaN, so label the type "null_model".
            a["model_type"] = "null_model"
            a["reference_mode"] = mode
            a["beta"] = np.nan
            a["beta_variant"] = ""
            a["target_metric"] = ""
            a["degenerate"] = degen
            a["rank_within_substrate"] = a["score"].rank(ascending=False, method="first").astype(int)
            allp.append(a)
            sel.append(_topk(a, top_k))
    return sel, allp


def compute_nearest_degeneracy(new_subs, data, ref_subs, embeddings, null_cfg):
    """For each new substrate, find its nearest of the existing ref substrates
    (null embedding/metric) and flag degenerate if that nearest is floor-dominated."""
    names = embeddings["substrate_names"]
    emb = embeddings[f"substrate_{null_cfg['emb']}"]
    dist = compute_pairwise_distances(emb, null_cfg["metric"])
    idx = {n: i for i, n in enumerate(names)}
    floor_frac = {}
    for s in ref_subs:
        d = data[data["substrate"] == s]
        floor_frac[s] = float((d["log_fc"] <= FLOOR_THRESHOLD).mean()) if len(d) else 1.0
    out = {}
    for sub in new_subs:
        if sub not in idx:
            out[sub] = True
            continue
        cands = [(s, dist[idx[sub], idx[s]]) for s in ref_subs if s in idx and s != sub]
        if not cands:
            out[sub] = True
            continue
        nearest = min(cands, key=lambda c: c[1])[0]
        out[sub] = floor_frac.get(nearest, 1.0) > 0.9
        logger.info("  %s: nearest=%s (floor_frac=%.2f) degenerate=%s",
                    sub, nearest, floor_frac.get(nearest, 1.0), out[sub])
    return out


# ════════════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════════════
def _safe(s):
    return str(s).replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def plot_agreement_heatmap(sel, out_path):
    subs = sorted(sel["substrate"].unique())
    ncol = min(4, len(subs))
    nrow = int(np.ceil(len(subs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4.0 * nrow), squeeze=False)
    vmax = max(1, sel.groupby(["substrate", "mutation"])["model"].nunique().max())
    for i, sub in enumerate(subs):
        ax = axes[i // ncol][i % ncol]
        s = sel[sel["substrate"] == sub]
        positions = sorted({_parse_pos_aa(m)[0] for m in s["mutation"]})
        pidx = {p: j for j, p in enumerate(positions)}
        grid = np.zeros((len(AA_ORDER), len(positions)))
        for m, n in s.groupby("mutation")["model"].nunique().items():
            p, aa = _parse_pos_aa(m)
            if aa in AA_ORDER and p in pidx:
                grid[AA_ORDER.index(aa), pidx[p]] = n
        im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(positions))); ax.set_xticklabels(positions, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(AA_ORDER))); ax.set_yticklabels(AA_ORDER, fontsize=6)
        ax.set_title(sub, fontsize=9); ax.set_xlabel("position", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, label="# selectors")
    for j in range(len(subs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Selected single mutants — selector agreement (darker = more agree)", y=1.01)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight", dpi=140); plt.close(fig)


def plot_selector_overlap(sel, sub, out_path):
    s = sel[sel["substrate"] == sub]
    models = sorted(s["model"].unique())
    muts = sorted(s["mutation"].unique(), key=_parse_pos_aa)
    grid = np.array([[1 if m in set(s[s["model"] == md]["mutation"]) else 0
                      for m in muts] for md in models])
    fig, ax = plt.subplots(figsize=(max(6, 0.42 * len(muts)), 0.45 * len(models) + 2))
    ax.imshow(grid, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(muts))); ax.set_xticklabels(muts, rotation=90, fontsize=7)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=7)
    ax.set_title(f"{sub}: which selector picked which mutant")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight", dpi=140); plt.close(fig)


def plot_consensus_bar(consensus, sub, out_path):
    s = consensus[consensus["substrate"] == sub].sort_values(["position", "mutation"])
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.34 * len(s)), 4))
    colors = plt.cm.viridis(s["num_models"] / max(1, s["num_models"].max()))
    ax.bar(range(len(s)), s["num_models"], color=colors)
    ax.set_xticks(range(len(s))); ax.set_xticklabels(s["mutation"], rotation=90, fontsize=7)
    ax.set_ylabel("# selectors"); ax.set_title(f"{sub}: selected-mutant consensus")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight", dpi=140); plt.close(fig)


def plot_score_distribution(sel, out_path):
    models = sorted(sel["model"].unique())
    data = [sel[sel["model"] == m]["score"].dropna().values for m in models]
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(models)), 5))
    ax.boxplot(data, vert=True, labels=models)
    ax.set_xticklabels(models, rotation=90, fontsize=7)
    ax.set_ylabel("selection score"); ax.set_title("Top-k score distribution by selector")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight", dpi=140); plt.close(fig)


def plot_bnn_vs_null(consensus, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    c = consensus.copy()
    c["n_bnn"] = c["models"].apply(lambda s: sum(1 for m in s.split(";") if m.startswith("BNN")))
    c["n_null"] = c["models"].apply(lambda s: sum(1 for m in s.split(";") if m.startswith("null")))
    jit = (np.random.RandomState(0).rand(len(c), 2) - 0.5) * 0.25
    ax.scatter(c["n_null"] + jit[:, 0], c["n_bnn"] + jit[:, 1], alpha=0.5, s=30)
    ax.set_xlabel("# null selectors picking the mutant")
    ax.set_ylabel("# BNN selectors picking the mutant")
    ax.set_title("Per-mutant agreement: BNN vs null")
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight", dpi=140); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-substrates-json", nargs="+",
                    default=["data/round3/new_substrates.json",
                             "data/round3/extra_substrates_smiles.json"])
    ap.add_argument("--endpoints", default=",".join(ENDPOINTS.keys()))
    ap.add_argument("--null-modes", default=",".join(MODES))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--explore-beta-bump", type=float, default=1.0)
    ap.add_argument("--reference-policy", choices=["all", "active_only"], default="all")
    ap.add_argument("--runs-root", default="results/new_05_bnn2/substrate")
    ap.add_argument("--output-dir", default="results/new_05_bnn2/deployment")
    ap.add_argument("--availability-xlsx", default="data/round3/extra_substrates.xlsx")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    config = load_config(args.config)
    # honor reference policy (BNN side reads this from config)
    config.setdefault("bnn2", {}).setdefault("pairwise", {})["ref_substrates"] = args.reference_policy
    logger.info("Reference policy: %s", args.reference_policy)

    processed_dir = PROJECT_ROOT / config["data"]["output_dir"]
    embeddings = load_all_embeddings(processed_dir)
    substrate_meta = load_substrate_metadata(processed_dir)
    data = load_multi_substrate_data(processed_dir)
    fc_lookup = build_fc_lookup(processed_dir)

    existing = list(embeddings["substrate_names"])
    active_subs = sorted(s for s, m in substrate_meta.items() if m.get("is_active"))
    ref_subs = active_subs if args.reference_policy == "active_only" else list(existing)

    # new substrate SMILES (union of the given files)
    new_smiles = {}
    for jp in args.new_substrates_json:
        with open(_abs(jp)) as f:
            new_smiles.update(json.load(f))
    logger.info("Loaded %d candidate new substrates from %s",
                len(new_smiles), args.new_substrates_json)

    runs_root = _abs(args.runs_root)
    run_dirs = {ep: runs_root / ep for ep in args.endpoints.split(",")}
    # embedding types needed = union over endpoints + maccs (null)
    need_types = set(["maccs"])
    for ep in run_dirs:
        hp = json.load(open(run_dirs[ep] / "hyperparams.json"))
        need_types.add(hp["substrate_embedding_type"])
    logger.info("Embedding types to compute for new substrates: %s", sorted(need_types))

    new_subs = inject_new_substrates_multi(embeddings, substrate_meta, new_smiles,
                                            sorted(need_types), config)
    missing = [s for s in new_subs if s not in embeddings["substrate_names"]]
    if missing:
        raise SystemExit(f"injection failed for: {missing}")

    # null config (data-driven; consistent across runs)
    tm = json.load(open(run_dirs[args.endpoints.split(",")[0]] / "train_metadata.json"))
    null_cfg = {"emb": tm["null_model_embedding"], "metric": tm["null_model_distance_metric"],
                "tau": float(tm.get("distance_weighted_temperature", 1.0))}
    logger.info("Null config: emb=%s metric=%s tau=%.2f",
                null_cfg["emb"], null_cfg["metric"], null_cfg["tau"])

    mutations = sorted(data["mutation_string"].unique().tolist())

    # BNN selectors
    train_lookup = data[data["substrate"].isin(ref_subs)][
        ["mutation_string", "substrate", "log_fc"]].copy()
    train_lookup["fold"] = 0
    bnn_sel, bnn_all, winners = run_bnn_selectors(
        run_dirs, mutations, new_subs, embeddings, substrate_meta, config,
        fc_lookup, train_lookup, null_cfg, args.device, args.top_k,
        args.explore_beta_bump)

    # Null selectors (+ degeneracy)
    degen = compute_nearest_degeneracy(new_subs, data, ref_subs, embeddings, null_cfg)
    null_sel, null_all = run_null_selectors(
        mutations, new_subs, data, ref_subs, embeddings, null_cfg, degen, args.top_k)

    # ── Assemble outputs ──
    out_dir = _abs(args.output_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    sel = pd.concat(bnn_sel + null_sel, ignore_index=True).rename(
        columns={"mutation_string": "mutation"})
    sel_cols = ["substrate", "mutation", "model", "model_type", "reference_mode",
                "beta", "beta_variant", "target_metric", "rank", "score",
                "y_pred", "acq_std", "degenerate"]
    sel = sel[[c for c in sel_cols if c in sel.columns]].sort_values(
        ["substrate", "model", "rank"]).reset_index(drop=True)

    allp = pd.concat(bnn_all + null_all, ignore_index=True).rename(
        columns={"mutation_string": "mutation"})
    allp_cols = ["substrate", "mutation", "model", "model_type", "reference_mode",
                 "beta", "beta_variant", "target_metric", "rank_within_substrate",
                 "score", "y_pred", "acq_std", "null_pred", "n_refs", "degenerate"]
    allp = allp[[c for c in allp_cols if c in allp.columns]]

    # availability metadata (extra_substrates.xlsx)
    avail = {}
    xp = _abs(args.availability_xlsx)
    if xp.exists():
        try:
            ax = pd.read_excel(xp)
            name_col = next((c for c in ax.columns if "substrate" in str(c).lower()), None)
            for _, r in ax.iterrows():
                if name_col and pd.notna(r[name_col]):
                    avail[str(r[name_col]).strip()] = {
                        "available": True,
                        "storage": r.get("Storage", ""), "class": r.get("Class", "")}
        except Exception as e:
            logger.warning("could not read availability xlsx: %s", e)
    if avail:
        sel["available"] = sel["substrate"].isin(avail)
        allp["available"] = allp["substrate"].isin(avail)

    sel.to_csv(out_dir / "selections.csv", index=False)
    allp.sort_values(["substrate", "model", "rank_within_substrate"]).to_csv(
        out_dir / "all_predictions.csv", index=False)
    logger.info("Wrote selections.csv (%d rows), all_predictions.csv (%d rows)",
                len(sel), len(allp))

    # consensus
    rows = []
    for (sub, mut), g in sel.groupby(["substrate", "mutation"]):
        p, aa = _parse_pos_aa(mut)
        rows.append({"substrate": sub, "mutation": mut, "position": p,
                     "wt_aa": mut[0], "mut_aa": aa,
                     "num_models": g["model"].nunique(),
                     "models": ";".join(sorted(g["model"].unique())),
                     "mean_rank": float(g["rank"].mean()),
                     "avg_score": float(g["score"].mean())})
    consensus = pd.DataFrame(rows)
    if avail:
        consensus["available"] = consensus["substrate"].isin(avail)
    consensus = consensus.sort_values(
        ["substrate", "num_models", "mean_rank"], ascending=[True, False, True]
    ).reset_index(drop=True)
    consensus.to_csv(out_dir / "mutation_consensus.csv", index=False)
    logger.info("Wrote mutation_consensus.csv (%d unique selected mutants)", len(consensus))

    # plots
    fig = out_dir / "figures"
    plot_agreement_heatmap(sel, fig / "agreement_heatmap.png")
    plot_score_distribution(sel, fig / "score_distribution_by_selector.png")
    plot_bnn_vs_null(consensus, fig / "bnn_vs_null_overlap.png")
    for sub in new_subs:
        plot_selector_overlap(sel, sub, fig / f"selector_overlap_{_safe(sub)}.png")
        plot_consensus_bar(consensus, sub, fig / f"consensus_{_safe(sub)}.png")
    logger.info("Wrote figures to %s", fig)

    # winner summary
    logger.info("=" * 60)
    logger.info("Deployed BNN configs (locked-in CV winners):")
    for ep, w in winners.items():
        logger.info("  %-34s mode=%-18s beta=%g%s", ep, w["mode"], w["beta"],
                    "  [near-tie]" if w["near_tie"] else "")
    logger.info("Done.")


if __name__ == "__main__":
    main()
