#!/usr/bin/env python
"""
compare_sigma_modes.py — Compare BNN-vs-null across acquisition-σ regimes.

Each acquisition-σ regime (total / within_epi_ale / within_epi) lives in its own
git commit on the `hacl` branch, with a self-consistent fresh hyperopt + CV (the
σ used to RANK matches the σ the hyperopt optimised under). This script reads each
regime's metrics.json straight out of its commit (``git show <ref>:<path>`` — no
checkout) and produces the same per-endpoint bar plots the scorer makes, but with
one BNN bar per σ strategy alongside the (σ-invariant) null.

For every endpoint × ranking metric it writes:
  - <endpoint>/<overall|per_position>_<metric>_by_mode.png
        grouped bars: x = reference modes, bars = [null, BNN@bestβ per σ]
  - <endpoint>/delta_<metric>_by_mode.png
        bars = BNN@bestβ − null per σ (the "improvement over null")
Plus a cross-endpoint headline (each endpoint's own optimised metric, best mode)
and summary.csv with every (endpoint, mode, metric, sigma, beta, bnn, null, delta).

Usage:
  python compare_sigma_modes.py \
      --sigma-refs total=2ad82ba epi_ale=e53878e epi_only=HEAD
Refs that don't resolve (e.g. epi_only before it's committed) are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]                      # .../data/projects/MDA_HACL
REL_BASE = "data/projects/MDA_HACL/results/new_05_bnn2/substrate"

ENDPOINTS = ["ndcg", "spearman_rho", "top3_recovery",
             "per_position_ndcg_mean", "per_position_top3_recovery_mean"]
MODES = ["formaldehyde", "nearest", "avg_all", "distance_weighted"]

# (display metric, BNN flat key in bnn_at_best_beta, null dotted path in null block)
METRICS: List[Tuple[str, str, str]] = [
    ("spearman_rho",                    "spearman_rho",                    "spearman_rho"),
    ("ndcg",                            "ndcg",                            "ndcg"),
    ("top1_recovery",                   "top1_recovery",                   "top1_recovery.recovery"),
    ("top3_recovery",                   "top3_recovery",                   "top3_recovery.recovery"),
    ("top5_recovery",                   "top5_recovery",                   "top5_recovery.recovery"),
    ("per_position_spearman_mean",      "per_position_spearman_mean",      "per_position_spearman_mean.mean"),
    ("per_position_ndcg_mean",          "per_position_ndcg_mean",          "per_position_ndcg_mean.mean"),
    ("per_position_top1_recovery",      "per_position_top1_recovery",      "per_position_top1_recovery.recovery"),
    ("per_position_top3_recovery_mean", "per_position_top3_recovery_mean", "per_position_top3_recovery_mean.mean"),
    ("per_position_top5_recovery_mean", "per_position_top5_recovery_mean", "per_position_top5_recovery_mean.mean"),
]
OVERALL = {"spearman_rho", "ndcg", "top1_recovery", "top3_recovery", "top5_recovery"}
# each endpoint's own optimised metric (for the headline figure)
ENDPOINT_METRIC = {
    "ndcg": "ndcg",
    "spearman_rho": "spearman_rho",
    "top3_recovery": "top3_recovery",
    "per_position_ndcg_mean": "per_position_ndcg_mean",
    "per_position_top3_recovery_mean": "per_position_top3_recovery_mean",
}

NULL_COLOR = "#999999"
SIGMA_COLORS = {"total": "#4477aa", "epi_ale": "#66ccee", "epi_only": "#228833"}
_FALLBACK = ["#ee6677", "#ccbb44", "#aa3377"]


def _repo_root() -> str:
    return subprocess.run(
        ["git", "-C", str(SCRIPT_DIR), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True).stdout.strip()


def git_show_json(repo: str, ref: str, relpath: str) -> Optional[dict]:
    """Return parsed JSON of <relpath> at <ref>, or None if it doesn't exist."""
    r = subprocess.run(["git", "-C", repo, "show", f"{ref}:{relpath}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def _dotted(d, path: str) -> float:
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return float("nan")
    try:
        return float(cur)
    except (TypeError, ValueError):
        return float("nan")


def load_all(repo: str, refs: Dict[str, str]) -> Dict[str, Dict[str, Optional[dict]]]:
    """{endpoint: {sigma_label: metrics.json dict or None}}; warns on missing refs."""
    out: Dict[str, Dict[str, Optional[dict]]] = {}
    for ep in ENDPOINTS:
        relpath = f"{REL_BASE}/{ep}/scoring/metrics.json"
        out[ep] = {}
        for label, ref in refs.items():
            mj = git_show_json(repo, ref, relpath)
            if mj is None:
                print(f"  [skip] {label} ({ref}): {relpath} not found in commit")
            out[ep][label] = mj
    return out


def _mode_metrics(mj: dict, mode: str) -> Optional[dict]:
    try:
        return mj["modes"][mode]["metrics"]
    except (KeyError, TypeError):
        return None


def _color(label: str, i: int) -> str:
    return SIGMA_COLORS.get(label, _FALLBACK[i % len(_FALLBACK)])


def plot_metric(ep: str, disp: str, bnn_key: str, null_path: str,
                data: Dict[str, Optional[dict]], sigma_labels: List[str],
                out_dir: Path):
    """Grouped bars: x = modes; per mode [null, BNN per σ]. Returns rows for CSV."""
    rows = []
    # null is σ-invariant: take it from the first σ that has each mode.
    null_vals, bnn_vals = {}, {s: {} for s in sigma_labels}
    betas = {s: {} for s in sigma_labels}
    for mode in MODES:
        nv = float("nan")
        for s in sigma_labels:
            mj = data.get(s)
            mm = _mode_metrics(mj, mode) if mj else None
            if mm is None:
                bnn_vals[s][mode] = float("nan"); betas[s][mode] = float("nan")
                continue
            bnn_vals[s][mode] = _dotted(mm.get("bnn_at_best_beta", {}), bnn_key)
            betas[s][mode] = float(mm.get("best_beta", float("nan")))
            if math.isnan(nv):
                nv = _dotted(mm.get("null", {}), null_path)
        null_vals[mode] = nv
        for s in sigma_labels:
            rows.append({"endpoint": ep, "mode": mode, "metric": disp, "sigma": s,
                         "beta": betas[s][mode], "bnn": bnn_vals[s][mode],
                         "null": nv, "delta": bnn_vals[s][mode] - nv})

    n_series = 1 + len(sigma_labels)            # null + each σ
    x = np.arange(len(MODES))
    w = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(MODES)), 5))
    # null bars (left of each group)
    ax.bar(x + (0 - n_series / 2 + 0.5) * w, [null_vals[m] for m in MODES], w,
           label="null", color=NULL_COLOR, edgecolor="black", linewidth=1.2, hatch="//")
    for i, s in enumerate(sigma_labels):
        offs = (i + 1 - n_series / 2 + 0.5) * w
        vals = [bnn_vals[s][m] for m in MODES]
        ax.bar(x + offs, vals, w, label=f"BNN [{s}]", color=_color(s, i),
               edgecolor="black", linewidth=1.2)
        for xi, m in zip(x, MODES):
            v, b = bnn_vals[s][m], betas[s][m]
            if not math.isnan(v):
                ax.text(xi + offs, v, f"{v:.2f}\nβ{b:g}", ha="center", va="bottom",
                        fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(MODES, rotation=15, ha="right")
    ax.set_ylabel(disp)
    ax.set_title(f"{ep} — {disp}: BNN @ best β per σ vs null")
    ax.axhline(0, color="#444", lw=1.0)
    ax.legend(fontsize=8, ncol=n_series, loc="best")
    ax.grid(axis="y", ls="--", lw=0.7, alpha=0.4)
    plt.tight_layout()
    prefix = "overall" if disp in OVERALL else "per_position"
    fig.savefig(out_dir / f"{prefix}_{disp}_by_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # delta-to-null companion
    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(MODES)), 4.5))
    w2 = 0.8 / max(1, len(sigma_labels))
    for i, s in enumerate(sigma_labels):
        offs = (i - len(sigma_labels) / 2 + 0.5) * w2
        deltas = [bnn_vals[s][m] - null_vals[m] for m in MODES]
        ax.bar(x + offs, deltas, w2, label=f"{s}", color=_color(s, i),
               edgecolor="black", linewidth=1.2)
        for xi, d in zip(x, deltas):
            if not math.isnan(d):
                ax.text(xi + offs, d, f"{d:+.2f}", ha="center",
                        va="bottom" if d >= 0 else "top", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(MODES, rotation=15, ha="right")
    ax.set_ylabel(f"{disp}  (BNN − null)")
    ax.set_title(f"{ep} — {disp}: improvement over null, per σ")
    ax.axhline(0, color="#444", lw=1.2)
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", ls="--", lw=0.7, alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_dir / f"delta_{disp}_by_mode.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows


def plot_headline(all_rows: List[dict], sigma_labels: List[str], out_dir: Path):
    """One figure: per endpoint, best-over-modes improvement-over-null of its own
    optimised metric, grouped by σ."""
    x = np.arange(len(ENDPOINTS))
    w = 0.8 / max(1, len(sigma_labels))
    fig, ax = plt.subplots(figsize=(max(9, 2.4 * len(ENDPOINTS)), 5))
    for i, s in enumerate(sigma_labels):
        offs = (i - len(sigma_labels) / 2 + 0.5) * w
        vals = []
        for ep in ENDPOINTS:
            met = ENDPOINT_METRIC[ep]
            ds = [r["delta"] for r in all_rows
                  if r["endpoint"] == ep and r["metric"] == met and r["sigma"] == s
                  and not math.isnan(r["delta"])]
            vals.append(max(ds) if ds else float("nan"))
        ax.bar(x + offs, vals, w, label=s, color=_color(s, i),
               edgecolor="black", linewidth=1.2)
        for xi, v in zip(x, vals):
            if not math.isnan(v):
                ax.text(xi + offs, v, f"{v:+.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ep}\n({ENDPOINT_METRIC[ep]})" for ep in ENDPOINTS],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("best-mode improvement over null (BNN − null)")
    ax.set_title("Headline: best improvement over null per σ, "
                 "each endpoint at its own optimised metric")
    ax.axhline(0, color="#444", lw=1.2)
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls="--", lw=0.7, alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_dir / "headline_improvement_over_null.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sigma-refs", nargs="+",
                   default=["total=2ad82ba", "epi_ale=e53878e", "epi_only=HEAD"],
                   help="label=gitref pairs (order = bar order). Missing refs skipped.")
    p.add_argument("--out", type=str,
                   default=str(PROJECT / "results/new_05_bnn2/sigma_comparison"))
    return p.parse_args()


def main():
    args = parse_args()
    repo = _repo_root()
    refs = {}
    for pair in args.sigma_refs:
        if "=" not in pair:
            raise SystemExit(f"--sigma-refs entries must be label=ref, got {pair!r}")
        label, ref = pair.split("=", 1)
        refs[label] = ref
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    print(f"repo={repo}\nrefs={refs}\nout={out_root}")

    data = load_all(repo, refs)
    # keep only σ labels that resolved for at least one endpoint
    sigma_labels = [s for s in refs
                    if any(data[ep][s] is not None for ep in ENDPOINTS)]
    print(f"σ regimes available: {sigma_labels}")

    all_rows: List[dict] = []
    for ep in ENDPOINTS:
        ep_dir = out_root / ep; ep_dir.mkdir(parents=True, exist_ok=True)
        for disp, bnn_key, null_path in METRICS:
            all_rows += plot_metric(ep, disp, bnn_key, null_path,
                                    data[ep], sigma_labels, ep_dir)
        print(f"  wrote plots for endpoint: {ep}")

    plot_headline(all_rows, sigma_labels, out_root)

    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["endpoint", "mode", "metric", "sigma",
                                            "beta", "bnn", "null", "delta"])
        wtr.writeheader()
        for r in all_rows:
            wtr.writerow(r)
    print(f"wrote {csv_path}  ({len(all_rows)} rows)")
    print(f"done — figures under {out_root}")


if __name__ == "__main__":
    main()
