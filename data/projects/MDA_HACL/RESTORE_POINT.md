# MDA_HACL restore point — deployment / selections (2026-06-16)

Tag: **`restore/deploy-selections-20260616`** (annotated). Checking it out reproduces
the full state below: locked-in epi+ale hyperopt, substrate-split CV + plots, the
data-sensitivity sweep plots, the new-substrate selections, and the trained final
models. Use this to roll back after experimental code changes.

## How to get back to this state
```bash
# inspect / restore everything (code + results + models)
git checkout restore/deploy-selections-20260616          # detached HEAD at the snapshot
# or, while staying on a branch, restore just the project tree:
git checkout restore/deploy-selections-20260616 -- data/projects/MDA_HACL
# or hard-reset main to it (destructive to later main work):
git reset --hard restore/deploy-selections-20260616
```
The trained models and optuna `study.sqlite` are normally gitignored; this restore
commit **force-adds** them so the snapshot is self-contained and survives re-training
or `git clean -fdx`.

## Where everything lives (paths under data/projects/MDA_HACL/)
- **Hyperopt** (per metric: ndcg, spearman_rho, top3_recovery, per_position_ndcg_mean,
  per_position_top3_recovery_mean): `results/new_opt_05/all_substrates_<metric>/`
  — `best_hyperparams.json`, `study_results.json`, `working_range.json`,
  `figures/` (objective_trace, objective_vs_beta, param_importance, pareto, …),
  `study.sqlite` (force-added; reopen the optuna study).
- **CV (substrate-split) + plots**: `results/new_05_bnn2/substrate/<metric>/`
  — `scoring/metrics.json`, `scoring/figures/`, `pairwise_predictions.csv`,
  `train_lookup.csv`, `train_metadata.json`, `hyperparams.json`.
- **Trained final models** (all-data; force-added): `results/new_05_bnn2/substrate/<metric>/models/`
  — `final_model.pt`, `model_metadata.json`, `preprocessing_*.joblib`.
  NOTE: architecture is taken from the checkpoint at load time (see
  `new_05d_bnn2_deploy_select.load_model_from_checkpoint`); rebuild with
  `new_05_bnn2_train.py --skip-cv --hyperparams <best_hyperparams.json>`.
- **Data-sensitivity sweep plots**: `results/new_05_bnn2/sensitivity/<metric>/summary/`
  — `sensitivity_summary.csv`, pooled/best/bars PNGs.
- **Selections (deployment on 25 new substrates)**: `results/new_05_bnn2/deployment/`
  — `selections.csv` (12 selectors × 25 subs × top-5), `all_predictions.csv`,
  `mutation_consensus.csv` (sorted by num_models), `figures/`.

## Deployment config captured here
4 BNN endpoints (locked-in-CV winning reference mode + recorded β) + their explore
variants (β+1) + 4 null modes = 12 selectors per substrate. References = "all"
(matches training). Final models were retrained via `--skip-cv` to match the
committed `best_hyperparams.json` (they previously drifted because `models/` is
gitignored — see git history). Footprint: 40 distinct protein variants / 349
substrate×variant assays across the 25 new substrates.

## NOT included (gitignored, reconstructable / not needed to view results)
Substrate + ESM2/SaProt embeddings (`data/round3/processed/embeddings/`), the
sensitivity per-subset `substrate/` runs, raw data, slurm logs. Embeddings are
on-disk and untouched by git operations; rebuild via `01_embeddings.py` if cleaned.
