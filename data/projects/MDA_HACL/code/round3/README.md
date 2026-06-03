# Round 3, Part 1: Multi-Substrate Activity Prediction via BNN

## Goal

Given SSM activity data across 6 active + 3 inactive substrates at 10 positions, build a model that predicts **fold-change activity of a mutation on an arbitrary substrate** with calibrated uncertainty. Use this to select ~10 mutations to test for each of ~15 untested substrates.

## Architecture Overview

```
                         BNN2 (final predictor)
                         ========================
  Features (ref_mode = "formaldehyde_only"):
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. FC_ref             (scalar: fold-change on formaldehyde) │
  │ 2. X_target_substrate (target substrate embedding)          │
  │ 3. X_aa               (learned AA-position embedding from   │
  │                         BNN1 latent space)                  │
  │ 4. ESM_aa_wt          (ESM2 residue embedding, wild-type)   │
  │ 5. ESM_aa_mut         (ESM2 residue embedding, mutant)      │
  │ 6. SaProt_zs          (SaProt zero-shot fitness score)      │
  └─────────────────────────────────────────────────────────────┘

  Features (ref_mode = "pairwise"):
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. FC_ref             (scalar: fold-change on ref substrate)│
  │ 2. X_ref_substrate    (reference substrate embedding)       │
  │ 3. X_target_substrate (target substrate embedding)          │
  │ 4. X_aa               (learned AA-position embedding)       │
  │ 5. ESM_aa_wt          (ESM2 residue embedding, wild-type)   │
  │ 6. ESM_aa_mut         (ESM2 residue embedding, mutant)      │
  │ 7. SaProt_zs          (SaProt zero-shot fitness score)      │
  └─────────────────────────────────────────────────────────────┘
              │
              ▼
  BNN2 → predicted fold-change on target substrate + uncertainty
```

### BNN1: AA-Position Embedder (pretrained on formaldehyde data)

BNN1 is a Bayesian neural network trained on the **full 64-position formaldehyde SSM dataset** (~1,280 mutations). Its purpose is to learn a rich, position-aware amino acid representation.

**Training strategy:**
1. **Phase A — Position classification (sanity check):** Train BNN1 to predict which position a mutation is at, given ESM2 residue embeddings of the mutant AA only. WT embeddings are excluded because they are identical for all ~20 substitutions at a given position, which would let the model memorize position trivially. This validates that ESM2 natively encodes positional context in mutant residue representations. Random split over all 1,272 mutations.
2. **Phase B — Formaldehyde activity regression:** Retrain BNN1 to predict fold-change on formaldehyde. Hyperopt with random split (predict mutations at held-out positions we've seen). Regression target = log10(fold-change + epsilon).
3. **Embedder extraction:** Once BNN1 is trained, extract a latent vector (penultimate hidden layer) as `X_aa` for use in BNN2.

### Embedding Preprocessing

Raw embeddings (ESM2 ~1280-dim per residue, substrate fingerprints ~2048-dim, etc.) go through a configurable preprocessing pipeline before entering any BNN. Preprocessing choices are **hyperparameters**:

- **Scaling:** None, StandardScaler, MinMaxScaler, RobustScaler
- **Dimensionality reduction:** None, PCA (n_components as hyperparam), or truncated SVD
- **Per feature group:** Each feature group (ESM_wt, ESM_mut, X_substrate, X_aa, etc.) can have its own preprocessing pipeline

Fitted sklearn preprocessing objects are saved alongside model checkpoints for reproducibility. Preprocessing is fit on training data only and applied to val/test.

### BNN2: Multi-Substrate Predictor

BNN2 combines all features to predict fold-change on any substrate.

**Evaluation splits (in order of difficulty):**
1. **Random split:** Random held-out mutation-substrate pairs (easiest).
2. **Positional generalization:** Val set contains unseen positions for a seen substrate plus an unseen substrate. Tests within-substrate extrapolation.
3. **Substrate generalization:** Val set is an entire substrate not seen during training (hardest). Tests cross-substrate transfer.

**Training set balancing (hyperparam):** The 3 inactive substrates produce mostly-zero activity, which could dominate training. Options searched via hyperopt:
- No balancing (raw data)
- Oversample active-substrate rows
- Undersample inactive-substrate rows
- Sample weights inversely proportional to substrate activity prevalence

### Reference Substrate Mode (hyperparam)

BNN2 takes a "reference fold-change" as one of its input features — the activity of this mutation on some known substrate. How this reference is chosen is controlled by `ref_mode`, searched via hyperopt:

**`formaldehyde_only` (original design):**
- FC_ref = formaldehyde fold-change for every row. One training row per (mutation, target_substrate) pair.
- Simple, uses formaldehyde as a universal activity anchor since it has the most data (64 positions).
- Limitation: formaldehyde FC is uninformative for substrates with very different activity profiles (e.g. pyruvate, Spearman ~0.2).

**`pairwise` (reference substrate as explicit input):**
- FC_ref comes from a variable reference substrate, paired with its embedding (X_ref_substrate).
- Training data is expanded to (mutation, ref_substrate, target_substrate) triplets. For 6 active substrates, each mutation generates up to 6x5 = 30 ordered pairs (excluding self-reference) instead of 6 rows. ~5x data increase.
- The model learns transfer functions between substrate pairs: "given FC on substrate R, predict FC on substrate T."
- Only active substrates are used as references (inactive substrates have FC=0 for all mutations — no discriminative information).
- Self-reference (ref == target) is excluded to avoid the trivial identity solution.
- X_ref_substrate and X_target_substrate share the same preprocessing pipeline (same scaler/PCA) since they are the same type of vector.

**Inference on new substrates (pairwise mode):**
- For each mutation, predict using every known substrate as reference and aggregate (mean).
- The spread across reference substrates provides a complementary uncertainty signal beyond the BNN's epistemic uncertainty — it captures "how much do the known substrates agree about this prediction?"

**CV split implications (pairwise mode):**
- **Substrate generalization (leave-one-out):** The held-out substrate never appears as ref or target in training. At test time, it is the target with known substrates as refs — matches the real use case exactly.
- **Positional generalization:** Split by position first, then expand into pairs. A mutation at a held-out position must not appear as a reference observation in training.

### Target Transform: log10(FC + epsilon)

We use `log10(fold_change + epsilon)` as the regression target. Rationale:
- FC=1 → 0 (WT-like), FC>1 → positive (beneficial), FC<1 → negative (detrimental)
- Dead mutations (FC≈0) → large negative (≈-2 with epsilon=0.01)
- MAE in log-space = multiplicative accuracy (0.3 error ≈ 2x off)
- Not dominated by a few high-FC outliers
- Calibration intervals in log-space = fold-range intervals
- Sign directly captures beneficial vs detrimental — important for selection

### Pyruvate Handling

Pyruvate has no detectable WT activity; data was normalized to the lowest-activity mutant (V83P). For pyruvate, the V83P variant is treated as the "reference" sequence. This means:
- ESM_wt embeddings for pyruvate = ESM residue embeddings of V83P (not true WT)
- Fold-changes are relative to V83P, not WT
- This is flagged in the data so models can potentially learn substrate-specific reference handling

### Inactive Substrates

The 3 inactive substrates (no detected activity for any mutation) are included as training data. This provides the model with negative examples — some substrates simply don't work. Considerations:
- **Balancing** (see above) prevents these from dominating the loss
- **Metrics:** We use metrics robust to class imbalance:
  - Spearman rho (rank-based, not affected by zero-inflation)
  - MAE on active-only subset (separately reported)
  - Binary classification metrics: can the model distinguish active vs inactive mutation-substrate pairs? (AUROC, AUPRC)
  - Calibration evaluated separately for active vs full dataset

### Calibration

For all tasks, beyond standard metrics (MAE, R2, Spearman), we evaluate:
- **Calibration:** Expected calibration error of the predictive intervals (e.g. do 90% CI contain 90% of observations?)
- **Sharpness:** Average width of the predictive intervals (tighter = better, given calibration).

## Data Sources

| Data | Source file | N |
|------|------------|---|
| Formaldehyde SSM (64 pos) | `data/experimental/ssm_data.csv` | 1,272 mutations |
| Multi-substrate SSM (10 pos) | `data/round3/all_ssm_data.xlsx` | 1,798 rows (9 substrates: 6 active + 3 inactive) |
| Combined SSM + Rd1 | `data/experimental/combined_data.csv` | ~1,570 |
| Protein structure | `structure/mda_hacl_b4fe6_relaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb` | 1 |
| MSA | `data/msa/mda_hacl.a2m` | 5,126 seqs |
| WT sequence | 565 residues, extracted from synonymous mutations | 1 |
| Unique mutations (union) | formaldehyde ∪ multi-substrate, deduplicated | 1,274 |

## Pipeline Scripts

Scripts are numbered and should be run in order. All paths are relative to the project root (`MDA_HACL/`).

### 00 — Data Wrangling
`00_data_wrangling.py`
- [x] Load `ssm_data.csv` (formaldehyde, 64 positions → 1,272 mutations)
- [x] Load `all_ssm_data.xlsx` (9 substrates: 6 active + 3 inactive, 10 positions each → 1,798 rows)
- [x] Parse mutation info: position, wt_aa, mut_aa, fold-change
- [x] Unify schema across substrates
- [x] Handle pyruvate: flag reference sequence (V83P), store ref_sequence per substrate
- [x] Handle inactive substrates: all FCs = 0, flag as inactive
- [x] Extract WT sequence (565 residues)
- [x] Compute and store log10(FC + epsilon) target
- [x] Save clean CSVs: `processed/formaldehyde_ssm.csv`, `processed/multi_substrate_ssm.csv`
- [x] Save substrate metadata: names, SMILES, ref_sequence, is_active flag → `processed/substrate_metadata.json`

`00b_data_exploration.py`
- [x] Activity distributions per substrate (histograms, zero-fraction bar chart)
- [x] Formaldehyde heatmap (64 positions × 20 AAs)
- [x] Multi-substrate mutation heatmaps
- [x] Cross-substrate correlation matrix and scatter plots
- [x] Position activity profiles across substrates
- [x] Pyruvate detail analysis (V83P reference)
- [x] Beneficial mutation overlap across substrates
- [x] Outputs: 10 plots in `results/00_data_exploration/`

### 01 — Precompute Embeddings
`01_embeddings.py` — CLI tool with argparse for reuse on new substrates/mutations.
- [x] ESM2 (650M) residue-level embeddings for WT → `esm2_wt_residues.npy` (565, 1280)
- [x] ESM2 residue-level embeddings for V83P (pyruvate reference) → `esm2_v83p_residues.npy` (565, 1280)
- [x] ESM2 residue-level embeddings for each mutant at mutated position → `esm2_mutant_residues.npz` (1,274 × 1280)
- [x] SaProt zero-shot scores for each mutation (wildtype marginal) → `saprot_scores.json` (1,274 scores, range [-18.7, 1.3])
- [x] Substrate embeddings — **precompute ALL types** (hyperopt selects which to use):
  - [x] RDKit Morgan fingerprints → `substrate_morgan.npy` (9, 2048)
  - [x] RDKit MACCS keys → `substrate_maccs.npy` (9, 166)
  - [x] Mordred 2D descriptors → `substrate_mordred.npy` (9, 1124) — 489 NaN columns dropped
  - [x] MoLFormer-XL pretrained embeddings → `substrate_molformer.npy` (9, 768) — IBM, trained on ~110M molecules
- [x] Position verification: checks wt_seq[pos]==wt_aa, variant[pos]==mut_aa, single-mutation-only for all 1,274 mutations
- [x] Save all embeddings to `processed/embeddings/` as .npy / .npz / .json files

`01b_embedding_exploration.py`
- [x] ESM2 WT PCA (colored by position and AA identity)
- [x] ESM2 mutation effect analysis (L2 distance WT vs mutant by position, vs activity)
- [x] SaProt vs formaldehyde activity (Spearman rho = 0.51, p = 4.6e-81)
- [x] SaProt score distribution (active vs dead mutations)
- [x] Substrate embedding PCA (Morgan, MACCS, Mordred — 3 subplots)
- [x] Substrate pairwise distance heatmaps
- [x] Feature dimension summary (bar chart)
- [x] Per-position ESM2 embedding variance vs mean fold-change
- [x] Outputs: 8 plots in `results/01_embeddings/`

**CLI usage for 01_embeddings.py:**
```bash
# Default: compute everything for the training pipeline
python 01_embeddings.py

# Compute only specific embedding types
python 01_embeddings.py --only esm2
python 01_embeddings.py --only saprot
python 01_embeddings.py --only substrates
python 01_embeddings.py --only esm2,substrates

# New substrates (for inference on untested substrates)
python 01_embeddings.py --substrates-json new_substrates.json --output-dir data/round3/inference/

# New mutations (for inference on new variants)
python 01_embeddings.py --mutations-csv new_mutations.csv --output-dir data/round3/inference/

# Override device / force recomputation
python 01_embeddings.py --device cuda --force
```

**Note on preprocessing:** Scaling and PCA are NOT done in 01 — they are applied at training time within each CV fold (fit on train, transform train+val) so there is no data leakage. The preprocessing pipeline objects are searched via hyperopt and saved with each model checkpoint.

### 02 — BNN1 Phase A: Position Classification (Sanity Check)
`02_bnn1_position_classification.py` — single-run training + CV evaluation
- [x] Load formaldehyde SSM data + ESM2 mutant embeddings only (no WT, no SaProt)
- [x] Feature matrix: ESM2 mutant residue embedding (1280-dim per mutation)
- [x] Target: position index (64-class classification)
- [x] All hyperparams via CLI args (defaults from config): `--hidden-dims`, `--prior-std`, `--dropout-rate`, `--learning-rate`, `--kl-weight`, `--esm-mut-scaler`, `--esm-mut-pca`
- [x] Preprocessing: optional scaling (none/standard/robust) → optional PCA (int, float variance fraction, or none), fit per CV fold
- [x] Stratified K-fold CV → accuracy, top-5 accuracy, confusion matrix, per-position accuracy
- [x] Train final model on all data → save model (`models/final_model.pt`) + preprocessing (`models/preprocessing.joblib`)
- [x] Plots: confusion matrix, per-position accuracy bar chart, training curves (all folds), loss decomposition (NLL/KL/beta)
- [x] Log results to `results/02_position_classification/`

`opt_02_position_classification.py` — Optuna hyperopt wrapper
- [x] Imports core functions from 02, loads data once
- [x] Searches over all hyperparams from config search spaces
- [x] Saves best params + ready-to-run command for 02 to `results/opt_02_position_classification/`

```bash
# Single run with defaults
python 02_bnn1_position_classification.py --device cuda:1

# Single run with overrides
python 02_bnn1_position_classification.py --hidden-dims '[128, 64]' --esm-scaler standard --esm-pca 64

# Hyperopt (50 trials)
python opt_02_position_classification.py --device cuda:1

# Re-run best config with full eval + final model
python 02_bnn1_position_classification.py --hidden-dims '<best>' --prior-std <best> ...
```

### 03 — BNN1 Phase B: Formaldehyde Activity Regression
`03_bnn1_formaldehyde_regression.py` — single-run training + CV evaluation
- [ ] Load formaldehyde SSM data + ESM2 WT and mutant embeddings (both used for regression)
- [ ] Feature matrix: [ESM2_wt_residue, ESM2_mut_residue] (2×1280 = 2560-dim)
- [ ] Target: log10(fold_change + epsilon) — continuous regression
- [ ] All hyperparams via CLI args (defaults from config): `--hidden-dims`, `--prior-std`, `--dropout-rate`, `--learning-rate`, `--kl-weight`, `--esm-wt-scaler`, `--esm-wt-pca`, `--esm-mut-scaler`, `--esm-mut-pca`
- [ ] Preprocessing: independent pipelines for WT and mutant features, fit per CV fold
- [ ] K-fold CV → regression metrics (MAE, RMSE, R², Spearman, per-position Spearman)
- [ ] Uncertainty evaluation: calibration curve, sharpness, coverage at 50/90/95% CI
- [ ] Train final model on all data → save model + preprocessing
- [ ] Plots: parity plot (predicted vs actual), residuals, calibration curve, per-position performance, training curves, loss decomposition, uncertainty vs error
- [ ] Log results to `results/03_formaldehyde_regression/`

`opt_03_formaldehyde_regression.py` — Optuna hyperopt wrapper
- [ ] Imports core functions from 03, loads data once
- [ ] Searches over model + preprocessing hyperparams from config search spaces
- [ ] Optimizes for minimum mean CV loss (negative Spearman or MAE)
- [ ] Saves best params + ready-to-run command for 03 to `results/opt_03_formaldehyde_regression/`

```bash
# Single run with defaults
python 03_bnn1_formaldehyde_regression.py --device cuda:1

# Single run with overrides
python 03_bnn1_formaldehyde_regression.py --hidden-dims '[128, 64]' --esm-wt-scaler standard --esm-mut-pca 0.95

# Hyperopt
python opt_03_formaldehyde_regression.py --device cuda:1

# Re-run best config with full eval + final model
python 03_bnn1_formaldehyde_regression.py --hidden-dims '<best>' --prior-std <best> ...
```

### 04 — Extract BNN1 Latent Embeddings
`04_extract_bnn1_embeddings.py`
- [x] Load trained BNN1 model + preprocessing pipelines from script 03
- [x] Load all unique mutations (formaldehyde + multi-substrate union, 1,274)
- [x] Preprocess with saved WT + mutant pipelines
- [x] MC-averaged forward pass through hidden layers only (penultimate layer)
- [x] Save latent embeddings as `processed/embeddings/bnn1_latent.npz` (keyed by mutation_string)
- [x] Save embedding uncertainty as `processed/embeddings/bnn1_latent_std.npz`
- [x] Save metadata to `processed/embeddings/bnn1_latent_meta.json`

```bash
# Default (uses config n_inference_samples)
python 04_extract_bnn1_embeddings.py --device cuda:1

# More MC samples for smoother embeddings
python 04_extract_bnn1_embeddings.py --n-samples 200

# Custom model directory
python 04_extract_bnn1_embeddings.py --model-dir results/03_formaldehyde_regression/models
```

### 05 — BNN2: Multi-Substrate Prediction ✅
`05_bnn2_common.py` — shared BNN2Model class, data loading, pairwise expansion, metrics, 15 plots
`05_bnn2_multi_substrate.py` — main evaluation script (4 split strategies)
`opt_05_bnn2.py` — Optuna hyperopt wrapper

**Architecture:** Composite BNN2 = BNN1 backbone (frozen/partial/trainable) + BNN2 head.
Pairwise mode: expand to (mutation, ref_substrate, target_substrate) triplets; aggregate via law of total variance.

- [x] Assemble features (pairwise mode only): [FC_ref, X_ref_substrate, X_target_substrate, BNN1_backbone, SaProt_zs]
- [x] Expand training data to (mutation, ref, target) triplets (split before expansion to prevent leakage)
- [x] BNN1 backbone freeze modes: `full` (frozen but stochastic), `partial` (last layer trainable), `none` (end-to-end)
- [x] **Split 1 — Random:** KFold CV over mutations (`--split random`)
- [x] **Split 2 — Positional:** GroupKFold by position (`--split position`)
- [x] **Split 3 — Substrate:** LeaveOneSubstrateOut (`--split substrate`) + Tanimoto distance analysis
- [x] **Split 4 — Single-shot:** 1 mutation per (substrate, position) for train, rest for val (`--split singleshot`)
- [x] Metrics: MAE, RMSE, R², Spearman ρ, NLPD, CRPS, calibration, sharpness, per-substrate, per-position
- [x] 15 plots per run: parity, residuals, calibration, uncertainty decomp, training curves, per-substrate/position bars, heatmaps, parity grid, substrate transfer matrix, Tanimoto-vs-perf, singleshot distributions
- [x] Logs to `results/05_bnn2/{split}/`

```bash
python 05_bnn2_multi_substrate.py --split random --device cuda:1
python 05_bnn2_multi_substrate.py --split substrate
python 05_bnn2_multi_substrate.py --split singleshot --n-singleshot-repeats 10
python opt_05_bnn2.py --split random --n-trials 30 --device cuda:1
```

### 06 — Select Mutations for Untested Substrates
`06_select_mutations.py`
- [ ] Load best BNN2 model
- [ ] For each untested substrate: predict all 200 SSM mutations
- [ ] In pairwise mode: predict using each known substrate as reference, aggregate
- [ ] Rank by predicted activity and/or UCB (mean + k*std)
- [ ] Select top ~10 per substrate, ensuring diversity
- [ ] Save selection to `results/06_selections/`

## Config

All hyperparameter ranges, feature flags, and experimental settings are defined in `config.yaml`:

See `config.yaml` for the full configuration. Key convention:
- `value`: active setting used by training scripts
- `search`: hyperopt search space (list of options); `search: null` means fixed at `value`
- BNN1 uses top-level `preprocessing.esm_mut` (not a BNN1-specific preprocessing section)
- All tunable params follow this pattern throughout the config

## Results Tracking

Each script logs results to `results/<script_name>/`:
- `metrics.json`: all evaluation metrics
- `config_used.yaml`: snapshot of config at runtime
- `best_hyperparams.json`: best hyperopt trial
- `figures/`: parity plots, calibration curves, etc.
- `models/`: saved model checkpoints

## Dependencies

- `aide_predict` (ESM2 embeddings, SaProt zero-shot scores)
- `torch` (BNN training)
- `optuna` (hyperparameter optimization)
- `pandas`, `numpy`, `scikit-learn` (data handling, metrics)
- `matplotlib`, `seaborn` (plotting)
- `rdkit` (Morgan fingerprints, MACCS keys)
- `mordred` (2D molecular descriptors)
- `pyyaml` (config)
- BNN module from `code/bnns/` (local)

## Resolved Decisions

- [x] **ESM2 model size:** 650M — precomputing, so may as well use large model
- [x] **Substrate embeddings:** Precompute all types (Morgan, MACCS, Mordred); hyperopt selects
- [x] **BNN1 freezing:** Hyperopt over none/partial/full freeze
- [x] **Target transform:** log10(FC + epsilon) — sign = beneficial/detrimental, MAE = multiplicative accuracy
- [x] **Pyruvate:** Use V83P as reference sequence; ESM_wt for pyruvate = V83P embeddings
- [x] **Inactive substrates:** Include with balancing as hyperparam; report metrics on active-only subset too
- [x] **Preprocessing:** Per-feature-group scaling + PCA as hyperparams; fitted objects saved
- [x] **Reference substrate mode:** Hyperopt over formaldehyde_only vs pairwise; pairwise gives ~5x data augmentation and multi-reference inference
- [x] **Substrate SMILES:** Yes, needed for Morgan/MACCS fingerprints. Canonical SMILES from PubChem stored in config and substrate_metadata.json.
- [x] **BNN1 features:** ESM-only — SaProt zero-shot is a global fitness score, not informative for position-level tasks
- [x] **Position classification:** Full 64-class — confusion patterns between structurally adjacent positions are themselves informative

## Open Questions

- [x] **BNN1 features:** ESM-only (no SaProt). SaProt is a global score, not useful for position-level classification.
- [x] **Position classification granularity:** Full 64-class classification (not coarse). The confusion matrix itself reveals if the model confuses structurally adjacent positions.
