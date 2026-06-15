# MDA HACL - ML-Guided Directed Evolution

Engineering MDA Hydroxyacyl-CoA Lyase (565 residues) for improved catalytic activity through multi-round ML-guided variant design.

## Project Workflow

### Round 0: Site-Saturation Mutagenesis (SSM)
Experimental saturation mutagenesis at 64 positions (~1,280 single-mutation variants). Measured fold-change activity vs WT. Best single mutation: A399G at 7.22x.

### Round 1: ML-Guided Multi-Mutation Design
Trained ESM2 (35M, mean-pool) + MLP ensemble on SSM data. Designed 3 plates of 96 variants each:
- **Plate 1 (Additive):** Combinatorial hypothesis testing with top 12 SSM mutations
- **Plate 2 (Greedy):** Top predicted mean activity from candidate library
- **Plate 3 (UCB):** Upper Confidence Bound (mean + 2*std) for exploration

Key discovery: strong epistasis. V83A and S407A are individually neutral (~1x) but together produce ~7x activity.

### Round 2: Epistasis-Aware Modeling
Retrained on SSM + Round 1 data (~1,570 variants). Hyperopt grid: 4 embedders x 3 model heads.

**Winner:** High-resolution ESM2 per-position embeddings + LASSO feature selection + MLP (MAE = 0.057 log10).

Variant generation:
- 2-3 mutation candidates filtered by prediction threshold (>7x WT)
- 5-mutation candidates generated via BADASS (Boltzmann Annealing)
- ESM1v filtering for evolutionary plausibility
- Network-based max-min distance selection for diversity

Final plate: 88 variants (predicted 6-30x WT).

### Post-Hoc Analysis
Activity distribution comparisons across rounds, epistasis quantification, diversity analysis, and selection of 24 variants for substrate scope testing.

### Round 3: Multi-Substrate Expansion (current)

Expanded from formaldehyde to a panel of 24 candidate substrates. 9 substrates were tested across the 10-position SSM library (200 mutations); 6 showed activity, 3 did not.

**SSM data across substrates** (10 positions x 20 AA = 200 mutations each):

| Substrate | Active/200 | Above WT | Best Mutation | Best Fold-Change |
|-----------|-----------|----------|---------------|-----------------|
| Formaldehyde | 152 | 47 | A399G | 7.22x |
| Acetaldehyde | 107 | 56 | A420S | 9.99x |
| Acetone | 110 | 57 | A399G | 10.59x |
| Glycoaldehyde | 42 | 20 | A399G | 6.60x |
| Phenylacetaldehyde | 109 | 53 | A420S | 5.52x |
| Pyruvate* | 5 | 4 | V83G | 8.22x |

\*Pyruvate: no WT activity; normalized to lowest-activity mutant (V83P).

**Round 3 goals:**

1. **Single-mutant selection for untested substrates (~15 remaining):** Use ML trained on the 6-substrate SSM dataset to predict which 10 mutations per untested substrate are most informative starting points. Once those are tested, use the sparse results + cross-substrate model to extrapolate predicted activity for all 200 SSM mutations per new substrate.

2. **Combinatorial variant selection for 5 new substrates:** From the existing freezer stock (Rd1 + Rd2 plates, 376 combinatorial variants), select ~5 variants per substrate to test, prioritizing glycoaldehyde and pyruvate. Variants should be non-overlapping across substrates to maximize information (~30 total).

## Directory Structure

```
MDA_HACL/
|-- code/                          # All scripts and notebooks
|   |-- round1.py / .ipynb         # Round 1 pipeline
|   |-- round2.py / .ipynb         # Round 2 pipeline (includes hyperopt)
|   |-- post_hoc_analysis.py/.ipynb# Activity analysis across rounds
|   |-- epistasis_analysis.ipynb   # Epistasis quantification
|
|-- data/
|   |-- experimental/              # Raw wetlab inputs
|   |   |-- ssm_data.csv           # SSM results (~1,280 variants)
|   |   |-- ssm_compiled.xlsx      # Original SSM Excel
|   |   |-- combined_data.csv      # SSM + Round 1 merged training set
|   |   |-- round0_1_compiled.xlsx # SSM + Rd1 original Excel
|   |   |-- round2_compiled.xlsx   # Rd2 original Excel
|   |
|   |-- round1/
|   |   |-- designed_plates/       # 96-well plates sent to wetlab
|   |   |   |-- plate1_additive.csv
|   |   |   |-- plate2_greedy.csv
|   |   |   |-- plate3_ucb.csv
|   |   |-- all_variants.csv       # All 288 designed variants
|   |   |-- greedy_candidates.csv  # Full greedy candidate library (11K)
|   |   |-- ucb_candidates.csv     # Full UCB candidate library (5K)
|   |   |-- best_params.json       # Round 1 model hyperparameters
|   |
|   |-- round2/
|   |   |-- designed_plate.csv     # Final 88-variant plate
|   |   |-- variants_with_predictions.csv
|   |   |-- libraries.csv          # Combined low + high order libraries
|   |   |-- badass_results.csv     # All BADASS-generated variants (~316K)
|   |   |-- badass_results_filtered.csv  # Passing prediction threshold
|   |   |-- badass_stats.csv       # BADASS iteration statistics
|   |   |-- epistasis.csv          # Per-mutation epistasis scores
|   |   |-- best_model.joblib      # Best Round 2 trained model
|   |
|   |-- round3/
|   |   |-- all_ssm_data.xlsx      # SSM activity for 6 substrates (200 muts each)
|   |                                #   Tabs: Formaldehyde, Acetaldehyde, Acetone,
|   |                                #         Glycoaldehyde, Phenylacetaldehyde, Pyruvate
|   |                                #   + Guide and Notes tab (full 24-substrate list)
|   |
|   |-- msa/
|   |   |-- mda_hacl.a2m           # EVcouplings MSA (5,126 sequences)
|   |
|   |-- intermediate/
|       |-- all_data_pca.pkl       # PCA-transformed embeddings
|       |-- all_data_pca_embeddings.pkl
|
|-- models/
|   |-- hyperopt/                  # Round 2 hyperopt grid search results
|       |-- hyperopt_results.csv   # Summary of all 12 combos
|       |-- *_hyperopt.joblib      # Fitted models (12 files)
|
|-- embedders/                     # ESM embedding caches (~137 GB)
|   |-- esm1v/                     # ESM1v masked marginal scores
|   |-- esm2/                      # ESM2 base embeddings
|   |-- esm2_individual/           # ESM2 per-position (high-res)
|   |-- esm2_maxpool/              # ESM2 max-pooled
|   |-- esm2_meanpool/             # ESM2 mean-pooled
|   |-- esm_high_res_feature_selected.joblib  # LASSO selector
|
|-- hf_datasets/                   # HuggingFace Arrow cached datasets
|   |-- ssm/                       # SSM variants with embeddings
|   |-- plate1/                    # Plate 1 additive variants
|   |-- greedy/                    # Greedy plate variants
|   |-- ucb/                       # UCB plate variants
|   |-- dopt/                      # D-optimal variants
|   |-- variants/                  # Full ~1M variant library
|   |-- variants_with_rd2_predictions/
|   |-- variants_with_rd2_passes/
|
|-- evcouplings/                   # EVcouplings analysis outputs
|   |-- alignment.a2m              # Input MSA
|   |-- coupling_scores.csv        # All pairwise coupling scores
|   |-- coupling_scores_longrange.csv
|   |-- top_ecs.txt                # Top evolutionary couplings
|   |-- enrichment.csv             # Per-position enrichment
|   |-- frequencies.csv            # Per-position AA frequencies
|   |-- model.bin                  # Fitted plmc model (510 MB)
|   |-- evzoom.json                # EVzoom visualization data
|   |-- couplings_standard.outcfg
|   |-- couplings_standard_plmc.outcfg
|   |-- iteration_table.csv
|   |-- pymol/                     # PyMOL visualization scripts
|       |-- draw_ec_lines.pml
|       |-- enrichment_spheres.pml
|       |-- enrichment_sausage.pml
|
|-- structure/                     # AlphaFold2 (ColabFold) predictions
|   |-- mda_hacl_b4fe6_relaxed_rank_001_*.pdb  # Best structure (pLDDT=94)
|   |-- mda_hacl_b4fe6_unrelaxed_rank_*.pdb    # All 5 unrelaxed models
|   |-- mda_hacl_b4fe6_scores_rank_*.json      # Per-residue confidence
|   |-- mda_hacl_b4fe6.a3m         # ColabFold MSA (12K sequences)
|   |-- mda_hacl_b4fe6_env/        # MMseqs2 search environment
|   |   |-- templates_101/         # 16 PDB template structures
|   |   |-- uniref.a3m, bfd.*.a3m  # Database MSAs
|   |   |-- msa.sh                 # MMseqs2 search script
|   |-- log.txt, config.json, cite.bibtex
|   |-- *_coverage.png, *_plddt.png, *_pae.png
|
|-- figures/
|   |-- round1/
|   |   |-- ssm_heatmap.png
|   |   |-- mutation_counts.png
|   |   |-- pca_design_space.png
|   |   |-- plate_activity_distributions.png
|   |   |-- plate_activity_distributions_pdf.png
|   |   |-- plate1_pred_vs_additive.png
|   |   |-- model_calibration.png
|   |   |-- model_parity.png
|   |-- round2/
|   |   |-- hyperopt_results.png
|   |   |-- best_parity.png
|   |   |-- additive_error.png
|   |   |-- data_vs_additive_model.png
|   |   |-- data_vs_multiplicative_model.png
|   |   |-- lasso_alpha_search.png
|   |   |-- mutcv_vs_kfold.png
|   |   |-- aa_preferences/        # Per-position AA preference plots
|   |   |-- badass/                 # BADASS sampling visualizations
|   |-- structure/
|       |-- target_sites.pse       # PyMOL session with target sites
|
|-- archive/                       # Old/superseded code
    |-- round1_original.py         # Pre-cleanup Round 1 script
    |-- round1.ipynb               # Early Round 1 notebook
    |-- round1_clean.pdf / .txt    # Exported notebook
    |-- in_silico_mutagenesis.ipynb
    |-- predictions.csv / .png
    |-- esm2_embedder/             # Old embedding cache (1 MB)
```

## Key Results

| Metric | Value |
|--------|-------|
| Best single mutation | A399G (7.22x) |
| Best Round 1 variant | V83G\|A420S\|S425A (~9x) |
| Strongest epistasis | V83A + S407A: individually ~1x, together ~7x |
| Best model | High-res ESM2 + LASSO + MLP (MAE = 0.057 log10) |
| Round 2 predictions | 6-30x WT for 88-variant plate |
| Substrates with activity | 6 of 9 tested (24 total candidates) |
| Best non-native substrate | Acetone: A399G at 10.59x |
| Freezer stock available | 376 combinatorial variants (Rd1 + Rd2) |

## Current data landscape
- Formaldehyde SSM @ 64 positions, 10 selected as important positions (1280 total data points)
- Formaldehyde combinatorial 2-5 order mutants, 376 total
- 8 other substrates: full SSM of 10 selected positions (200 points each for 1600 total), 5 substartes with nonzero activity.

## Running the Code

All scripts use relative paths and assume they are run from the `code/` directory:

```bash
cd data/projects/MDA_HACL/code
python round1.py
python round2.py
```

Or open the corresponding `.ipynb` notebooks in Jupyter.
