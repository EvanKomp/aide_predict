---
title: Sequence Optimization towards target function with BADASS
---

# Protein Optimization with BADASS

## Overview

AIDE integrates BADASS, an adaptive simulated annealing algorithm that efficiently explores protein sequence space to find variants with optimal properties. The BADASS algorithm was introduced in [this paper](https://www.biorxiv.org/content/10.1101/2024.10.25.620340v1) and has been adapted in AIDE to work with any of its protein prediction models.

## Installation

To use BADASS with AIDE, install the required dependencies:

```bash
pip install -r requirements-badass.txt
```

## Basic Usage

Here's a complete example of using BADASS with an ESM2 zero-shot predictor:

```python
from aide_predict import ProteinSequence, ESM2LikelihoodWrapper
from aide_predict.utils.badass import BADASSOptimizer, BADASSOptimizerParams

# 1. Define your protein sequence and prediction model
wt = ProteinSequence("MKLLVLGLPGAGKGTQAEKIVAAYGIPHISTGDMFRAAMKEGTPLGLQAKQYMDEGDLVPDEVTIGIVRERLSKDDCQNGFLLDGFPRTVAQAEALETMLADIASRLSALPPATQTRMILMVEDELRNLHRGQVLPSENTFRVADDNEETIKKIRQKYGNSSGVI")

# 2. Set up a prediction model
# Note that this can be a supervised model. In general, any ProteinModel or 
# scikit-learn pipeline whose input models are ProteinModelWrapper can be used.
model = ESM2LikelihoodWrapper(wt=wt)
model.fit([])  # No training needed for zero-shot model

# 3. Configure optimization parameters
params = BADASSOptimizerParams(
    num_mutations=3,       # Maximum mutations per variant
    num_iter=100,          # Number of optimization iterations
    seqs_per_iter=200      # Sequences evaluated per iteration
)

# 4. Create and run the optimizer
optimizer = BADASSOptimizer(
    predictor=model.predict,
    reference_sequence=wt,
    params=params
)

# 5. Run optimization
# This returns protein variants as well as scores from the optimizer 
# (which may be scaled and not equal to direct model outputs)
results_df, stats_df = optimizer.optimize()

# 6. Visualize the optimization process
optimizer.plot()

# 7. Print top variants
print(results_df.sort_values('scores', ascending=False).head(10))
```

## Optimization Parameters

BADASS behavior can be extensively customized through the `BADASSOptimizerParams` class:

```python
params = BADASSOptimizerParams(
    # Core parameters
    seqs_per_iter=500,              # Sequences per iteration
    num_iter=200,                   # Total optimization iterations
    num_mutations=5,                # Maximum mutations per variant
    init_score_batch_size=500,      # Batch size for initial scoring
    
    # Algorithm behavior
    temperature=1.5,                # Initial temperature
    cooling_rate=0.92,              # Cooling rate for SA
    seed=42,                        # Random seed
    gamma=0.5,                      # Variance boosting weight
    
    # Constraints
    sites_to_ignore=[1, 2, 3],      # Positions to exclude from mutation (1-indexed)
    
    # Advanced options
    normalize_scores=True,          # Normalize scores
    simple_simulated_annealing=False, # Use simple SA without adaptation
    cool_then_heat=False,           # Use cooling-then-heating schedule
    adaptive_upper_threshold=None,  # Threshold for adaptivity (float for quantile, int for top N)
    n_seqs_to_keep=None,            # Number of sequences to keep in results
    score_threshold=None,           # Score threshold for phase transitions (auto-computed if None)
    reversal_threshold=None         # Score threshold for phase reversals (auto-computed if None)
)
```

## How BADASS Works

BADASS operates through the following key mechanisms:

1. **Initialization**: Computes a score matrix of all single-point mutations
2. **Sampling**: Uses Boltzmann sampling to generate candidate sequences
3. **Scoring**: Evaluates candidates with the provided predictor function
4. **Phase detection**: Identifies when the optimizer has found a promising region
5. **Adaptive temperature**: Adjusts temperature to balance exploration/exploitation
6. **Score normalization**: Standardizes scores for better comparison

During optimization, BADASS maintains several tracking matrices:
- Score matrix for each amino acid at each position
- Observation counts for statistical significance
- Variance estimates for uncertainty quantification

## Optimization Results

The `optimize()` method returns two DataFrames:

1. `results_df`: Contains information about all evaluated sequences:
   - `sequences`: Compact mutation representation (e.g., "M1L-K5R")
   - `scores`: Predicted fitness scores
   - `full_sequence`: Complete protein sequence
   - `counts`: Number of times each sequence was evaluated
   - `num_mutations`: Number of mutations in each sequence
   - `iteration`: When the sequence was first observed

2. `stats_df`: Contains statistics for each iteration:
   - `iteration`: Iteration number
   - `avg_score`: Average score per iteration
   - `var_score`: Variance of scores
   - `n_eff_joint`: Effective number of joint samples
   - `n_eff_sites`: Effective number of sites explored
   - `n_eff_aa`: Effective number of amino acids explored
   - `T`: Temperature at each iteration
   - `n_seqs`: Number of sequences evaluated
   - `n_new_seqs`: Number of new sequences evaluated
   - `num_phase_transitions`: Cumulative number of phase transitions

## Analyzing Results

After optimization, BADASS offers several visualization and analysis options:

```python
# Plot optimization progress
optimizer.plot()  # Creates multiple plots showing optimization trajectory

# Save results to CSV
optimizer.save_results("optimization_run")

# Get best sequences
best_sequences = results_df.sort_values('scores', ascending=False).head(10)

# Create a ProteinSequences object from best variants
from aide_predict import ProteinSequences
top_variants = ProteinSequences(best_sequences['full_sequence'].tolist())

# Further analyze with other AIDE tools
from aide_predict.utils.plotting import plot_mutation_heatmap
mutations = [seq.get_mutations(wt)[0] for seq in top_variants]
scores = best_sequences['scores'].values
plot_mutation_heatmap(mutations, scores)
```

The visualization includes:
1. Statistics by iteration (scores, effective samples, temperature)
2. Score distributions vs temperature
3. Score density distributions across early and late iterations

## Performance Considerations

- BADASS evaluates thousands of sequences, so efficient predictors are important
- For computationally expensive models, consider:
  - Using model caching (via `CacheMixin`)
  - Reducing `seqs_per_iter` and `num_iter`
  - Using batch processing in custom predictors
  - Increasing `init_score_batch_size` for better initial sampling

## References

- BADASS: [Biophysically-inspired Adaptive Directed evolution with Simulated Annealing and Statistical testing](https://www.biorxiv.org/content/10.1101/2024.10.25.620340v1)
