#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --account=proteinml
#SBATCH --job-name=bnn_sens
#SBATCH --output=slurm_logs/sensitivity_%A_%a.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --time=36:00:00            # ~570 sweep items × ~1-2 min each; 36h leaves headroom
#SBATCH --cpus-per-task=30
#SBATCH --qos=high
#SBATCH --array=0-4                # one task per metric (see METRICS[] below)

# ─────────────────────────────────────────────────────────────────────────────
# Training-substrate sensitivity sweep for every hyperopt study in new_opt_05.
#
# For each metric we optimised (post per-substrate-semantics bugfix): launch
# new_05c_bnn2_sensitivity.py with that study's best_hyperparams.json. Each
# array task gets its own GPU and writes to its own sweep root, so the 5
# metrics run fully in parallel.
#
# The sensitivity script itself iterates over (held-out substrate × training
# subset) combinations, calling new_05_bnn2_train.py + new_05b_bnn2_score.py
# per item. It auto-skips items whose metrics.json already exists, so this
# script is idempotent — re-submit the same array task after a timeout to
# resume.
#
# Submit:    sbatch data/projects/MDA_HACL/slurm_submit_sensitivity.sh
# Resume:    sbatch --array=<missing_ids> data/projects/MDA_HACL/slurm_submit_sensitivity.sh
# Summary-only re-aggregate (no compute):
#            see SUMMARY_ONLY block at the bottom of this file
# Monitor:   squeue -u $USER
# ─────────────────────────────────────────────────────────────────────────────

module load conda
module load cuda
conda activate aidep

PROJECT_DIR=/kfs2/projects/proteinml/repos/aide_predict/data/projects/MDA_HACL
cd "$PROJECT_DIR" || exit 1
mkdir -p slurm_logs

# Same 5 metrics as the hyperopt array. Order/index match for sanity.
METRICS=(
    "ndcg"
    "spearman_rho"
    "top3_recovery"
    "per_position_ndcg_mean"
    "per_position_top3_recovery_mean"
)

METRIC="${METRICS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$METRIC" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no matching metric"
    exit 2
fi

HP="results/new_opt_05/all_substrates_${METRIC}/best_hyperparams.json"
OUT_DIR="results/new_05_bnn2/sensitivity/${METRIC}"

echo "============================================================"
echo "task ${SLURM_ARRAY_TASK_ID}: ${METRIC}"
echo "  hyperparams: ${HP}"
echo "  output:      ${OUT_DIR}"
echo "  device:      cuda:0"
echo "  started:     $(date)"
echo "============================================================"

if [ ! -f "$HP" ]; then
    echo "ERROR: missing hyperparams at ${HP}"
    echo "       Make sure the corresponding hyperopt array task has finished."
    exit 3
fi

# Trailing "--" forwards --hyperparams through to new_05_bnn2_train.py via the
# sensitivity script's REMAINDER positional. Any other train flags can go here
# too (e.g. extra --skip-* toggles).
python code/round3/new_05c_bnn2_sensitivity.py \
    --output-dir "${OUT_DIR}" \
    --device cuda:0 \
    --acq-sigma within_epi_ale \
    -- \
    --hyperparams "${HP}"

echo "============================================================"
echo "task ${SLURM_ARRAY_TASK_ID} (${METRIC}) finished: $(date)"
echo "============================================================"

# ── Re-aggregate plots only (no GPU compute) ─────────────────────────────────
# If the heavy sweep has already run for a metric and you only want to
# regenerate the summary plots (e.g. after editing new_05c_bnn2_sensitivity.py):
#
#   python code/round3/new_05c_bnn2_sensitivity.py \
#       --output-dir results/new_05_bnn2/sensitivity/<metric> \
#       --summary-only
#
# This is cheap (no CUDA) — run it directly on a login node, no SLURM needed.
