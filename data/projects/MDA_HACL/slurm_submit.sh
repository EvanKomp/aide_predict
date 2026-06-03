#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --account=proteinml
#SBATCH --job-name=bnn_cv
#SBATCH --output=slurm_cv.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --time=20:00:00            # 5 metrics × ~2-3h CV each, sequential
#SBATCH --cpus-per-task=30
#SBATCH --qos=high

#source ~/.bashrc
module load conda
module load cuda
conda activate aidep

# ─────────────────────────────────────────────────────────────────────────────
# Full substrate-split CV + analysis for every hyperopt study in new_opt_05.
#
# For each metric we optimised (post per-substrate-semantics bugfix): run the
# substrate-split CV with that study's best_hyperparams.json
# (new_05_bnn2_train.py), then the matched-pair scoring + analysis plot
# (new_05b_bnn2_score.py). Each metric writes to its own folder under
# results/new_05_bnn2/substrate/<run_id>/.
#
# Names dropped the "_v2" suffix — these are the post-bugfix run dirs and
# should not collide with the pre-bugfix _v2 folders (delete those first if
# you want them gone). The per-metric guard below skips any study whose
# best_hyperparams.json hasn't been written yet, so this script is safe to
# submit while one or two hyperopt array tasks are still finishing.
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR=/kfs2/projects/proteinml/repos/aide_predict/data/projects/MDA_HACL
cd "$PROJECT_DIR" || exit 1

# "<hyperopt study dir>|<CV run-id / output folder>"
RUNS=(
    "all_substrates_ndcg|ndcg"
    "all_substrates_per_position_ndcg_mean|per_position_ndcg_mean"
    "all_substrates_per_position_top3_recovery_mean|per_position_top3_recovery_mean"
    "all_substrates_spearman_rho|spearman_rho"
    "all_substrates_top3_recovery|top3_recovery"
)

for entry in "${RUNS[@]}"; do
    opt_dir="${entry%%|*}"
    run_id="${entry##*|}"
    hp="results/new_opt_05/${opt_dir}/best_hyperparams.json"
    run_dir="results/new_05_bnn2/substrate/${run_id}"

    echo "============================================================"
    echo "=== CV + analysis: ${run_id}"
    echo "============================================================"

    if [ ! -f "$hp" ]; then
        echo "SKIP ${run_id}: missing ${hp}"
        continue
    fi

    # (1) full substrate-split CV
    python code/round3/new_05_bnn2_train.py \
        --split substrate \
        --hyperparams "$hp" \
        --run-id "$run_id" \
        --device cuda:0

    # (2) analysis + plot
    python code/round3/new_05b_bnn2_score.py \
        --run-dir "$run_dir"
done
