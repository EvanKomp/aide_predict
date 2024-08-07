#!/bin/bash
#SBATCH --partition=standard
#SBATCH --account=proteinml
#SBATCH --job-name=dvc_stage
#SBATCH --output=dvc_stage.out
#SBATCH --nodes=1
#SBATCH --mem=100GB              # memory per node
#SBATCH --time=0-12:30            # Max time (DD-HH:MM)

############### CARBON TRACKING
# start codecarbon
PID=$(/projects/bpms/ekomp_tmp/software/carbon/start_tracker.sh)
echo main
echo $PID
# Save its PID

# Define a cleanup function
cleanup() {
    echo "Cleaning up..."
    kill -SIGINT $PID
    sleep 10
}
# # Set the trap
trap cleanup EXIT
#################### END CARBON TRACKING
dvc repro -s $1 --force
