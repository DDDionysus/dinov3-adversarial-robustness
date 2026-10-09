#!/bin/bash

#SBATCH --job-name=dinov3-7b
#SBATCH --partition=tide
#SBATCH --nodes=1
#SBATCH --qos=intermediate
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch_tide/%u/logs/dinov3-7b-%A_%a.out
#SBATCH --error=/scratch_tide/%u/logs/dinov3-7b-%A_%a.err

module load python/3.14.2
module load pytorch/2.9.1

export IMAGENET_VAL_PATH="path to imagenet dataset"
export DINOV3_7B_CHECKPOINT="path to dinov3 7b checkpoint"
export EXPERIMENT_OUTPUT_DIR="path to output"

# optional environment configuration
# export TORCH_HOME=/path/to/torch_cache
# export PYTHONPATH=/path/to/repository/code:$PYTHONPATH

mkdir -p /scratch_tide/$USER/logs
mkdir -p "$EXPERIMENT_OUTPUT_DIR"

cd "/scratch_tide/$USER/dinov3-adversarial-robustness/code"

export EXPERIMENT_STAGE="$1"

if [ -z "$EXPERIMENT_STAGE" ]; then
    echo "Missing experiment stage"
    exit 1
fi

echo "Running stage: $EXPERIMENT_STAGE"

srun python3 -m src.extension.representation.dinov3_7b_experiment
