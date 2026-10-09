#!/bin/bash

#SBATCH --job-name=pgd_vitl
#SBATCH --partition=tide
#SBATCH --nodes=1
#SBATCH --qos=intermediate
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --output=/scratch_tide/%u/logs/pgd_vitl-%A_%a.out
#SBATCH --error=/scratch_tide/%u/logs/pgd_vitl-%A_%a.err

module load python/3.14.2
module load pytorch/2.9.1

export TORCH_HOME=/scratch_tide/$USER/torch_cache
export PYTHONPATH=/scratch_tide/$USER/irp-hw2825/code:/scratch_tide/$USER/python_packages:$PYTHONPATH
export MPLCONFIGDIR=/scratch_tide/$USER/matplotlib

mkdir -p /scratch_tide/$USER/logs
mkdir -p /scratch_tide/$USER/outputs
mkdir -p "$MPLCONFIGDIR"

cd /scratch_tide/$USER/irp-hw2825

srun python3 code/src/experiments/hpc_pgd_comparison.py
