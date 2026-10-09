#!/bin/bash

#SBATCH --job-name=other_datasets
#SBATCH --partition=tide
#SBATCH --nodes=1
#SBATCH --qos=intermediate
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch_tide/%u/logs/other_datasets-%j.out
#SBATCH --error=/scratch_tide/%u/logs/other_datasets-%j.err
#SBATCH --array=0-3

module load python/3.14.2
module load pytorch/2.9.1

export TORCH_HOME=/scratch_tide/$USER/torch_cache
export PYTHONPATH=/scratch_tide/$USER/irp-hw2825/code:/scratch_tide/$USER/python_packages:$PYTHONPATH
export MPLCONFIGDIR=/scratch_tide/$USER/matplotlib

mkdir -p /scratch_tide/$USER/logs
mkdir -p /scratch_tide/$USER/outputs
mkdir -p "$MPLCONFIGDIR"

cd /scratch_tide/$USER/irp-hw2825

srun python3 code/src/experiments/hpc_other_datasets.py
