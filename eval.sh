#!/bin/bash
# Usage: ./eval.sh EXPERIMENT_NAME [CHECKPOINT_PATH]
# If CHECKPOINT_PATH is omitted, uses ./logs/train/runs/EXPERIMENT_NAME/checkpoints/last.ckpt

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_OPENMP=1 # prevents openblas to override OMP_NUM_THREADS

source .venv/bin/activate
experiment_name=$1
checkpoint="${2:-./logs/train/runs/${experiment_name}/checkpoints/last.ckpt}"
metrics_folder="results/"
results_folder="results/"$experiment_name

python eval.py --dataset 'HO3D-VAL' --checkpoint "$checkpoint" --exp_name $experiment_name --results_folder $results_folder --metrics_folder $metrics_folder --batch_size 1 --efficient_hamerss