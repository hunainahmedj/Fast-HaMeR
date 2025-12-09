#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_OPENMP=1 # prevents openblas to override OMP_NUM_THREADS

source .venv/bin/activate
experiment_name=$1
# checkpoint="/home/jillani/datasets/hamer_stuff/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
# checkpoint="/home/jillani/Dev/hamer-experiments/logs/train/runs/"$experiment_name"/checkpoints/last.ckpt"
checkpoint="/netscratch/jillani/hamer-experiments/logs/train/runs/"$experiment_name"/checkpoints/last.ckpt"
metrics_folder="results/"
results_folder="results/"$experiment_name

python eval.py --dataset 'HO3D-VAL' --checkpoint $checkpoint --exp_name $experiment_name --results_folder $results_folder --metrics_folder $metrics_folder --batch_size 1 --efficient_hamer
