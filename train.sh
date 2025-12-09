#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export USE_OPENMP=1 # prevents openblas to override OMP_NUM_THREADS

exp_name=$1
experiment=$2

source .venv/bin/activate
python train_knowledge_dist.py exp_name=$exp_name data=mix_all experiment=$experiment trainer=gpu launcher=local
# python train.py exp_name=$exp_name data=mix_all experiment=$experiment trainer=gpu launcher=local
