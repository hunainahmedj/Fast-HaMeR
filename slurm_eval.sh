#!/bin/bash

srun -K \
  --container-image=/netscratch/$USER/enroot/hamer-dev.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds-av:/ds-av,/ds:/ds:ro,/home/$USER:/home/$USER \
  --container-workdir="$(pwd)" \
  --container-remap-root \
  -p V100-16GB \
  --gpus=1 \
  --mem=98304 \
  --cpus-per-gpu=8 \
  --job-name="HaMeR Mobilenet Large KD Full Evaluation" \
  --immediate=3600 \
  --time=01:00:00 \
  bash eval.sh $1
