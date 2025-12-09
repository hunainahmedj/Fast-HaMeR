#!/bin/bash

srun -K \
  --container-image=/netscratch/$USER/enroot/hamer-dev.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds-av:/ds-av,/ds:/ds:ro,/home/$USER:/home/$USER \
  --container-workdir="$(pwd)" \
  --container-remap-root \
  -p RTXA6000 \
  --gpus=1 \
  --mem=98304 \
  --cpus-per-gpu=8 \
  --job-name="$3" \
  --time=72:00:00 \
  bash train.sh $1 $2
