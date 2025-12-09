#!/bin/bash

srun -K \
  --container-image=/netscratch/$USER/enroot/hamer-dev.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds-av:/ds-av,/ds:/ds:ro,/home/$USER:/home/$USER \
  --container-save=/netscratch/$USER/enroot/hamer-dev.sqsh \
  --container-workdir="$(pwd)" \
  --container-remap-root \
  -p V100-32GB \
  --gpus=1 \
  --mem=98304 \
  --cpus-per-gpu=8 \
  --job-name="Short Interactive Debugging Session" \
  --immediate=3600 \
  --time=01:00:00 \
  --pty bash -i
