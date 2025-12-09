#! /bin/bash

srun -K \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_23.12-py3.sqsh \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/home/$USER:/home/$USER \
  --container-save=/netscratch/$USER/enroot/hamer-dev.sqsh \
  --container-workdir="$(pwd)" \
  --mem=98304 \
  --time=04:00:00 \
  --immediate=3600 \
  --job-name="image_setup" \
  image_setup.sh
