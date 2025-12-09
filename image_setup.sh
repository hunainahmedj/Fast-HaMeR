#!/bin/bash

apt update && apt upgrade -y && apt clean

apt-get install -y --no-install-recommends --fix-missing \
  gcc g++ \
  make \
  python3 python3-dev python3-pip python3-venv python3-wheel \
  espeak-ng libsndfile1-dev \
  git \
  wget \
  ffmpeg \
  libsm6 libxext6 \
  libglfw3-dev libgles2-mesa-dev &&
  rm -rf /var/lib/apt/lists/*

python3 -m venv .venv
source .venv/bin/activate

# Activate virtual environment and install dependencies:
# REVIEW: We need to install/upgrade wheel and setuptools first because otherwise installation fails:
pip install --upgrade wheel setuptools

pip install torch torchvision torchaudio

# REVIEW: Numpy is installed separately because otherwise installation fails:
pip install numpy

pip install opencv-python

# Install gdown (used for fetching scripts):
pip install gdown

pip install -e .[all]
pip install -v -e third-party/ViTPose
