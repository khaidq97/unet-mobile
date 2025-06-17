#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
python training/train.py \
  --config configs/config.yml