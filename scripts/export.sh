#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
python training/export.py \
  --model_path logs/unet-nano/best_model.keras \
  --output_dir logs/export/unet-nano \
  --type tfjs