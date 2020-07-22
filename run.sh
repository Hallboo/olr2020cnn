#!/bin/bash

step=1

echo "Step:$step" && sleep 2;

# CUDA_VISIBLE_DEVICES="3"

if [ $step -eq 1 ]; then
    python preprocess.py data/train/ data/train/feats --num_workers=10
fi

