#!/bin/bash

step=1

echo "Step:$step" && sleep 2;

# CUDA_VISIBLE_DEVICES="3"

if [ $step -eq 1 ]; then
    #all_dataset="task1_dev  task1_dev_enroll  task1_enroll  task2_dev  task2_enroll  task3_enroll  train"
    all_dataset="task2_dev  task2_enroll  task3_enroll  train"
    for dataset in $all_dataset; do
        python preprocess.py data/$dataset data/$dataset/feats --num_workers=10
    done
fi

