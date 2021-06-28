#!/bin/bash
# set -e

batch_path='/data/sunjiaming/neucon_ar_demo/tianren_lowtexture/new_batch4'

echo "Running NeuralRecon demo.."
eval "$(conda shell.bash hook)"
conda activate neucon


for folder in $batch_path/*
do
    echo "==== Reconstructing $folder... ===="
    python demo.py --cfg config/demo.yaml TEST.PATH $folder
    sleep 1s
done

echo "Finished."
