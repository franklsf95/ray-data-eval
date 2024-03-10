#!/bin/bash

NUM_RUNS=3
PERCENTAGE=5

for (( i=1; i<=NUM_RUNS; i++ ))
do
    echo "[Run #$i]"
    python create_sample_dataset.py --percentage $PERCENTAGE
    python image_loader_microbenchmark.py --data-root /home/ubuntu/image-data-$PERCENTAGE-percent/ILSVRC/Data/CLS-LOC/
done

echo "[Completed]"
