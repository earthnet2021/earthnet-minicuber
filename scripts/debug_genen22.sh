#!/bin/bash

function pwait() {
    while [ $(jobs -p | wc -l) -ge $1 ]; do
        sleep 1
    done
}

conda activate dea
for i in {0..50000}
do
  nice python scripts/generate_en22.py $i local train /Net/Groups/BGI/scratch/DeepCube/UC1/sampled_minicubes_v2_200000.csv /Net/Groups/BGI/work_1/scratch/DeepCube/en22/ & #-m cProfile -o scripts/genen22_localtrain_$i.prof 
  sleep 1
  pwait 40
done

wait