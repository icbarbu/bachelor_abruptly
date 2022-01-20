#!/bin/bash


experiments=("h_foraging-TD" "foraging-TD")
runs=21
poses=7
repetitons=3

for experiment in ${experiments[@]}; do

    for r in $(seq $runs)
     do
        for p in $(seq 0 $((${poses}-1)) )
         do
               for i in $(seq $repetitons )
                do
                    python3 src/final_tests.py --pos ${p} --experiment-name "${experiment}_${r}"
                done
         done
     done

 done