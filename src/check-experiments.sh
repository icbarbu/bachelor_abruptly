#!/bin/bash
#set -e
#set -x



experiments=("foraging-TD")
runs=21
checkpoints=6


# discover unfinished experiments

to_do=()
for i in $(seq $runs)
do
    run=$(($i))

    for experiment in "${experiments[@]}"
    do

     printf  "\n${experiment}_${run} \n"
     file="experiments/${experiment}_${run}/log.txt";

     #check experiments status
     if [[ -f "$file" ]]; then

         value=$(grep "SAVED checkpoint" $file|tail -n1|sed -E "s/\SAVED checkpoint ([0-9]+).*/\1/g");
         echo " ${value} "

     else
         # not started yet
         echo " None";
     fi

    done
done

