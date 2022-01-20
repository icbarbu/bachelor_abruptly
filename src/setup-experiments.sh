#!/bin/bash
#set -e
#set -x


experiments=("foraging-TD")

runs=21
checkpoints=35
num_terminals=20
start_port=20000


possible_screens=()

# defines possible ports for screens
for port in $(seq $start_port $((${start_port}+${num_terminals}-1))); do
    possible_screens+=($port)
done


while true
	do

    printf "\n  >>>> loop ... \n"

    # discover free terminals

    active_screens=()
    free_screens=()
    active_experiments=()


    declare -a arr="$(screen -list)"


    for obj in ${arr[@]}; do

        if [[ "$obj" == *"exp_"* ]]; then
          printf "\n screen ${obj} is on\n"
          screen_port="$(cut -d'_' -f2 <<<"$obj")"
          active_experiments+=("$(cut -d'_' -f3 -<<<"$obj")_$(cut -d'_' -f4 -<<<"$obj")")
          active_screens+=($screen_port)
        fi
    done

   for possible_screen in "${possible_screens[@]}"; do
       if [[ ! " ${active_screens[@]} " =~ " ${possible_screen} " ]]; then
           free_screens+=($possible_screen)
     fi
      done


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

             # unfinished TODO change != to < ?
             if [ "$value" != "$checkpoints" ]; then

                # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
             fi
         else
             # not started yet
             echo " None";
              # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
         fi

        done
    done


    # spawns N experiments (N is according to free screens)

    max_fs=${#free_screens[@]}
    to_do=("${to_do[@]:0:$max_fs}")


    p=0
    for to_d in "${to_do[@]}"; do

        exp=$(cut -d'_' -f1 <<<"${to_d}")
        task=$(cut -d'-' -f1 <<<"${exp}")
        alg=$(cut -d'-' -f2 <<<"${exp}")

        screen -d -m -S exp_${free_screens[$p]}_${to_d} -L -Logfile experiments/"${to_d}.log" nice -n19 python3  src/human_experiments.py --task ${task} --algorithm ${alg} --experiment-name ${to_d} --robot-port ${free_screens[$p]} --real-time 1;

        printf "\n >> (re)starting exp_${free_screens[$p]}_${to_d} \n\n"
        p=$((${p}+1))

    done


   sleep 3600s;

done

# screen -ls  | egrep "^\s*[0-9]+.exp_" | awk -F "." '{print $1}' |  xargs kill
# killall screen
# screen -r naaameee
# screen -list

