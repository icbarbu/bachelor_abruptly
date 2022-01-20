#!/bin/bash


num_terminals=20
start_port=20000


for i in $(seq $num_terminals)
 do

     echo "";
     screen -d -m -S sim_${start_port} ../learning_robobos/src/setup-vrep.sh $start_port;
     sleep 0.1s;

     start_port=$((${start_port}+1));

 done


 # from VREP root run:
 # ../learning_robobos/src/run-vrep.sh
