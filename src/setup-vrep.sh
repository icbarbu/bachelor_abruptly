#!/bin/bash
set -e
set -x

#./vrep.app/Contents/MacOS/vrep -h ../../../scenes/robobo_food_arena.ttt -gREMOTEAPISERVERSERVICE_"${1}"_FALSE_TRUE


# runs vrep inside a fake display

xvfb-run ./coppeliaSim.sh -h scenes/robobo_food_arena.ttt  -gREMOTEAPISERVERSERVICE_"${1}"_FALSE_TRUE


#xvfb-run ./coppeliaSim.sh -h scenes/robobo_avoid_arena.ttt  -gREMOTEAPISERVERSERVICE_"${1}"_FALSE_TRUE


