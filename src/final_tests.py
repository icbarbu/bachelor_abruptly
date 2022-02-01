#final_tests.py
import os
os.environ["KMP_WARNINGS"] = "FALSE"
import pandas as pd
from config import Config
import pprint
import sys
import time
from experiment_manager import ExperimentManager

from foraging import ForagingEnv as ForagingEnv

from stable_baselines3 import TD3
from stable_baselines3 import PPO


def load_td(name, env):
    # print(config)
    # td = TD3(config, env)
    # td.load(dir)
    # return td
    return TD3.load(name, env)



def run_test(log_food_print, experiment, run, load, env, checkpoint):
    checkpoint = int(checkpoint)
    config.experiment_name = f'{experiment}'
    ExperimentManager(config=config,
                      model=None,
                      environment=env,
                      load=load,
                      log_food_print=log_food_print
                     ).test_policy(experiment, run, checkpoint)


def extract_info(experiment):
    experiment = "foraging-TD"
    algorithm = experiment.split("-")[1]
    task = experiment.split("-")[0]
    
    print(algorithm, task)
    
    if algorithm == 'TD':
        load = load_td

    if task == 'foraging':
        env = ForagingEnv(config=config)

    return load, env


###
config = Config()
config = config.parser.parse_args()
config.robot_port =  19997 #20020
config.train_or_test = 'test'

# for final tests, run once with sim and once with hard
config.sim_hard = 'sim'
#config.sim_hard = 'hard'

log_food_print = True
#choice or bests
tests_type = 'best'
#tests_type = 'choice'

experiments = ['6']
# runs=20
# for i in range(1, runs+1):
#     experiments.append('foraging-TD_'+str(i))


if tests_type == 'choice':

    experiment = 'foraging-TD'
    run = '11'
    checkpoint = 35

    # load, env = extract_info(experiment)
    # run_test(log_food_print, experiment, run, load, env, checkpoint)


else:
    for exp in experiments:
        print(exp)
        # aux_str = exp.split('_')
        # experiment = aux_str[0]
        # run = aux_str[1]
        experiment = exp

        full_data_agreg = pd.read_csv(f'experiments/anal/{experiment}_full_data.csv')
        exp_run = full_data_agreg[(full_data_agreg['experiment'] == int(experiment))] #& (full_data_agreg['run'] == int(run))
        exp_run = exp_run[exp_run['distance'] == exp_run["distance"].min()]
        exp_run = exp_run[exp_run['total_hurt'] == exp_run["total_hurt"].min()]
        exp_run = exp_run[(exp_run['total_success'] == exp_run["total_success"].max())]
        exp_run = exp_run[exp_run['steps'] == exp_run["steps"].min()]
        exp_run = exp_run[exp_run['checkpoint'] == exp_run["checkpoint"].max()]

        pprint.pprint(exp_run)

        for row in exp_run.iterrows():

            checkpoint = row[1]["checkpoint"]
            run = row[1]["run"]

            load, env = extract_info(experiment)
            print('checkpoint', checkpoint)

            run_test(log_food_print, experiment, run, load, env, checkpoint)
