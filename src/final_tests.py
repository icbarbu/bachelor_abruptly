import os
os.environ["KMP_WARNINGS"] = "FALSE"
import pandas as pd
from config import Config
import pprint
import sys
import time
from experiment_manager import ExperimentManager

from foraging import ForagingEnv as ForagingEnv

from TD3_loop import TD3_loop


def load_td(dir1, dir2, env, config):
    td = TD3_loop(env, config)
    td.load(dir1, dir2)
    return td


def run_test(log_food_print, experiment, run, load, env, checkpoint):
    config.experiment_name = f'{experiment}_{run}'
    ExperimentManager(config=config,
                      model=None,
                      environment=env,
                      load=load,
                      log_food_print=log_food_print
                     ).test_policy(experiment, run, checkpoint)


def extract_info(experiment):

    algorithm = experiment.split("-")[1]
    task = experiment.split("-")[0]

    if algorithm == 'TD':
        load = load_td

    if 'foraging' in task:
        env = ForagingEnv(config=config)

    return load, env


###
config = Config()
config = config.parser.parse_args()
config.robot_port = 19997
config.train_or_test = 'test'

config.sim_hard = 'sim'
log_food_print = True
tests_type = 'bests'

# hardware tests are made using manual choice, because it is not practical to automatize it
#config.sim_hard = 'hard'
#tests_type = 'choice'
if tests_type == 'choice':

    experiment = 'foraging-TD'
    run = '11'
    checkpoint = 35
    config.pos = 1

    load, env = extract_info(experiment)
    run_test(log_food_print, experiment, run, load, env, checkpoint)

else:

    exp = config.experiment_name
    if exp[0] == 'h':
        aux_str = exp.split('_')
        experiment = aux_str[0]+'_'+aux_str[1]
        run = aux_str[2]
    else:
        aux_str = exp.split('_')
        experiment = aux_str[0]
        run = aux_str[1]

    full_data_agreg = pd.read_csv(f'experiments/anal/{experiment}_full_data.csv')

    exp_run = full_data_agreg[
        (full_data_agreg['experiment'] == experiment) & (full_data_agreg['run'] == int(run))]
    exp_run = exp_run[(exp_run['total_success'] == exp_run["total_success"].max())]
    exp_run = exp_run[exp_run['steps'] == exp_run["steps"].min()]
    # TODO: in case of h_, get max checkpoint ONLY among human stages,
    #  so to avoid having to run manually the cases of non-improvement
    exp_run = exp_run[exp_run['checkpoint'] == exp_run["checkpoint"].max()]

    pprint.pprint(exp_run)

    for row in exp_run.iterrows():

        checkpoint = row[1]["checkpoint"]
        run = row[1]["run"]

        load, env = extract_info(experiment)
        print('checkpoint', checkpoint)

        run_test(log_food_print, experiment, run, load, env, checkpoint)

