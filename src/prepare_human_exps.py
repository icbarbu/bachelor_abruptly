import numpy as np
import pprint
import pandas as pd
import shutil, os
import glob

pd.set_option("display.max_rows", 99999)


class PlotData:

    def __init__(self, experiment, runs):
        self.experiment = experiment
        self.dir = 'experiments/'
        self.runs = runs
        self.robot_port = 19997

    def replicate(self):

        full_data_agreg = pd.read_csv(f'{self.dir}anal/{self.experiment}_full_data.csv')

        out_file = open(f'experiments/run_human_experiments.txt', "w")
        for run in self.runs:
            exp_run = full_data_agreg[
                (full_data_agreg['experiment'] == experiment) & (full_data_agreg['run'] == int(run))]
            exp_run = exp_run[(exp_run['total_success'] == exp_run["total_success"].max())]
            exp_run = exp_run[exp_run['total_hurt'] == exp_run["total_hurt"].min()]
            exp_run = exp_run[exp_run['steps'] == exp_run["steps"].min()]
            exp_run = exp_run[exp_run['checkpoint'] == exp_run["checkpoint"].max()]
            print(exp_run)

            experiment_name_new = f"h_{experiment}_{run}"
            os.mkdir(f'{self.dir}{experiment_name_new}')

            for file in glob.glob(f'{self.dir}{experiment}_{run}/*'):
                if file[-4:-1] != '.tx':
                    checkpoint = int(file.split('_')[3].split('.')[0])
                    if checkpoint <= exp_run["checkpoint"].max():
                        file_name = file.split('/')[2]
                        shutil.copyfile(f'{file}', f'{self.dir}{experiment_name_new}/{file_name}')

            command = f'python3 src/human_experiments.py --experiment-name {experiment_name_new}  --robot-port {self.robot_port}' \
                f' --human-interference 1 --real-time 1 \n'
            out_file.write(command)
        out_file.close()


experiments = ['foraging-TD']


for experiment in experiments:
    cd = PlotData(experiment, runs=range(1, 21+1))
    cd.replicate()








