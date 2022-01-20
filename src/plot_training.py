import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from statannot import add_stat_annotation
from itertools import combinations
import sys

pd.set_option("display.max_rows", 99999)


class PlotData:

    def __init__(self, anal, experiments, runs):
        self.anal = anal
        self.experiments = experiments
        self.runs = runs

        self.dir = 'experiments/anal/'
        self.measures = ['steps', 'total_success', 'rewards']
        # self.metrics = ['max', 'mean', 'min', 'median']
        self.measures_limits = [[150, 210],  [0, 8], [0, 800]]
        self.metrics = ['median']
        self.clrs = ['#FF3333', '#006600']

        plt.style.use('classic')

    def lines(self):

        full_data_agreg = pd.read_csv(f'{self.dir}{self.anal}_full_data_agreg.csv')
        pprint.pprint(full_data_agreg)

        for idx_measure, measure in enumerate(self.measures):

            for metric in self.metrics:
                font = {'font.size': 20}
                plt.rcParams.update(font)
                fig, ax = plt.subplots()

                for idx_experiment, experiment in enumerate(self.experiments):

                    data = full_data_agreg[full_data_agreg['experiment'] == experiment]

                    ax.plot(data['checkpoint'], data[f'{measure}_{metric}'], label=experiment, c=self.clrs[idx_experiment])

                    if metric == 'median':
                        ax.fill_between(data['checkpoint'],
                                         data[f'{measure}_q25'],
                                         data[f'{measure}_q75'],
                                        alpha=0.3, facecolor=self.clrs[idx_experiment])

                    if metric == 'mean':
                        ax.fill_between(data['checkpoint'],
                                        data[f'{measure}_{metric}'] - data[f'{measure}_std'],
                                        data[f'{measure}_{metric}'] + data[f'{measure}_std'],
                                        alpha=0.3, facecolor=self.clrs[idx_experiment])

                    ax.set_ylim((self.measures_limits[idx_measure][0], self.measures_limits[idx_measure][1]))
                    plt.xlabel('Stage')
                    plt.ylabel(f'{measure}_{metric}')
                    #plt.title(self.anal)
                    # ax.legend()

                    if measure.find('total_success') != -1:
                        plt.plot([1, 35], [7, 7], 'k--', linewidth=1.5)

                plt.grid()
                plt.savefig(f'{self.dir}{self.anal}_{measure}_{metric}.png', bbox_inches='tight')

    def best(self):

        full_data_agreg = pd.read_csv(f'{self.dir}{self.anal}_full_data.csv')
        data_filtered = pd.DataFrame()

        for experiment in self.experiments:
            for run in self.runs:

                exp_run = full_data_agreg[(full_data_agreg['experiment'] == experiment) & (full_data_agreg['run'] == run)]
                exp_run = exp_run[(exp_run['total_success'] == exp_run["total_success"].max())]
                exp_run = exp_run[exp_run['steps'] == exp_run["steps"].min()]
                # TODO: in case of h_, get max checkpoint ONLY among human stages,
                #  so to avoid having to run manually the cases of non-improvement
                exp_run = exp_run[exp_run['checkpoint'] == exp_run["checkpoint"].max()]

                data_filtered = pd.concat([data_filtered, exp_run], axis=0)

        pprint.pprint(data_filtered)
        data_filtered.to_csv(f'{self.dir}{self.anal}_best_checkpoints.csv')


#analysis = {  'foraging-TD':  ["foraging-TD"] }
analysis = {  'h_foraging-TD':  ["h_foraging-TD"] }

runs = range(1, 21+1)
for anal in analysis:
    cd = PlotData(anal, analysis[anal], runs=runs )
    cd.lines()
    cd.best()








