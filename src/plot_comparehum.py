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
        self.full_data_agreg = {}

        self.dir = 'experiments/anal/'
        self.measures = ['steps', 'total_success']
        self.measures_limits = [[100, 210],  [-0.5, 8]]
        self.clrs = ['#FF3333', '#006600']

        plt.style.use('classic')

    def lines(self):

        for experiment in self.experiments:
            self.full_data_agreg[experiment] = pd.read_csv(f'{self.dir}{experiment}_full_data.csv')
            pprint.pprint( self.full_data_agreg)

        best_stages = [28,33,12,21,30,22,30,27,15,26,35,34,29,20,19,33,32,18,15,20,14]

        for idx_measure, measure in enumerate(self.measures):

            for run in self.runs:
                font = {'font.size': 20}
                plt.rcParams.update(font)
                fig, ax = plt.subplots()

                if run not in [1, 2, 3, 6, 7, 8, 9, 10, 11, 13, 14]:
                    exps = ['foraging-TD']
                else:
                    exps = self.experiments

                for idx_experiment, experiment in enumerate(exps):

                    tmp = self.full_data_agreg[experiment]
                    data = tmp[(tmp['run'] == run) & (tmp['experiment'] == experiment)]

                    ax.plot(data['checkpoint'], data[f'{measure}'], label=experiment, alpha=0.7, c=self.clrs[idx_experiment], linewidth=5)

                    ax.set_ylim((self.measures_limits[idx_measure][0], self.measures_limits[idx_measure][1]))
                    plt.xlabel('Stage')
                    plt.ylabel(f'{measure}')
                    #plt.title(self.anal)
                    # ax.legend()

                    if measure.find('total_success') != -1:
                        plt.plot([1, 35], [7, 7], 'b--', linewidth=1.5, alpha=0.3)

                    plt.plot([best_stages[run-1], best_stages[run-1]], [0, 250], 'b--', linewidth=1.5, alpha=0.3)

                plt.grid()
                plt.savefig(f'{self.dir}{self.anal}_{run}_{measure}.png', bbox_inches='tight')
                plt.clf()

    def boxes(self):

        plt.clf()

        data = pd.read_csv(f'{self.dir}final_tests_sim2.txt', sep="\t", header=None)
        data.columns=["experimentrun", "type", "pos", "total_success", "total_hurt", "steps"]

        data[['experiment', 'run']] = data.experimentrun.str.split("-TD_", expand=True, )

        tests_combinations = [('h_foraging', 'foraging')]

        for run in self.runs:

            exp_run = data[data['run'] == str(run)]
            if run not in [1, 2, 3, 6, 7, 8, 9, 10, 11, 13, 14]:
                exp_run = exp_run[exp_run['experiment'] == 'foraging']

            exp_run = exp_run.sort_values(by=['experiment'])
            pprint.pprint(exp_run)

            for idx_measure, measure in enumerate(self.measures):

                sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
                sb.set_style("whitegrid")

                plot = sb.boxplot(x='experiment', y=measure, data=exp_run,
                                  palette=self.clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                                  meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})

                try:
                    if len(tests_combinations) > 0:
                        add_stat_annotation(plot, data=exp_run, x='experiment', y=measure,
                                            box_pairs=tests_combinations,
                                            comparisons_correction=None,
                                            test='Wilcoxon', text_format='star', fontsize='xx-large', loc='inside', verbose=1)
                except Exception as error:
                    print(error)

                if measure.find('total_success') != -1:
                    plot.axhline(7, ls='--', color="blue", alpha=0.7)

                plot.set_ylim((self.measures_limits[idx_measure][0], self.measures_limits[idx_measure][1]))
                #plt.title(self.anal)
                plot.get_figure().savefig(f'{self.dir}{self.anal}_{run}_{measure}_box.png', bbox_inches='tight')
                plt.clf()


analysis = { 'h_foraging-TD':  ["foraging-TD", "h_foraging-TD"] }

runs = range(1, 21+1)
for anal in analysis:
    cd = PlotData(anal, analysis[anal], runs=runs )
    cd.lines()
    cd.boxes()
