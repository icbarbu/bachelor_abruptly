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
        self.measures_limits = [[100, 210],  [-0.5, 8]]
        self.measures = ['steps', 'total_success']
        self.clrs = ['#FF3333', '#006600']

        plt.style.use('classic')

    def compare(self):

        plt.clf()

        data = pd.read_csv(f'{self.dir}final_tests_sim2.txt', sep="\t", header=None)
        data.columns=["experimentrun", "type", "pos", "total_success", "total_hurt", "steps"]

        data[['experiment', 'run']] = data.experimentrun.str.split("-TD_", expand=True, )

        data = data[data.run.isin(['1', '2', '3', '6', '7', '8', '9', '10', '11', '13', '14'])]
        pprint.pprint(data)

        data_ag = data.groupby(['experiment', 'run']).agg(
            steps=('steps', 'mean'),
            total_success=('total_success', 'mean')
        ).reset_index()
        pprint.pprint(data_ag)

        data_ag2 = data_ag.groupby(['experiment']).agg(
            steps=('steps', 'mean'),
            steps_std=('steps', 'std'),
            total_success=('total_success', 'mean'),
            total_success_std=('total_success', 'std')
        ).reset_index()
        pprint.pprint(data_ag2)

        for idx_measure, measure in enumerate(self.measures):

            sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
            sb.set_style("whitegrid")

            plot = sb.boxplot(x='experiment', y=measure, data=data_ag,
                              palette=self.clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                              meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})

            try:

                add_stat_annotation(plot, data=data, x='experiment', y=measure,
                                    box_pairs=[('foraging', 'h_foraging')],
                                    comparisons_correction=None,
                                    test='Wilcoxon', text_format='star', fontsize='xx-large', loc='inside',
                                    verbose=1)
            except Exception as error:
                print(error)

            if measure.find('total_success') != -1:
                plot.axhline(7, ls='--', color="blue", alpha=0.7)

            plot.set_ylim((self.measures_limits[idx_measure][0], self.measures_limits[idx_measure][1]))
            # plt.title(self.anal)
            plot.get_figure().savefig(f'{self.dir}{measure}_gain.png', bbox_inches='tight')
            plt.clf()


analysis = { 'h_foraging-TD':  ["foraging-TD", "h_foraging-TD"] }

runs = range(1, 21+1)
for anal in analysis:
    cd = PlotData(anal, analysis[anal], runs=runs )
    cd.compare()









