import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from statannot import add_stat_annotation
from itertools import combinations
import sys

pd.set_option("display.max_rows", 99999)


dir = 'experiments/anal/'

runs = {12: 'run 12 without human help', 7: 'run 7 with human help'}

data_sim = pd.read_csv(f'{dir}final_tests_sim.txt', sep="\t", header=None)
data_sim.columns=["experimentrun", "type", "pos", "total_success", "total_hurt", "steps"]
data_sim = data_sim[(data_sim['experimentrun'] == 'foraging-TD_12') | (data_sim['experimentrun'] == 'h_foraging-TD_7')]
data_sim["experimentrun"] = data_sim.experimentrun.str.split("-TD_", expand=True, )[1]

data_hard = pd.read_csv(f'{dir}final_tests_hard.txt', sep=" ", header=None)
data_hard.columns=["experimentrun","pos", "total_success"]
data_hard["type"] = 'hard'

data_sim = data_sim.drop(columns=["pos", "total_hurt", "steps"])
data_hard = data_hard.drop(columns=[ "pos"])

data = pd.concat([data_sim, data_hard])
data[['experimentrun']] = data[['experimentrun']].astype(float)

pprint.pprint(data)

clrs = ['#000066', '#006600']


for run in runs:
    print('run----', run)
    data_pl = data[data['experimentrun'] == run]

    pprint.pprint(data_pl)

    full_data_agreg = data_pl.groupby(['experimentrun', 'type']) \
        .agg(
        total_success_mean=('total_success', 'mean')
    ).reset_index()
    pprint.pprint(full_data_agreg)

    sb.set(rc={"axes.titlesize": 23, "axes.labelsize": 23, 'ytick.labelsize': 21, 'xtick.labelsize': 21})
    sb.set_style("whitegrid")

    plot = sb.boxplot(x='type', y="total_success", data=data_pl,
                      palette=clrs, width=0.4, showmeans=True, linewidth=2, fliersize=6,
                      meanprops={"marker": "o", "markerfacecolor": "yellow", "markersize": "12"})


    add_stat_annotation(plot, data=data_pl, x='type', y="total_success",
                        box_pairs=[('sim', 'hard')],
                        comparisons_correction=None,
                        test='Wilcoxon', text_format='star', fontsize='xx-large', loc='inside', verbose=1, line_height=0.4)


    plot.axhline(7, ls='--', color="blue", alpha=0.7)
    plot.set_ylim(-0.5, 8.5)
    plot.set(xlabel='environment', ylabel='total_success')

    plt.title(runs[run])
    plot.get_figure().savefig(f'{dir}{run}_hardware_box.png', bbox_inches='tight')
    plt.clf()
