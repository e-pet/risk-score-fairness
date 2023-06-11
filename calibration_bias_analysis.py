from collections import OrderedDict
from itertools import product
from multiprocessing import Pool
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import random
import colorcet as cc

from calibration import get_unbiased_calibration_rmse
from metrics import ece

calib_test_curves = [lambda x: x, lambda x: x ** 2, lambda x: x ** 3]


calibration_metrics = OrderedDict({
                       'ECE': lambda y, yhat: ece(y, yhat, method='static', num_bins=15),
                       'ECE (BCS)': lambda y, yhat: ece(y, yhat, method='static', num_bins='sweep'),
                       'ACE': lambda y, yhat: ece(y, yhat, method='adaptive', num_bins=15),
                       'ACE (BCS)': lambda y, yhat: ece(y, yhat, method='adaptive', num_bins='sweep'),
                       'Kumar': lambda y, yhat: get_unbiased_calibration_rmse(y, yhat, num_bins=15),
                       'Kumar (BCS)': lambda y, yhat: get_unbiased_calibration_rmse(y, yhat, num_bins='sweep'),
                       })


N_repetitions = 100
N_bootstrap = 500
sample_sizes = [100, 1000, 10000]
example_descriptions = [r'$\rho=R$ (perfect calibration)', r'$\rho=R^2$ (poor calibration)',
                        r'$\rho=R^3$ (very poor calibration)']


def sample_example(N, example):
    y_pred = random.uniform(0.0, 1.0, (N,))
    risk = calib_test_curves[example](y_pred)
    y = random.binomial(n=1, p=risk)
    return y, y_pred


def run_example(repetition_idx, example, sample_size_idx):
    y, y_pred = sample_example(sample_sizes[sample_size_idx], example)
    results_lst = []
    for calibration_metric_index, (calibration_metric_name, calibration_metric_fun) in \
            enumerate(calibration_metrics.items()):

        metric_value = calibration_metric_fun(y, y_pred)
        results_lst.append(pd.DataFrame([{'metric': calibration_metric_name,
                                          'sample_size': sample_sizes[sample_size_idx],
                                          'value': metric_value,
                                          'example': example}]))

    results_df = pd.concat(results_lst, ignore_index=True)
    return results_df


if __name__ == '__main__':

    reload_results = False

    if not reload_results:

        results_lst = []
        bootstrap_results_lst = []
        pool = Pool(cpu_count())

        # rslts = starmap(run_example, product(range(N_repetitions), range(3), range(len(sample_sizes))))
        rslts = pool.starmap(run_example, product(range(N_repetitions), range(3), range(len(sample_sizes))))

        results = pd.concat(rslts, ignore_index=True)
        results.to_csv("calib_sample_bias_results.csv")
    else:
        results = pd.read_csv("calib_sample_bias_results.csv", index_col=0)

    # metric sample size bias analysis plots
    gt_eces = []
    gt_rmses = []
    for example in range(3):
        plt.figure()
        pred_test = np.arange(0.0, 1.0, 0.001)
        plt.plot(pred_test, calib_test_curves[example](pred_test))
        plt.title(f'True calibration curve {example} ({example_descriptions[example]})+ N=100 sample')
        plt.xlabel('Model confidence')
        plt.ylabel('Fraction of positives')
        plt.plot([0.0, 1.0], [0.0, 1.0], '--')
        gt_eces.append(np.mean(np.abs(pred_test - calib_test_curves[example](pred_test))))
        gt_rmses.append(np.sqrt(np.mean((pred_test - calib_test_curves[example](pred_test)) ** 2)))
        plt.text(0.05, 0.95, f'True ECE: {gt_eces[example]:.3f}, true RMSE: {gt_rmses[example]}')
        y, y_pred = sample_example(100, example)
        plt.scatter(y_pred, y)

    fontsize = 8
    params = {
        'axes.labelsize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}'
    }
    plt.rcParams.update(params)
    cmap = sns.color_palette(cc.glasbey_category10, as_cmap=True)

    # define outlier properties
    flierprops = dict(marker='o', s=1, color="black")
    rect_props = dict(edgecolor="black")
    med_props = dict(color="black")
    g = sns.catplot(results[results.example.isin([0, 1])], x="sample_size", kind="boxen", col="example", hue="metric",
                    hue_order=["ECE", "ECE (BCS)", "ACE", "ACE (BCS)", "Kumar", "Kumar (BCS)"],
                    palette=cmap, y="value", sharey=False, linewidth=0.5, showfliers=True,
                    height=2.0, aspect=1.5, flier_kws=flierprops, line_kws=med_props, box_kws=rect_props)  # 1.9, 1.1 for all three
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.44, 0.0), ncol=6)
    g._legend.set_title("")

    for ii in range(2):
        ax = g.axes[0, ii]
        plt.sca(ax)
        plt.title(example_descriptions[ii])
        plt.xlabel("Test sample size $N_S$")
        if ii == 0:
            plt.ylabel("Calibration error metric value")
        plt.axhline(gt_rmses[ii], color="gray", zorder=0, linestyle=':', linewidth=1)
        plt.axhline(gt_eces[ii], color="gray", zorder=0, linestyle='-', linewidth=1)

    g.figure.savefig(f"figs/calibration_bias.pdf", dpi=600, bbox_inches='tight', pad_inches=0)

    plt.show()
