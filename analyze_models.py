import itertools
from itertools import repeat
from multiprocessing import Pool
from multiprocessing import cpu_count
from argparse import ArgumentParser
import pickle
import warnings
import os

import matplotlib.colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from calibration import get_unbiased_calibration_rmse
from metrics import accuracy, precision, recall, specificity, f1_score, selection_rate, auroc, \
    loess_calibration, bootstrap_roc_curve, bootstrap_prg_curve, avg_underrepr
from model_fitting import load_results, load_data, get_group_name
from utils import bootstrap_metric, ecdf
from prg import calc_auprg_from_data

warnings.filterwarnings('error', category=RuntimeWarning)

# These must all be of the form metric(target, pred), where pred are the predicted probabilities.
# - Except if the final boolean is True, when it must take what avg_underrepr likes.
metrics = {
    'DRMSCE': [get_unbiased_calibration_rmse, False],
    'AUROC': [auroc, False],  # purely discrimination
    'AUPRG': [calc_auprg_from_data, False],
#    'EUR': [lambda yhat_test, group_mask_test, y_all, group_mask_all:
#                                       avg_underrepr(yhat_test, group_mask_test, y_all, group_mask_all,
#                                                     return_datapoints=True), True]
    #'accuracy=(TP+TN)/(P+N)': accuracy,
    #'precision=TP/(TP+FP)': precision,
    #'recall=sensitivity=TPR=TP/P': recall,
    #'specificity=TNR=TN/N': specificity,
    #'FPR': lambda x, y: 1 - specificity(x, y),
    #'FNR': lambda x, y: 1 - recall(x, y),
    #'log loss': lambda x, y: log_loss(x, y, labels=[0, 1]),  # discrimination + calibration
    #'f1_score': f1_score
}

N_bootstrap = 200
ci_alpha = 0.95
num_loess_calibration_samples = 200
add_med_group = False
ci_plot_alpha = 0.3


def subsets(lst, k):
    return list(itertools.combinations(lst, k))


def get_group_filter(sens_var_data, group_sens_vals):
    group_filter = pd.Series(True, index=sens_var_data.index)
    for sens_var_idx, sens_var_val in enumerate(group_sens_vals):
        if not isinstance(sens_var_val, float) or not np.isnan(sens_var_val):
            group_filter = group_filter & (sens_var_data.iloc[:, sens_var_idx] == sens_var_val)
    return group_filter


def get_group_combinations(sens_var_data, group_name_fun=None, min_group_size=50, max_var_com=3):
    sens_vars = sens_var_data.columns

    sens_var_vals = []
    for idx, sens_var in enumerate(sens_vars):
        sens_var_vals.append(sens_var_data[sens_var].unique())

    combinations = []
    for ii in range(min(len(sens_vars), max_var_com)):
        combinations = combinations + subsets(range(len(sens_vars)), ii+1)

    groups_dict = {}
    for sens_var_combination in combinations:
        sens_var_val_combinations = itertools.product(*[sens_var_vals[idx] for idx in sens_var_combination])
        for sens_var_val_combination in sens_var_val_combinations:
            group_sens_vals = [sens_var_val_combination[sens_var_combination.index(sens_idx)]
                               if sens_idx in sens_var_combination else np.nan
                               for sens_idx in range(len(sens_vars))]
            group_filter = get_group_filter(sens_var_data, group_sens_vals)
            if group_filter.sum() >= min_group_size:
                if group_name_fun is not None:
                    group_name = group_name_fun(group_sens_vals)
                else:
                    group_name = ''
                    for sens_val_idx, sens_var_idx in enumerate(sens_var_combination):
                        if sens_val_idx > 0:
                            group_name = group_name + "_"
                        if sens_var_data.columns[sens_var_idx][2] == '_':
                            col_abbrv = sens_var_data.columns[sens_var_idx][3:6]
                        elif sens_var_data.columns[sens_var_idx][3] == '_':
                            col_abbrv = sens_var_data.columns[sens_var_idx][4:7]
                        else:
                            col_abbrv = sens_var_data.columns[sens_var_idx][0:3]
                        group_name = group_name + f'{col_abbrv}={sens_var_val_combination[sens_val_idx]}'

                groups_dict[group_name] = group_sens_vals

    # Add "all" group
    groups_dict["all"] = [np.nan for _ in range(len(sens_vars))]

    return groups_dict


def metric_plot(metric_name, metric_results_df, plot_groups, fig_title="", ax=None, add_counts=True, legend=True,
                cmap=None):

    if ax is None and add_counts:
        f, (a0, a1) = plt.subplots(2, 1, height_ratios=[3, 1])
    elif ax is None and not add_counts:
        f = plt.figure()
        a0 = plt.gca()
    else:
        assert not add_counts
        a0 = ax

    if cmap is None:
        cmap = sns.color_palette("Set1", as_cmap=True)

    msk = (metric_results_df.metric_name == metric_name) & metric_results_df.group.isin(plot_groups)
    sns.pointplot(data=metric_results_df[msk], y='med', x='group', palette=cmap,
                  hue='group', order=plot_groups, hue_order=plot_groups, errorbar=None, join=False, ax=a0, scale=0.75)

    if legend:
        a0.legend(loc="upper right", fontsize="small", title="")
    else:
        a0.legend([], [], frameon=False)

    # Unfortunately, seaborn (still) doesn't seem to support pre-computed errorbars. Hence, we add them manually.
    # Solution adapted from https://stackoverflow.com/a/46811162/2207840.
    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    for point_pair in a0.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)

    # for reasons I do not know anything about, the above code now yields various "masked" entries.
    # The following lines remove those and yield the desired numerical values.
    x_coords = [x_coord for x_coord in x_coords if not np.ma.is_masked(x_coord)]
    y_coords = [y_coord for y_coord in y_coords if not np.ma.is_masked(y_coord)]
    assert len(x_coords) == len(plot_groups)

    # Calculate the type of error to plot as the error bars
    # Make sure the order is the same as the points were looped over
    # yerr must be (2,N), first row lower errors, second row upper errors. All gt 0.
    msks = [(metric_results_df.group == group) & (metric_results_df.metric_name == metric_name) for group in plot_groups]
    errors = np.array([[metric_results_df.med[msk].item() -
                        metric_results_df.lower[msk].item() for msk, group in zip(msks, plot_groups)],
                       [metric_results_df.upper[msk].item() -
                        metric_results_df.med[msk].item() for msk, group in zip(msks, plot_groups)]])

    for idx, _ in enumerate(x_coords):
        with warnings.catch_warnings():
            # Very likely a NaN axis was encountered; this is to be occasionally expected and fine
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            a0.errorbar(x_coords[idx], y_coords[idx], yerr=np.atleast_2d(errors[:, idx]).T, fmt='none',
                        ecolor=cmap(idx) if isinstance(cmap, matplotlib.colors.ListedColormap) else cmap[idx])

    a0.set_ylabel(metric_name)
    # Turn off xtick labels
    a0.axes.xaxis.set_ticklabels([])
    a0.set(xlabel="Group", xticklabels=[], title=fig_title)
    a0.yaxis.grid(True, linestyle='--')

    if add_counts:
        msk = metric_results_df.metric_name == metric_name
        h = sns.barplot(data=metric_results_df[msk], x="group", y="count", order=plot_groups, ax=a1)
        h.invert_yaxis()
        h.set_yscale("log")
        a1.legend([], [], frameon=False)
        a1.set(xlabel=None, xticklabels=[], xticks=[])


def rel_diag(eval_data, groups=None, plot_groups=None, sens_var_data=None, fig_title='', group_color_dict=None,
             ax=None, add_risk_density=True, cmap=None):
    # Reference for bootstrapping CIs for loess-based calibration:
    # https://onlinelibrary.wiley.com/doi/10.1002/sim.6167

    if ax is None and add_risk_density:
        f, (a0, a1) = plt.subplots(2, 1, height_ratios=[3, 1])
    elif ax is not None:
        assert not add_risk_density
        a0 = ax

    if groups is None:
        assert plot_groups is None and sens_var_data is None
        y_true = eval_data["y"].to_numpy()
        y_pred_proba = eval_data["y_pred_proba"].to_numpy()
        xvals = np.linspace(0, 1, 100)
        _, calib_probs_samples = loess_calibration(y_true, y_pred_proba, n_samples=num_loess_calibration_samples,
                                                   xvals=xvals)
        a0.plot(xvals, np.median(calib_probs_samples, axis=1), label="Calibration Curve")
        a0.fill_between(xvals,
                        np.quantile(calib_probs_samples, (1 - ci_alpha) / 2, axis=1),
                        np.quantile(calib_probs_samples, 1 - (1 - ci_alpha) / 2, axis=1),
                        color="blue", alpha=ci_plot_alpha)

        if add_risk_density:
            data_df = pd.DataFrame({"y_pred_proba": y_pred_proba})
            h = sns.histplot(data=data_df, x="y_pred_proba", ax=a1, element="poly", stat="density", legend=False)
            h.invert_yaxis()

    else:
        assert sens_var_data is not None

        if plot_groups is None:
            plot_groups = groups.keys()

        if group_color_dict is None and cmap is None:
            cmap = sns.color_palette("colorblind", n_colors=len(plot_groups), as_cmap=True)
        elif group_color_dict is not None:
            cmap = [group_color_dict[group] for group in plot_groups]

        dfs = []
        for idx, group_name in enumerate(plot_groups):
            if not group_name == 'group_med':
                group_sens_vals = groups[group_name]
                group_filter = get_group_filter(sens_var_data, group_sens_vals)
                y_true = eval_data["y"].to_numpy()[group_filter]
                y_pred_proba = eval_data["y_pred_proba"].to_numpy()[group_filter]
                dfs.append(pd.DataFrame({"y_pred_proba": y_pred_proba, "group": group_name}))
                xvals = np.linspace(0, 1, 100)
                calib_probs, calib_probs_samples = loess_calibration(y_true, y_pred_proba,
                                                                     n_samples=num_loess_calibration_samples,
                                                                     xvals=xvals)

                a0.plot(xvals, np.median(calib_probs_samples, axis=1), color=cmap[idx], label=group_name)
                a0.fill_between(xvals,
                                np.quantile(calib_probs_samples, (1-ci_alpha)/2, axis=1),
                                np.quantile(calib_probs_samples, 1 - (1-ci_alpha)/2, axis=1),
                                color=cmap[idx], alpha=ci_plot_alpha)
        if add_risk_density:
            data_df = pd.concat(dfs, ignore_index=True)
            h = sns.histplot(data=data_df, x="y_pred_proba", hue="group", hue_order=plot_groups, ax=a1, element="poly",
                             stat="density", common_norm=False, legend=False, fill=False, color=cmap)
            h.invert_yaxis()
            plt.xlim([0, 1])

    a0.plot([0, 1], [0, 1], 'k--', label="ideal")
    leg = a0.legend()
    leg.remove()
    a0.set(ylabel="$P(Y \mid R, G)$", xlim=[0, 1], ylim=[0, 1], title=fig_title,
           xticks=[0, 0.2, 0.4, 0.6, 0.8, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    a0.grid(visible=True)

    if add_risk_density:
        a0.set(xlabel=None, xticklabels=[])
        a1.set(xlabel="Risk Score $R$", xticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        a1.grid(visible=True)
    else:
        a0.set(xlabel="Risk Score $R$")


def repr_diag(yhat_test, plot_groups, bootstrap_data, group_color_dict=None, ax=None, cmap=None,
              quantile_range=[0.5, 1]):

    if ax is None:
        f, ax = plt.subplots()

    if group_color_dict is None and cmap is None:
        cmap = sns.color_palette("colorblind", n_colors=len(plot_groups), as_cmap=True)
    elif group_color_dict is not None:
        cmap = [group_color_dict[group] for group in plot_groups]

    target_thresh_cdfvals = np.linspace(quantile_range[0], quantile_range[1], 100)
    ecdf_fun = ecdf(yhat_test)

    for idx, group_name in enumerate(plot_groups):

        threshs, rel_repr = bootstrap_data['EUR'][group_name]
        rel_repr_resampled = np.zeros((N_bootstrap, len(target_thresh_cdfvals)))
        for bs_idx in range(N_bootstrap):
            threshs_cdfvals = ecdf_fun(threshs[bs_idx])
            rel_repr_resampled[bs_idx, :] = np.interp(target_thresh_cdfvals, threshs_cdfvals, rel_repr[bs_idx])  # linear interpolation

        ax.plot(target_thresh_cdfvals, np.median(rel_repr_resampled, axis=0), color=cmap[idx], label=group_name)
        ax.fill_between(target_thresh_cdfvals,
                        np.quantile(rel_repr_resampled, (1 - ci_alpha) / 2, axis=0),
                        np.quantile(rel_repr_resampled, 1 - (1 - ci_alpha) / 2, axis=0),
                        color=cmap[idx], alpha=ci_plot_alpha)
    ax.plot(quantile_range, [1, 1], 'k--', label="ideal")
    leg = ax.legend()
    leg.remove()
    ax.set(xlabel=r"ECDF($R$, $\tau$)", ylabel=r"$P(G \mid \hat{Y}{=}1) / P(G \mid Y{=}1)$")
    ax.grid(visible=True)


def curve_diag(eval_data, bootstrap_curve_fun, groups=None, plot_groups=None, sens_var_data=None, group_color_dict=None,
               ax=None, cmap=None):

    if ax is None:
        fig, ax = plt.figure()

    if groups is None:
        assert plot_groups is None and sens_var_data is None
        y_true = eval_data["y"].to_numpy()
        y_pred_proba = eval_data["y_pred_proba"].to_numpy()

        x, y_bs = bootstrap_curve_fun(y_true, y_pred_proba, num_bootstraps=N_bootstrap)

        ax.plot(x, np.median(y_bs, axis=0), label="ROC curve")
        ax.fill_between(x,
                        np.quantile(y_bs, (1 - ci_alpha) / 2, axis=0),
                        np.quantile(y_bs, 1 - (1 - ci_alpha) / 2, axis=0),
                        color="blue", alpha=ci_plot_alpha)

    else:
        assert sens_var_data is not None

        if plot_groups is None:
            plot_groups = groups.keys()

        if group_color_dict is None and cmap is None:
            cmap = sns.color_palette("colorblind", n_colors=len(plot_groups), as_cmap=True)
        elif group_color_dict is not None:
            cmap = [group_color_dict[group] for group in plot_groups]

        for idx, group_name in enumerate(plot_groups):
            if not group_name == 'group_med':
                group_sens_vals = groups[group_name]
                group_filter = get_group_filter(sens_var_data, group_sens_vals)
                y_true = eval_data["y"].to_numpy()[group_filter]
                y_pred_proba = eval_data["y_pred_proba"].to_numpy()[group_filter]

                x, y_bs = bootstrap_curve_fun(y_true, y_pred_proba, num_bootstraps=N_bootstrap)

                ax.plot(x, np.median(y_bs, axis=0), color=cmap[idx], label=group_name)
                ax.fill_between(x,
                                np.quantile(y_bs, (1 - ci_alpha) / 2, axis=0),
                                np.quantile(y_bs, 1 - (1 - ci_alpha) / 2, axis=0),
                                color=cmap[idx], alpha=ci_plot_alpha)

    return ax


def roc_diag(eval_data, groups=None, plot_groups=None, sens_var_data=None, fig_title='', group_color_dict=None,
             ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    ax = curve_diag(eval_data, bootstrap_roc_curve, groups=groups, plot_groups=plot_groups, sens_var_data=sens_var_data,
                    group_color_dict=group_color_dict, ax=ax)
    plt.sca(ax)
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    leg = ax.legend()
    leg.remove()
    plt.xlabel("FPR")  # = 1 - Specificity
    plt.ylabel("TPR")  # = Sensitivity = Recall
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(fig_title)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.grid(visible=True)
    return ax


def prg_diag(eval_data, groups=None, plot_groups=None, sens_var_data=None, fig_title='', group_color_dict=None,
             ax=None):

    if ax is None:
        fig, ax = plt.figure()

    ax = curve_diag(eval_data, bootstrap_prg_curve, groups=groups, plot_groups=plot_groups, sens_var_data=sens_var_data,
                    group_color_dict=group_color_dict, ax=ax)

    plt.sca(ax)
    plt.plot([0, 1], [1, 0], 'k--', label="$F_1$ baseline")
    leg = ax.legend()
    leg.remove()
    plt.xlabel("Recall Gain")
    plt.ylabel("Precision Gain")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(fig_title)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.grid(visible=True)
    return ax


def get_extreme_groups(metric_results, metric_name='all', top=2, add_default_groups=True):
    # Get top-k and bottom-k groups based on median value in a metric.
    # If metric_name == 'all' (default), do this for all groups in metric_results and return a single list with all.

    groups = []
    if metric_name == 'all':
        unique_metric_names = metric_results.metric_name.unique()
        for metric in unique_metric_names:
            groups = groups + get_extreme_groups(metric_results, metric, top, add_default_groups=False)

        # drop duplicates (result is unordered!)
        groups = list(set(groups))

        if add_default_groups:
            if add_med_group:
                target_group_count = 2 * top * len(unique_metric_names) + 2
            else:
                target_group_count = 2 * top * len(unique_metric_names) + 1
        else:
            target_group_count = 2 * top * len(unique_metric_names)

        if target_group_count < len(metric_results.group.unique()):
            slack = 1
            while len(groups) < target_group_count:
                # we now have fewer groups than we wanted, supplant with next-best ones
                for metric in unique_metric_names:
                    potential_groups = get_extreme_groups(metric_results, metric, top+slack, add_default_groups=False)
                    potential_groups = [grp for grp in potential_groups if grp not in groups]
                    while (len(groups) < target_group_count) and len(potential_groups) > 0:
                        groups.append(potential_groups[0])
                        del potential_groups[0]
                    if len(groups) == target_group_count:
                        break
                slack = slack + 1

        # sort by performance in first metric
        first_metric = metric_results.metric_name.unique()[0]
        metric_results_reduced = metric_results[metric_results.metric_name == first_metric]
        metric_results_reduced_sorted = metric_results_reduced.sort_values(by=["med", "upper", "lower"],
                                                                           ascending=False)
        groups = [group for group in metric_results_reduced_sorted.group if group in groups]

    else:
        metric_results_reduced = metric_results[metric_results.metric_name == metric_name]
        metric_results_reduced_sorted = metric_results_reduced.sort_values(by=["med", "upper", "lower"],
                                                                           ascending=False)
        shift = 0
        for idx in range(top):
            while metric_results_reduced_sorted.group.iloc[idx + shift] in ['all', 'group_med']:
                shift = shift + 1
            groups.append(metric_results_reduced_sorted.group.iloc[idx + shift])
        shift = 0
        for idx in range(top):
            while metric_results_reduced_sorted.group.iloc[-1 - idx - shift] in ['all', 'group_med']:
                shift = shift + 1
            groups.append(metric_results_reduced_sorted.group.iloc[-1 - idx - shift])

    if add_default_groups:
        if add_med_group:
            groups = groups + ['all', 'group_med']
        else:
            groups = groups + ['all']

    return groups


def plot_metric_overview(metric_results_df, groups, sens_var_data, yhat_test=None, bs_data=None):

    fontsize = 8
    params = {
        'axes.labelsize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize-1,
        'ytick.labelsize': fontsize,
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'xtick.major.pad': 0,
    }
    plt.rcParams.update(params)

    fig, axes = plt.subplots(len(metric_results_df.metric_name.unique())+1, 2, dpi=600)

    # textwidth in inches: 5.95114in text height in inches: 7.75024in -- but we also need space for the caption
    fig.set_size_inches(5.95114, 6.7)

    plot_groups = get_extreme_groups(metric_results_df, top=1)

    if len(plot_groups) > 12:
        import colorcet as cc
        # Ugly?
        cmap = sns.color_palette(cc.glasbey_category10, as_cmap=True)
    elif 11 <= len(plot_groups) <= 12:
        # creates problems with shading, because too similar (paired...) colors
        cmap = sns.color_palette('Paired', as_cmap=True)
    elif len(plot_groups) <= 10:
        import palettable
        cmap = palettable.tableau.Tableau_10.mpl_colors

    group_colors_dict = {group_name: cmap(idx) if isinstance(cmap, matplotlib.colors.ListedColormap) else cmap[idx]
                         for idx, group_name in enumerate(plot_groups)}

    for idx, metric_name in enumerate(metric_results_df.metric_name.unique()):
        plt.sca(axes[idx, 0])
        metric_plot(metric_name, metric_results_df, plot_groups, ax=axes[idx, 0], add_counts=False, legend=False,
                    cmap=cmap)

        if metric_name == 'DRMSCE':
            plot_groups_rel = get_extreme_groups(metric_results_df, 'DRMSCE', top=1)
            rel_diag(eval_data, groups, plot_groups_rel, sens_var_data, group_color_dict=group_colors_dict,
                     ax=axes[idx, 1], add_risk_density=False)
        elif metric_name == 'AUROC':
            plot_groups_roc = get_extreme_groups(metric_results_df, 'AUROC', top=1)
            roc_diag(eval_data, groups, plot_groups_roc, sens_var_data, group_color_dict=group_colors_dict,
                     ax=axes[idx, 1])
        elif metric_name == 'AUPRG':
            plot_groups_prg = get_extreme_groups(metric_results_df, 'AUPRG', top=1)
            prg_diag(eval_data, groups, plot_groups_prg, sens_var_data, group_color_dict=group_colors_dict,
                     ax=axes[idx, 1])
        elif metric_name == 'EUR':
            assert bs_data is not None and yhat_test is not None
            plot_groups_repr = get_extreme_groups(metric_results_df, 'EUR', top=1, add_default_groups=False)
            repr_diag(yhat_test, plot_groups_repr, bs_data, group_color_dict=group_colors_dict, ax=axes[idx, 1])

    # Add count plot in bottom row, left [choice of metric doesn't matter here since counts are the same
    msk = metric_results_df.metric_name == metric_results_df.metric_name.unique()[0]
    h = sns.barplot(metric_results_df[msk], x="group", y="count", order=plot_groups, ax=axes[-1, 0], palette=cmap)
    h.set_yscale("log")
    plt.sca(axes[-1, 0])
    xlabels = axes[-1, 0].get_xticklabels()
    axes[-1, 0].set_xticklabels(xlabels, rotation=45, ha='right', rotation_mode='anchor')
    plt.ylabel("Sample count")
    plt.xlabel("")
    plt.xticks(fontsize=4)

    # Add risk distribution plot in bottom row, right
    dfs = []
    for idx, group_name in enumerate(plot_groups):
        if not group_name == 'group_med':
            group_sens_vals = groups[group_name]
            group_filter = get_group_filter(sens_var_data, group_sens_vals)
            y_pred_proba = eval_data["y_pred_proba"].to_numpy()[group_filter]
            dfs.append(pd.DataFrame({"y_pred_proba": y_pred_proba, "group": group_name}))
    data_df = pd.concat(dfs, ignore_index=True)
    h = sns.kdeplot(data=data_df, x="y_pred_proba", hue="group", hue_order=plot_groups, ax=axes[-1, 1], #element="poly",
                    common_norm=False, clip=[0, 1], cut=0, palette=cmap, legend=False)
    plt.sca(axes[-1, 1])
    plt.xlim([0, 1])
    plt.xlabel("Risk score $R$")
    plt.ylabel("Density $p(R \mid G)$")
    axes[-1, 1].yaxis.grid(True, linestyle='--')

    plt.tight_layout()

    return fig


def get_group_results(metric_fun, metric_name, group_sens_vals, group_name, sens_var_data, eval_data):
    group_filter_test = get_group_filter(sens_var_data, group_sens_vals)
    y_true = eval_data["y"].to_numpy()[group_filter_test]
    y_pred_proba = eval_data["y_pred_proba"].to_numpy()[group_filter_test]
    metric_val = metric_fun(y_true, y_pred_proba)
    metric_bs = bootstrap_metric(metric_fun, [y_true, y_pred_proba], N_bootstrap)

    if metric_name[-5:] == '(pos)':
        count = len(y_true[y_true == 1])
    elif metric_name[-5:] == '(neg)':
        count = len(y_true[y_true == 0])
    else:
        count = len(y_true)

    return pd.DataFrame([{'group': group_name,
                                             'metric_name': metric_name,
                                             'metric': metric_val,
                                             'lower': np.nanquantile(metric_bs, (1 - ci_alpha) / 2),
                                             'med': np.nanmedian(metric_bs),
                                             'upper': np.nanquantile(metric_bs,
                                                                     ci_alpha + (1 - ci_alpha) / 2),
                                             'count': count}])


def analyze_model(eval_data, sens_var_data, groups, y_all=None, sens_var_data_all=None):
    metric_results_dfs = []
    if y_all is not None:
        y_all = y_all.to_numpy()

    # For some metrics we will want to store bootstrapped data for later reuse
    bs_data = {}
    for metric_name, [metric_fun, is_underrepr] in metrics.items():

        print(f"Processing metric {metric_name}...")

        if is_underrepr:
            bs_data[metric_name] = {}
            metric_results_lst = []
            for group_name, group_sens_vals in tqdm(groups.items(), total=len(groups.keys())):
                group_mask_all = get_group_filter(sens_var_data_all, group_sens_vals)
                group_filter_test = get_group_filter(sens_var_data, group_sens_vals).to_numpy()
                y_pred_proba_test = eval_data["y_pred_proba"].to_numpy()
                metric_val, _, _ = metric_fun(y_pred_proba_test, group_filter_test, y_all, group_mask_all)
                # metric_bs will be a list of length N_bootstrap, with a tuple (metric, threshs, rel_repr) for each
                # bootstrapped sample.
                metric_bs = bootstrap_metric(metric_fun, [y_pred_proba_test, group_filter_test], N_bootstrap,
                                             constant_params=[y_all, group_mask_all], num_returns=3)
                # unpack into three lists
                metric_bs, threshs, rel_repr = map(list, zip(*metric_bs))
                # turn the first into a numpy array
                metric_bs = np.array(metric_bs)
                # save the other two for later usage
                bs_data[metric_name][group_name] = (threshs, rel_repr)

                count = np.nan

                metric_results_lst.append(pd.DataFrame([{'group': group_name,
                                             'metric_name': metric_name,
                                             'metric': metric_val,
                                             'lower': np.nanquantile(metric_bs, (1 - ci_alpha) / 2),
                                             'med': np.nanmedian(metric_bs),
                                             'upper': np.nanquantile(metric_bs,
                                                                     ci_alpha + (1 - ci_alpha) / 2),
                                             'count': count}]))
        else:

            pool = Pool(cpu_count())

            # metric_fun, metric_name, group_sens_vals, group_name, sens_var_data, eval_data
            inputs = zip(repeat(metric_fun), repeat(metric_name), groups.values(),
                         groups.keys(), repeat(sens_var_data), repeat(eval_data))

            metric_results_lst = pool.starmap(get_group_results, tqdm(inputs, total=len(groups.keys())))

        metric_results_df = pd.concat(metric_results_lst, ignore_index=True)

        assert ((metric_results_df.upper >= metric_results_df.med) &
                    (metric_results_df.med >= metric_results_df.lower)).all()

        if add_med_group:
            # Add group median
            group_med_val = metric_results_df.med.median()
            metric_results_df = pd.concat([metric_results_df,
                                           pd.DataFrame([{'group': 'group_med', 'metric_name': metric_name,
                                                          'metric': np.nan, 'lower': np.nan,
                                                          'med': group_med_val, 'upper': np.nan,
                                                          'count': len(metric_results_df.group)}])])

        metric_results_dfs.append(metric_results_df)

    metric_results_all = pd.concat(metric_results_dfs, ignore_index=True)

    return metric_results_all, bs_data


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", default="catalan",
                        help="Dataset to use/fit", type=str)
    parser.add_argument("-r", "--reload", dest="reload", default=False,
                        help="Reload analysis results", type=bool)
    args = parser.parse_args()

    if args.dataset == "catalan":
        min_group_size = 100
    else:
        raise NotImplementedError

    for clf_name in ['LR', 'xgboost']:

        X_train, X_val, X_test, y_train, y_val, y_test, sens_train, sens_val, sens_test = load_data(args.dataset)
        y_all = pd.concat([y_train, y_val, y_test], ignore_index=True)
        sens_var_data_all = pd.concat([sens_train, sens_val, sens_test], ignore_index=True)

        eval_data, sens_var_data = load_results(args.dataset, clf_name)

        groups_to_analyze = \
            get_group_combinations(sens_var_data,
                                   group_name_fun=lambda group_sens_vals: get_group_name(args.dataset, group_sens_vals),
                                   min_group_size=min_group_size)
        if args.reload:
            metric_results = pd.read_parquet(f'out/{args.dataset}/metric_results_{clf_name}.pqt')
            with open(f'out/{args.dataset}/bs_data_{clf_name}.pickle', 'rb') as f:
                bs_data = pickle.load(f)
        else:
            metric_results, bs_data = analyze_model(eval_data, sens_var_data, groups_to_analyze, y_all=y_all,
                                                    sens_var_data_all=sens_var_data_all)
            metric_results.to_parquet(f'out/{args.dataset}/metric_results_{clf_name}.pqt')
            with open(f'out/{args.dataset}/bs_data_{clf_name}.pickle', 'wb') as f:
                pickle.dump(bs_data, f, pickle.HIGHEST_PROTOCOL)

        fig = plot_metric_overview(metric_results, groups_to_analyze, sens_var_data,
                                   eval_data["y_pred_proba"].to_numpy(), bs_data)

        if not os.path.exists(f'figs'):
            os.makedirs(f'figs')

        fig.savefig(f"figs/metrics_{clf_name}.pdf", bbox_inches="tight", pad_inches=0)

    plt.show()
