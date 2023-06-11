import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.interpolate import interp1d


def train_val_test_split(X, y, sizes, random_state=None, stratify=None):
    assert(sum(sizes) == 1)
    assert(len(sizes) == 3)
    if random_state is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(random_state)

    test_and_vali_size = sizes[1]+sizes[2]
    rel_test_size = sizes[2] / test_and_vali_size
    if stratify is not None:
        if isinstance(stratify, pd.DataFrame) and len(stratify.columns) > 1:
            # We want to stratify by multiple columns, sklearn train_test_split does not support that.
            # Thus, set up an artificial column for that.
            # (Inspired by https://stackoverflow.com/a/51525992/2207840.)
            strat = stratify.iloc[:, 0].astype(str)
            for col_idx in range(1, len(stratify.columns)):
                strat = strat + '_' + stratify.iloc[:, col_idx].astype(str)
            assert(len(np.unique(strat)) == np.prod(stratify.nunique(axis='rows')))
        else:
            strat = stratify

        X_train, X_vt, y_train, y_vt, strat_train, strat_vt = train_test_split(X, y, strat,
                                                                               test_size=test_and_vali_size,
                                                                               random_state=rng.integers(4294967295),
                                                                               stratify=strat)
        X_val, X_test, y_val, y_test = train_test_split(X_vt, y_vt, test_size=rel_test_size,
                                                        random_state=rng.integers(4294967295),
                                                        stratify=strat_vt)
    else:
        X_train, X_vt, y_train, y_vt = train_test_split(X, y, test_size=test_and_vali_size,
                                                        random_state=rng.integers(4294967295))
        X_val, X_test, y_val, y_test = train_test_split(X_vt, y_vt, test_size=rel_test_size,
                                                        random_state=rng.integers(4294967295))

    return X_train, X_val, X_test, y_train, y_val, y_test


def bootstrap_metric(metric_fun, data_to_bootstrap, N_bootstrap, constant_params=None, num_returns=1):
    # 'data' should be N_predictions x n_args numpy array, where n_args is the number of arguments expected by
    # metric_fun.
    # Bootstrap samples of the individual columns will then be passed to metric_fun in the order they appear in the
    # array.
    N_predictions = len(data_to_bootstrap[0])
    if num_returns == 1:
        metric_bs = np.zeros((N_bootstrap,))
    else:
        metric_bs = []

    for idx in range(N_bootstrap):
        bs_idces = np.random.choice(range(N_predictions), N_predictions)
        try:
            if constant_params is None:
                args = [dat[bs_idces] for dat in data_to_bootstrap]
            else:
                assert isinstance(constant_params, list)
                args = [dat[bs_idces] for dat in data_to_bootstrap] + constant_params
            if num_returns == 1:
                metric_bs[idx] = metric_fun(*args)
            else:
                metric_bs.append(metric_fun(*args))
        except ValueError:
            if num_returns == 1:
                metric_bs[idx] = np.nan
            else:
                metric_bs.append(())

    assert len(metric_bs) == N_bootstrap
    return metric_bs


def ecdf(x):
    # yields identical results as statsmodels.distributions.empirical_distribution.ECDF
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    # kind='previous' would give the staircase-version
    return interp1d(xs, ys, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
