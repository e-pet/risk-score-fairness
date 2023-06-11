import os
from joblib import dump
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, KBinsDiscretizer
from xgboost import XGBClassifier

from calibration import BetaCalibratedClassifier
from utils import train_val_test_split


catalan_sens_vars = ['V1_sex', 'age_group', 'V4_area_origin', 'V6_province']
catalan_group_name_map = {
    'V1_sex': ('S', {0: 'F', 1: 'M'}),
    'age_group': ('AG', {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}),
    'V4_area_origin': ('O', {'Spain': 'SPA', 'Maghreb': 'MAG', 'Latin America': 'LA', 'Europe': 'EUR', 'Other': 'OTH'}),
    'V6_province': ('P', {'Lleida': 'LLE', 'Barcelona': 'BAR', 'Girona': 'GIR', 'Tarragona': 'TAR'})
}


def load_catalan_data():
    file_path = 'catalan_data/catalan-juvenile-recidivism-subset.csv'
    id_var = 'id'
    y_var = 'V115_RECID2015_recid'

    dtypes = {'V1_sex': 'category',
              'V4_area_origin': 'category',  # where they are from (Spain, Maghreb, Latin America, Europe, Other)
              'V6_province': 'category',  # where they live?  (Lleida, Barcelona, Girona, Tarragona)
              'V8_age': 'int16',  # at time of crime!
              'V9_age_at_program_end': 'int16',
              'V12_n_criminal_record': pd.CategoricalDtype(["0", "1-2", "3-5", "5+"], ordered=True),
              'V15_main_crime_cat': 'category',  # Against people, against property, other
              'V19_committed_crime': 'category',  # 23 unique strings specifying the crime
              'V21_n_crime': 'int16',  # number of crimes in the current case
              'V23_territory_of_execution': 'category',  # province where the program was executed
              'V24_finished_program': 'category',  # What program has the juvenile finished
              'V26_finished_measure_grouped': 'category',  # Categorization of the program
              'V29_program_duration': 'int16',  # duration of assigned program in days
              }

    raw_data = pd.read_csv(file_path, dtype=dtypes)

    X = raw_data.drop([y_var, id_var], axis=1)
    y = raw_data[y_var]

    V12 = X.V12_n_criminal_record
    X.loc[:, 'V12_n_criminal_record'] = (V12 == '1-2').astype(int) + (V12 == '3-5').astype(int) + (V12 == '5+').astype(
        int)
    V13 = X.V13_n_crime_cat
    X.loc[:, 'V13_n_crime_cat'] = (V13 == '2').astype(int) + (V13 == '3+').astype(int)
    V27 = X.V27_program_duration_cat
    X.loc[:, 'V27_program_duration_cat'] = (V27 == '6 months < 1 year').astype(int) + (V27 == '>1 year').astype(int)

    cat_cols = list(X.select_dtypes(exclude=["number", "datetime", "bool"]).columns)
    num_cols = list(X.select_dtypes(include="number").columns)
    num_cols.remove('V8_age')  # will be transformed into ordinal age_group below
    x_ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoding_dicts = {}
    for col in cat_cols:
        if len(X[col].cat.codes.unique()) <= 2:
            # This variable is binary, do not one-hot-encode - simply use the 0/1 code.
            # I haven't exactly followed why, but this yields a dict like {0: 'female', 1: 'male'}.
            encoding_dicts[col] = dict(enumerate(X[col].cat.categories))
            X.loc[:, col] = X[col].cat.codes
            cat_cols.remove(col)

    assert (encoding_dicts['V1_sex'][0] == 'female') and (encoding_dicts['V1_sex'][1] == 'male')

    print('One-hot encoding the following columns: ' + str(cat_cols))
    x_col_trafo = x_ohe.fit_transform(X[cat_cols])
    X.loc[:, x_ohe.get_feature_names_out(cat_cols)] = x_col_trafo

    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = train_val_test_split(X, y, [0.7, 0.1, 0.2],
                                                                                      random_state=1, stratify=y)
    age_discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    X_train_raw.loc[:, 'age_group'] = age_discretizer.fit_transform(X_train_raw.V8_age.values.reshape(-1, 1))
    X_train = X_train_raw.drop(labels=cat_cols, axis=1)
    X_val_raw.loc[:, 'age_group'] = age_discretizer.transform(X_val_raw.V8_age.values.reshape(-1, 1))
    X_val = X_val_raw.drop(labels=cat_cols, axis=1)
    X_test_raw.loc[:, 'age_group'] = age_discretizer.transform(X_test_raw.V8_age.values.reshape(-1, 1))
    X_test = X_test_raw.drop(labels=cat_cols, axis=1)

    encoding_dicts['age_group'] = {idx: f'{age_discretizer.bin_edges_[0][idx]} <= age < {age_discretizer.bin_edges_[0][idx+1]}'
                                   for idx in range(5)}

    x_scaler = StandardScaler()
    x_num_trafo = x_scaler.fit_transform(X_train[num_cols])
    X_train.loc[:, num_cols] = x_num_trafo
    X_train.drop(labels='V8_age', inplace=True, axis=1)

    y_le = LabelEncoder()
    y_train = pd.Series(y_le.fit_transform(y_train), name=y_train.name, index=y_train.index)

    x_num_trafo = x_scaler.transform(X_val[num_cols])
    X_val.loc[:, num_cols] = x_num_trafo
    y_val = pd.Series(y_le.fit_transform(y_val), name=y_val.name, index=y_val.index)
    X_val.drop(labels='V8_age', inplace=True, axis=1)

    x_num_trafo = x_scaler.transform(X_test[num_cols])
    X_test.loc[:, num_cols] = x_num_trafo
    y_test = pd.Series(y_le.fit_transform(y_test), name=y_test.name, index=y_test.index)
    X_test.drop(labels='V8_age', inplace=True, axis=1)

    sens_train = X_train_raw[catalan_sens_vars]
    sens_val = X_val_raw[catalan_sens_vars]
    sens_test = X_test_raw[catalan_sens_vars]

    # export encoding dict
    with open('catalan_data/encoding.txt', 'w') as f:
        f.write(str(encoding_dicts))

    return X_train, X_val, X_test, y_train, y_val, y_test, sens_train, sens_val, sens_test


def load_data(dataset_name):
    if dataset_name == 'catalan':
        X_train, X_val, X_test, y_train, y_val, y_test, sens_train, sens_val, sens_test = \
            load_catalan_data()
    else:
        raise NotImplementedError

    print('Training data head:')
    print(X_train.head(n=2))
    print(y_train.head(n=2))
    print('Val data head:')
    print(X_val.head(n=2))
    print(y_val.head(n=2))
    print('Test data head:')
    print(X_test.head(n=2))
    print(y_test.head(n=2))

    return X_train, X_val, X_test, y_train, y_val, y_test, sens_train, sens_val, sens_test


def fit_models(X_train, X_val, y_train, y_val, dataset_name):

    xgboost_param_space = {
        'max_depth': [3, 4, 5, 6, 7, 10, 15, 20],
        'learning_rate': [1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01],
        'gamma': [0, 0.25, 1, 3, 5, 7],
        'reg_lambda': [0, 1, 10, 30, 50],
        'scale_pos_weight': [1, 3, 5],
        'subsample': [0.25, 0.5, 0.75, 1.0],
        'colsample_bytree': [0.4, 0.5, 0.6],
    }

    base_rate = sum(y_train) / len(y_train)
    print(f'Base rate in the training dataset is {base_rate}.')
    neg_pos_ratio = sum(y_train == 0) / sum(y_train == 1)
    print(f'Neg-to-pos-ratio in the training dataset is {neg_pos_ratio}:1.')

    clfs = {'xgboost': RandomizedSearchCV(
        XGBClassifier(objective='binary:logistic', use_label_encoder=False, scale_pos_weight=neg_pos_ratio,
                      eval_metric='logloss', random_state=1),
        xgboost_param_space, n_jobs=8, cv=5, n_iter=10, random_state=1, error_score='raise'),
            'LR': LogisticRegression(solver="liblinear", random_state=1)}

    for clf_name, clf in clfs.items():

        clf.fit(X_train, y_train)

        if len(y_val) > 1000:
            print("Using ísotonic regression for calibration")
            # Isotonic regression is only recommended for > 1000 samples, see
            # https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
            clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
            clf.fit(X_val, y_val)
        else:
            print("Using beta calibration")
            # Use beta calibration, cf. https://betacal.github.io/ and
            # Kull, Silva Filho, Flach (2017), https://doi.org/10.1214/17-EJS1338SI

            clf = BetaCalibratedClassifier(clf)
            clf.fit(X_val, y_val)

        print(clf)

        if not os.path.exists(f'out/{dataset_name}'):
            os.makedirs(f'out/{dataset_name}')

        dump(clf, f'out/{dataset_name}/{clf_name}_calibrated_sklearn.joblib')

    return clfs


def fit_and_store_model_and_results(dataset_name):
    X_train, X_val, X_test, y_train, y_val, y_test, sens_train, sens_val, sens_test = load_data(dataset_name)
    clfs = fit_models(X_train, X_val, y_train, y_val, dataset_name)

    # evaluate models on test set, store results
    for clf_name, clf in clfs.items():
        y_pred_proba = clf.predict_proba(X_test)
        if np.ndim(y_pred_proba) > 1:
            # this should be an N x 2 matrix with the probabilities of the two classes
            assert (sum(abs(y_pred_proba.sum(axis=1) - 1) < 1e-7) == len(y_pred_proba))
            # reduce it to just the likelihood of the "1" class, as usual
            y_pred_proba = y_pred_proba[:, 1]
        assert (y_pred_proba.min() >= 0 and y_pred_proba.max() <= 1)

        eval_data = sens_test.copy()
        eval_data['y'] = y_test
        eval_data['y_pred_proba'] = y_pred_proba
        eval_data.to_parquet(f'out/{dataset_name}/{clf_name}_results.pqt')


def load_results(dataset_name, clf_name):

    df = pd.read_parquet(f'out/{dataset_name}/{clf_name}_results.pqt')
    eval_data = df[['y', 'y_pred_proba']].astype(dtype={"y": "int16", "y_pred_proba": "float64"})
    sens_vars = df.columns.drop(["y_pred_proba", "y"])
    sens_var_data = df[sens_vars]

    return eval_data, sens_var_data


def get_group_name(dataset_name, sens_var_vals, use_latex=True):
    if dataset_name == 'catalan':
        group_name_dicts = catalan_group_name_map
        sens_vars = catalan_sens_vars
    else:
        raise NotImplementedError

    group_name = ''
    for sens_var_name, sens_val in zip(sens_vars, sens_var_vals):
        if not (isinstance(sens_val, float) and np.isnan(sens_val)):
            if not group_name == '':
                if use_latex:
                    group_name = group_name + " $\land$ "  # logical and symbol
                else:
                    group_name = group_name + " \u2227 "  # logical and symbol
            sens_var_name, sens_var_val_map = group_name_dicts[sens_var_name]
            if use_latex:
                # group_name = group_name + f'{sens_var_name}$\,=\,${sens_var_val_map[sens_val]}'
                group_name = group_name + sens_var_name + r'$^{\text{' + sens_var_val_map[sens_val] + '}}$'
            else:
                group_name = group_name + f'{sens_var_name}={sens_var_val_map[sens_val]}'

    assert len(group_name) > 0
    return group_name


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", dest="dataset", default="catalan",
                        help="Dataset to use/fit", type=str)

    args = parser.parse_args()

    fit_and_store_model_and_results(args.dataset)
