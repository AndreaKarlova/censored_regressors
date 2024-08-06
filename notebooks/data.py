# %%
import functools
import operator
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_base_path = Path(globals().get('__file__', './_dummy_.py')).parent


# %%
def slice_cats(x, cat_vars, selected):
    sliced_x = x[functools.reduce(operator.and_, (x[i] == v for i, v in zip(cat_vars.index, selected)))]
    return sliced_x.drop(cat_vars.index, axis=1)


def read_deepsurv_data(path):
    datasets = defaultdict(dict)
    with h5py.File(path, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]
    full_x = pd.concat([pd.DataFrame(datasets[k]['x']) for k in ['train', 'test']], axis=0).reset_index(drop=True)
    full_e = (
        pd.concat([pd.Series(datasets[k]['e']) for k in ['train', 'test']], axis=0).astype(bool).reset_index(drop=True)
    )
    full_y = (
        pd.concat([pd.Series(datasets[k]['t']) for k in ['train', 'test']], axis=0).reset_index(drop=True)
    )
    return full_x, full_e, full_y


# %%
def gbsg_cancer(include_categorical=True):
    """
        Features (6): [???], [???], [???], [???], [???], [???]
        Output: days from death
    """
    full_x, full_e, full_y = read_deepsurv_data(_base_path / "data/gbsg_cancer_train_test.h5")
    cat_vars = pd.Series({0: 2, 1: 3, 2: 2})
    is_censored = (~full_e).rename('is_censored')

    if not include_categorical:
        selected_categorical = [0, 0, 1, 0]
        full_x = slice_cats(full_x, cat_vars, selected_categorical)
        full_y = full_y.loc[full_x.index].reset_index(drop=True)
        is_censored = is_censored.loc[full_x.index].reset_index(drop=True)
        full_x = full_x.reset_index(drop=True)

    left_bound = pd.Series(-np.inf, index=full_y.index, dtype=float)
    right_bound = pd.Series(np.inf, index=full_y.index, dtype=float)
    right_bound[is_censored] = full_y[is_censored]

    return full_x, full_y, cat_vars, left_bound, right_bound, is_censored


# %%
def metabric_clinical(include_categorical=True):
    """
        Features (8): gene MK16, gene EGFR, gene PGR, gene ERRBB2, hormone treatment indicator, radiotherapy indicator, chemotherapy indicator, ER-positive indicator, age at diagnosis
        Output: months from death
    """
    full_x, full_e, full_y = read_deepsurv_data(_base_path / "data/metabric_IHC4_clinical_train_test.h5")
    cat_vars = pd.Series({4: 2, 5: 2, 6: 2, 7: 2})
    is_censored = (~full_e).rename('is_censored')

    if not include_categorical:
        selected_categorical = [1, 1, 0, 1]
        full_x = slice_cats(full_x, cat_vars, selected_categorical)
        full_y = full_y.loc[full_x.index].reset_index(drop=True)
        is_censored = is_censored.loc[full_x.index].reset_index(drop=True)
        full_x = full_x.reset_index(drop=True)

    left_bound = pd.Series(-np.inf, index=full_y.index, dtype=float)
    right_bound = pd.Series(np.inf, index=full_y.index, dtype=float)
    right_bound[is_censored] = full_y[is_censored]
    return full_x, full_y, cat_vars, left_bound, right_bound, is_censored


# %%
def support(include_categorical=True):
    """
        Features (14): age, sex, race, number of comorbidities, presence of diabetes, presence of dementia, presence of cancer, mean arterial blood pressure, heart rate, respiration rate, temperature, white blood cell count, serum’s sodium, serum’s creatinine
        Output: days from death
    """
    full_x, full_e, full_y = read_deepsurv_data(_base_path / 'data/support_train_test.h5')
    cat_vars = pd.Series({1: 2, 2: 9, 4: 2, 5: 2, 6: 3, })
    is_censored = (~full_e).rename('is_censored')

    if not include_categorical:
        selected_categorical = [0, 2, 0, 0, 1]  # 40.9% censored
        # selected_categorical = [0, 0, 0, 0, 1] # 51% censored
        full_x = slice_cats(full_x, cat_vars, selected_categorical)
        full_y = full_y.loc[full_x.index].reset_index(drop=True)
        is_censored = is_censored.loc[full_x.index].reset_index(drop=True)
        full_x = full_x.reset_index(drop=True)

    left_bound = pd.Series(-np.inf, index=full_y.index, dtype=float)
    right_bound = pd.Series(np.inf, index=full_y.index, dtype=float)
    right_bound[is_censored] = full_y[is_censored]
    return full_x, full_y, cat_vars, left_bound, right_bound, is_censored


# %%
def whas_heartattack(include_categorical=True):
    """
        Features (6): [???], age, sex, body-mass-index (BMI), left heart failure complications (CHF), and order of MI (MIORD).
        Output: days from death
    """
    full_x, full_e, full_y = read_deepsurv_data(_base_path / "data/whas_train_test.h5")
    cat_vars = pd.Series({0: 2, 2: 2, 4: 2, 5: 2})
    is_censored = (~full_e).rename('is_censored')

    if not include_categorical:
        selected_categorical = [0, 0, 1, 0]
        full_x = slice_cats(full_x, cat_vars, selected_categorical)
        full_y = full_y.loc[full_x.index].reset_index(drop=True)
        is_censored = is_censored.loc[full_x.index].reset_index(drop=True)
        full_x = full_x.reset_index(drop=True)

    left_bound = pd.Series(-np.inf, index=full_y.index, dtype=float)
    right_bound = pd.Series(np.inf, index=full_y.index, dtype=float)
    right_bound[is_censored] = full_y[is_censored]
    return full_x, full_y, cat_vars, left_bound, right_bound, is_censored
