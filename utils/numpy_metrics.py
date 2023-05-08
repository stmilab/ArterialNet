from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
import numpy as np

# define loss functions
def mean_accuracy(logits: list, y: list) -> float:
    return accuracy_score(logits, y)


def calc_f1_score(logits: list, y: list) -> float:
    """
    calc_f1_score compute F1_Score between 2 torch tensors

    Args:
        logits (list): predicted tensor
        y (list): ground truth tensor

    Returns:
        list: computed F1 Score in torch tensor format
    """
    return f1_score(logits, y)


def calc_RMSE(logits: list, y: list) -> float:
    """
    calc_RMSE compute RMSE between 2 torch tensors

    Args:
        logits (list): predicted tensor
        y (list): ground truth tensor

    Returns:
        list: computed RMSE in torch tensor format
    """
    return np.sqrt(mean_squared_error(logits, y))


def calc_Pearson(logits: list, y: list) -> float:
    """
    calc_Pearson compute Pearson's Correlation between 2 torch tensors

    Args:
        logits (list): predicted tensor
        y (list): ground truth tensor

    Returns:
        list: computed Pearson's R in torch tensor format
    """
    r_val, p_val = pearsonr(logits.reshape(-1), y.reshape(-1))
    return r_val


def calc_metrics(pred, test):
    """calc_metrics
        Calculating the needed metrics based on the provided prediction and ground truth

    Arguments:
        pred {list} -- predicted array
        test {list} -- ground truth array

    Returns:
        list of floats -- RMSE, MAE, Pearon's R and P value
    """
    rmse = np.sqrt(mean_squared_error(test, pred))
    mae = mean_absolute_error(test, pred)
    mean = np.mean(pred)
    std = np.std(pred)
    rval, pval = pearsonr(test, pred)
    return rmse, mae, mean, std, rval, pval
