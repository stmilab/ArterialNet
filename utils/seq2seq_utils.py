# Import packages
from __future__ import print_function
import pandas as pd
import os, glob, pdb
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import butter, lfilter, freqz
import sys
import wfdb
from tabulate import tabulate
from typing import Tuple

import phase_alignment as pa

# Compatibility layer between Python 2 and Python 3
from scipy.signal import firwin, lfilter, resample
import math
from arg_parser import sicong_argparse
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# End of Imports


class Sicong_Norm:
    def __init__(self, arr=None, min_val=0, max_val=200):
        if arr is None:
            self.min = min_val
            self.max = max_val
        else:
            self.min = arr.min()
            self.max = arr.max()

    def normalize(self, arr):
        return (arr - self.min) / (self.max - self.min)

    def denormalize(self, arr):
        return arr * (self.max - self.min) + self.min


def get_maximum_slope_from_ppg(ppg_arr: np.ndarray, sec=0.5, freq=125) -> np.ndarray:
    """
    get_maximum_slope_from_ppg computes maximum slopes from the given PPG waveform

    It computes all of the maximum slopes and stores the index into a list

    Args:
        ppg_arr (np.ndarry): The input PPG signal
        sec (float, optional): interval of a minimum heart beat in seconds. Defaults to 0.5.
        freq (int, optional): The sampling frequency of the signal. Defaults to 125.

    Returns:
        np.ndarray: list of maximum slope index
    """
    # formulating a windowsize based on time
    win_size = int(freq * sec)
    olap = int(0.5 * win_size)
    ms_ind = []
    grad = np.gradient(ppg_arr)
    sec_grad = np.gradient(grad)
    third_grad = np.gradient(sec_grad)
    for win_st in np.arange(0, len(grad), olap):
        win_ed = min(len(grad), win_st + win_size)
        # 95th percentile for first gradient -> positive maximum slope candidate
        p95 = np.percentile(grad[win_st:win_ed], 95)
        # 05th percentile for thirf gradient -> stronger candidate
        tp05 = np.percentile(third_grad[win_st:win_ed], 5)
        for i in np.arange(win_st, win_ed - 1, 1):
            # matching criteria to find strong maximum slope
            if grad[i] > p95 and third_grad[i] < tp05:
                # avoiding duplication
                if len(ms_ind) == 0 or (i - ms_ind[-1]) > 0.25 * freq:
                    ms_ind.append(i)
    # filtering duplicated ms points since we had sliding window with overlaps
    return np.unique(ms_ind)


def downsample_arr(arr: np.ndarray, desired_len: int) -> np.ndarray:
    # calculate the downsampling factor
    factor = len(arr) / desired_len
    # downsample the array using array slicing
    return (arr[:: int(factor)])[-desired_len:]


def form_cardiac_cycles(cwdf: pd.DataFrame, flags: argparse.Namespace) -> list:
    """
    form_cardiac_cycles Formulate cardiac cycles by performing cardiac Segmentation on the given waveform

    Actions:
    1. Calculating the maxmimum points of the PPG waveform
    2. Segmenting arrays depending on the maximum points
    3. Outputting the dictionary of segmented arrays

    Args:
        cwdf (pd.DataFrame): dataframe containing MIMIC waveform
        flags (argparse.Namespace): flgas containing all the script parameters

    Returns:
        list: Output an array of dictionary containing cardiac cycles in np.ndarray
    """
    abp_np = cwdf.ABP.to_numpy()
    ppg_np = cwdf.PPG.to_numpy()
    ecg_np = cwdf.ECG.to_numpy()
    idx_np = cwdf.IDX.to_numpy()
    ms_pts = get_maximum_slope_from_ppg(ppg_np, sec=0.5)
    whole_list = np.arange(len(idx_np))
    batch_of_dict_arr = []
    feature_list = [
        whole_list[ms_pts[i - flags.num_prev_cardiac_cycle_feature] : ms_pts[i]]
        for i in range(flags.num_prev_cardiac_cycle_feature, len(ms_pts))
    ]
    label_list = [
        whole_list[ms_pts[i - flags.num_prev_cardiac_cycle_label] : ms_pts[i]]
        for i in range(flags.num_prev_cardiac_cycle_feature, len(ms_pts))
    ]
    for i, fea in enumerate(feature_list):
        try:
            batch_of_dict_arr.append(
                {
                    "IDX": downsample_arr(
                        idx_np[label_list[i]], 50 * flags.num_prev_cardiac_cycle_label
                    ),
                    "PPG": downsample_arr(
                        ppg_np[fea], 50 * flags.num_prev_cardiac_cycle_feature
                    ),
                    "ECG": downsample_arr(
                        ecg_np[fea], 50 * flags.num_prev_cardiac_cycle_feature
                    ),
                    "ABP": downsample_arr(
                        abp_np[label_list[i]], 50 * flags.num_prev_cardiac_cycle_label
                    ),
                }
            )
        except Exception as e:
            continue
    return batch_of_dict_arr


def MIMIC_dataloader(flags: argparse.Namespace, override_sel_subject=None):
    """
    MIMIC_dataloader loads MIMIC waveform data according to the specific flags

    Actions:
        1. Identifying the subject it to be used
        2. Loading the corresponding subject's CSV file
        3. Extracting only a portion of CSV file if specified
        4. Applying FIR filter to PPG and removing zeros from the waveform
        5. Phase shifting the feature and label waveforms
        6. Formulating cardiac cycles from the filtered waveform
        7. Shuffling the datasets into train and test sets
        8. Splitting train/test sets from the filtered waveform

    Args:
        flags (argparse.Namespace): the flags containing specified parameters of the scripts
        override_sel_subject (int, optional): overrides the target subject to be loading. Defaults to None.

    Returns:
        torch.tensor: x_train
        torch.tensor: y_train
        torch.tensor: x_test
        torch.tensor: y_test
        argparse.Namespace: flags
    """
    # getting the selected subject from available list
    flags.subject = get_selected_subject(
        flags=flags, override_sel_subject=override_sel_subject
    )
    flags.subject_id = get_subject_id_from_str(flags.subject)
    print(f"Selected Subject Name == {flags.subject}")
    # loading selected subject's waveform into dataframe
    print(pretty_progress_bar("Loading Waveforms from CSV files"))
    # pdb.set_trace()
    if os.path.exists(f"{flags.rex_torch_path}mimic_patient_{flags.subject_id}_ppg.pt"):
        import torch

        X_data = torch.load(
            f"{flags.rex_torch_path}mimic_patient_{flags.subject_id}_ppg.pt"
        )
        y_data = torch.load(
            f"{flags.rex_torch_path}mimic_patient_{flags.subject_id}_abp.pt"
        )
    else:
        wdf = pd.read_csv(flags.data_path + flags.subject + "/waveforms.csv")
        # for more rapid results
        if flags.run_portion < 1:
            wdf = wdf[: int((flags.run_portion) * len(wdf))]
        elif flags.run_portion > 1:
            wdf = wdf[: int(flags.run_portion)]
        # dropping the zeros
        print(pretty_progress_bar("Filtering Waveforms"))
        cwdf = abp_waveform_filter(wdf)
        # applying FIR Filter
        cwdf["PPG"] = apply_FIR(cwdf["PPG"], numtaps=40)
        cwdf["ECG"] = apply_FIR(cwdf["ECG"], numtaps=40)
        # applying phase matching
        print(pretty_progress_bar("Phase Shifting waveforms"))
        cwdf = phase_shift(cwdf)
        # applying sliding window batch or cardiac segmentation
        print(pretty_progress_bar("Segmenting Waveforms into batches"))
        if flags.use_cardiac_seg:
            batched_arr = form_cardiac_cycles(cwdf, flags)
            # removing inconsistent window lapping
            batched_arr = drop_inconsistent_windows(batched_arr, thre=13)
            # batch into data available
            X_data, y_data = make_batched_data_cardiac_cycle(batched_arr, flags)
        else:
            batched_arr = form_sliding_window(
                cwdf, win_size=flags.win_size, overlap=flags.win_overlap
            )
            # removing inconsistent window lapping
            batched_arr = drop_inconsistent_windows(batched_arr, thre=13)
            # batch into data available
            X_data, y_data = make_batched_data(batched_arr, flags)
    # Shuffling data before train/test split
    if flags.shuffle_data:
        shuffled_indices = np.arange(len(X_data))
        np.random.shuffle(shuffled_indices)
        X_data = X_data[shuffled_indices]
        y_data = y_data[shuffled_indices]
        print(pretty_progress_bar("shuffling data from train/test split"))
    # Getting to train test split
    x_train = X_data[0 : int(len(X_data) * flags.training_size)]
    y_train = y_data[0 : int(len(y_data) * flags.training_size)]
    x_test = X_data[int(len(X_data) * flags.training_size) :]
    y_test = y_data[int(len(y_data) * flags.training_size) :]
    print(pretty_progress_bar(""))
    print(
        f"x_train={x_train.shape}",
        f"y_train={y_train.shape}",
        f"x_test={x_test.shape}",
        f"y_test={y_test.shape}",
    )
    print(pretty_progress_bar(""))
    return x_train, y_train, x_test, y_test, flags


def get_mimic_subject_lists(flags: argparse.Namespace) -> list:
    """get_mimic_subject_lists
        Locating the Selected Subject based on the selection of patient

    Arguments:
        flags {argpaese.Namespace} -- Parsed Flag containing specified parameters

    Returns:
        list -- the corresponding subject for the trial
    """
    if os.path.isdir(flags.data_path):
        subjects = next(os.walk(flags.data_path))[1]
        temp_subjects = []
        for i, sub in enumerate(subjects):
            if sub[0] == "p" and len(sub) == 24:
                waveform_size = os.stat(
                    glob.glob(flags.data_path + sub + "/waveforms.csv")[0]
                ).st_size
                if waveform_size > 4096:
                    temp_subjects.append(sub)
        subjects = temp_subjects
        print(f"There are {len(subjects)} subjects in MIMIC")
    else:
        with open("utils/mimic_file_list.txt") as file:
            lines = file.readlines()
            subjects = [line.strip() for line in lines]
    return subjects


def get_selected_subject(flags: argparse.Namespace, override_sel_subject=None) -> str:
    """get_selected_subject
        Locating the Selected Subject based on the selection of patient

    Arguments:
        flags {argpaese.Namespace} -- Parsed Flag containing specified parameters
        override_sel_subject {int, optional} -- Overrides the selected subject. Defaults to None.

    Returns:
        str -- the corresponding subject for the trial
    """
    subjects = get_mimic_subject_lists(flags)
    # if selected subject is already subject in subject, then return it
    idf_id = (
        int(override_sel_subject) if override_sel_subject != None else flags.sel_subject
    )
    for i, sub in enumerate(subjects):
        sub_id = get_subject_id_from_str(sub)
        if sub_id == idf_id:
            print(
                pretty_progress_bar(
                    f"--sel_subject is absolute subject ID  == {sub_id}"
                )
            )
            return sub
    print(pretty_progress_bar(f"--sel_subject is the index == {idf_id}"))
    return subjects[flags.sel_subject]


def calc_metrics(pred, test):
    """calc_metrics
        Calculating the needed metrics based on the provided prediction and ground truth

    Arguments:
        pred {list} -- predicted array
        test {list} -- ground truth array

    Returns:
        list of floats -- RMSE, MAE, Pearon's R and P value
    """
    rmse = math.sqrt(mean_squared_error(test, pred))
    mae = mean_absolute_error(test, pred)
    mean = np.mean(pred)
    std = np.std(pred)
    rval, pval = pearsonr(test, pred)
    return rmse, mae, mean, std, rval, pval


def apply_FIR(ppg_raw, numtaps=40):
    """apply_FIR
        Applying FIR filter on PPG Waveform

    Arguments:
        ppg_raw {list} -- raw PPG waveform

    Keyword Arguments:
        numtaps {int} -- Number of taps (default: {40})

    Returns:
        list -- Filtered PPG waveforms
    """
    taps = firwin(numtaps, [0.5, 8], pass_zero="bandpass", fs=125)
    fir_ppg = lfilter(taps, 1.0, ppg_raw)
    return fir_ppg


def phase_shift(waveform_df):
    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    """phase_shift 
        Performing phase shift to align PPG and ABP signals

    Arguments:
        waveform_df {dataframe} -- Dataframe containing PPG and ABP waveforms

    Returns:
        dataframe -- Aligned Dataframe Waveform 
    """
    try:
        if len(waveform_df["ABP"] > 1000):
            lag_val = pa.find_phase_align(
                np.diff(normalize(waveform_df["PPG"][-1000:].to_numpy())),
                np.diff(normalize(waveform_df["ABP"][-1000:].to_numpy())),
            )
        else:
            lag_val = pa.find_phase_align(
                np.diff(normalize(waveform_df["PPG"].to_numpy())),
                np.diff(normalize(waveform_df["ABP"].to_numpy())),
            )
        waveform_df["ABP"] = waveform_df["ABP"].shift(int(lag_val))
        waveform_df.dropna(inplace=True)
        print("lag", lag_val)
    except Exception as e:
        print(e)
        print("phase shift matching failed, outputting raw waveforms")
    finally:
        return waveform_df


def abp_waveform_filter(waveform_df):
    """abp_waveform_filter
        Filtering waveforms with wrong ABP waveform
        Cleaning any waveform with 3 consecutive flat values

    Arguments:
        waveform_df {dataframe} -- Waveform Dataframe

    Returns:
        dataframe -- Filtered Waveform
    """
    waveform_df["ABP_flat"] = waveform_df["ABP"].diff(periods=2)
    waveform_df = waveform_df[(waveform_df["ABP_flat"] != 0)]
    # waveform_df = waveform_df[(waveform_df["ABP"] <= 200)]
    # waveform_df = waveform_df[(waveform_df["ABP"] >= 20)]
    cleaned_df = pd.DataFrame()
    cleaned_df["ABP"] = waveform_df["ABP"]
    cleaned_df["PPG"] = waveform_df["PPG"]
    cleaned_df["ECG"] = waveform_df["II"]
    cleaned_df["IDX"] = waveform_df["Unnamed: 0"]
    return cleaned_df


def drop_inconsistent_windows(batch_of_dict_arr, thre=3):
    """drop_inconsistent_windows
        Dropping windows with inconsistent waveform (remove the ones with gap)

    Arguments:
        batch_of_dict_arr {list} -- Array of dictionaries

    Keyword Arguments:
        thre {int} -- Threshold for the gap (default: {3})
    Returns:
        out_arr {list} -- Array of output values
    """

    def is_arr_continuous(arr: np.ndarray, thre: int) -> bool:
        """
        is_arr_continuous Determining whether the index within array is continuous

        Args:
            arr (np.ndarray): array to be checked
            thre (int): number of index threshold

        Returns:
            bool: whether it meets the criteria or not
        """
        for i, e in enumerate(arr[:-1]):
            if arr[i + 1] - arr[i] > thre:
                return False
        return True

    def is_arr_outlier_free(arr: np.ndarray, thre=[]) -> bool:
        """
        is_arr_outlier_free Returns True if the array is outlier free

        Args:
            arr (np.ndarray): array to be checked for outliers
            thre (list, optional): lower and upper thresholds. Defaults to [].

        Returns:
            bool: whether this segment is true or false
        """
        # If thresholds not selected, then exit immediately
        if len(thre) == 0:
            return True
        # If the array contains outlier, then return True
        if arr.min() < thre[0] or arr.max() > thre[1]:
            return False
        # Otherwise, return False
        return True

    out_arr = []

    ppg_max = np.array([dic["PPG"].max() for dic in batch_of_dict_arr]).reshape(-1)
    ppg_min = np.array([dic["PPG"].min() for dic in batch_of_dict_arr]).reshape(-1)
    ppg_threshold = [np.percentile(ppg_min, 5), np.percentile(ppg_max, 95)]
    for dic in batch_of_dict_arr:
        if is_arr_continuous(dic["IDX"], thre=thre):
            if is_arr_outlier_free(dic["ABP"], thre=[20, 200]):
                if is_arr_outlier_free(dic["PPG"], thre=ppg_threshold):
                    out_arr.append(dic)
    return out_arr


def form_sliding_window(waveform_df, win_size=256, overlap=64):
    """form_sliding_window
        Formulating Sliding Windows by segmenting the input waveform dataframe

    Arguments:
        waveform_df {dataframe} -- Dataframe containing Waveforms

    Keyword Arguments:
        win_size {int} -- Size of each segmented waveform (default: {256})
        overlap {int} -- # of overlapped values between windows (default: {64})

    Returns:
        list -- Array of dictionaries ready to be processed and combined
    """
    abp_np = waveform_df["ABP"].to_numpy()
    ppg_np = waveform_df["PPG"].to_numpy()
    ecg_np = waveform_df["ECG"].to_numpy()
    idx_np = waveform_df["IDX"].to_numpy()
    batch_of_dict_arr = []
    whole_list = np.arange(len(idx_np))
    range_list = [
        whole_list[i : i + win_size]
        for i in range(0, len(whole_list), win_size - overlap)
    ]
    for batch in range_list:
        batch_of_dict_arr.append(
            {
                "IDX": idx_np[batch],
                "PPG": ppg_np[batch],
                "ECG": ecg_np[batch],
                "ABP": abp_np[batch],
            }
        )
    return batch_of_dict_arr


def make_batched_data_cardiac_cycle(batched_arr, flags):
    """make_batched_data_cardiac_cycle
        (cardiac cycle version) Assigning PPG and ABP data into torch ready train/test pairs

    Arguments:
        batched_arr {list} -- list of dictionaries based on windows
        flags {flag} -- list of arg parsed flags

    Returns:
        np.array -- torch array of input features
        np.array -- torch array of ground truth labels
    """
    X = []
    y = []
    for batch in batched_arr:
        if len(batch["PPG"]) == flags.num_prev_cardiac_cycle_feature * 50:
            if flags.model_used == "Sequnet":
                feat = batch["PPG"]
            else:
                feat = np.concatenate(
                    (batch["PPG"].reshape(-1, 1), batch["ECG"].reshape(-1, 1)), axis=1
                )
            X.append(feat)
            y.append(batch["ABP"])
    print(np.array(X).shape)
    if flags.model_used == "Sequnet":
        import torch

        X = np.array(X).reshape(-1, 1, flags.num_prev_cardiac_cycle_feature * 50)
        y = np.array(y).reshape(-1, 1, flags.num_prev_cardiac_cycle_label * 50)
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
    elif flags.model_used == "Transformer":
        X = np.array(X)
        y = np.array(y)
    else:
        raise Exception(f"[Sicong] Unknown Value: flags.model_used:{flags.model_used}")

    if flags.shuffle_data:
        shuffled_indices = np.arange(len(X))
        np.random.shuffle(shuffled_indices)
        X = X[shuffled_indices]
        y = y[shuffled_indices]
        print("shuffling data from train/test split")
    return X, y


def make_batched_data(batched_arr, flags):
    """make_batched_data
        Assigning PPG and ABP data into torch ready train/test pairs

    Arguments:
        batched_arr {list} -- list of dictionaries based on windows
        flags {flag} -- list of arg parsed flags

    Returns:
        torch.tensor -- torch array of input features
        torch.tensor -- torch array of ground truth labels
    """
    X = []
    y = []
    for batch in batched_arr:
        if len(batch["PPG"]) == flags.win_size:
            if flags.model_used == "Sequnet":
                feat = batch["PPG"]
            else:
                feat = np.concatenate(
                    (batch["PPG"].reshape(-1, 1), batch["ECG"].reshape(-1, 1)), axis=1
                )
            X.append(feat)
            y.append(batch["ABP"])
    print(np.array(X).shape)
    if flags.model_used == "Sequnet":
        import torch

        X = np.array(X).reshape(-1, 1, flags.win_size)
        y = np.array(y).reshape(-1, 1, flags.win_size)
        X = torch.from_numpy(X).float()  # stay at cpu to that gpu don't get overloaded
        y = torch.from_numpy(y).float()
    elif flags.model_used == "Transformer":
        X = np.array(X)
        y = np.array(y)
    else:
        raise Exception(f"[Sicong] Unknown Value: flags.model_used:{flags.model_used}")

    if flags.shuffle_data:
        shuffled_indices = np.arange(len(X))
        np.random.shuffle(shuffled_indices)
        X = X[shuffled_indices]
        y = y[shuffled_indices]
        print("shuffling data from train/test split")
    return X, y


def smoothing(data):
    """smoothing
        Smoothing the array by removing zeroed value entries

    Arguments:
        data {list} -- Array of data to be smoothed

    Returns:
        list -- Array of smoothed data
    """
    smoothed = []
    for i in range(len(data) - 1):
        if data[i] == 0:
            smoothed.append(data[i + 1])
        else:
            smoothed.append(data[i])
    smoothed.append(data[-1])
    return np.array(smoothed)


def check_valid_peaks_valleys(y_arr):
    """check_valid_peaks_valleys
        Checking whether the input array contains valid peaks and valleys

    Arguments:
        y_arr {list} -- Array of segmented waveform

    Returns:
        Boolean -- Whether there was at least 1 matching waveform
    """
    y_peaks, _ = find_peaks(y_arr, distance=50, height=80)
    y_troughs, _ = find_peaks(-y_arr, distance=50, height=-120)
    if len(y_peaks) < 1 or len(y_troughs) < 1:
        return False
    return True


def cumulative_error(pred_arr, test_arr, bp_type="sbp"):
    """cumulative_error
        Calculating and Reporting Cumulative Error for such data

    Arguments:
        pred_arr {list} -- _description_
        test_arr {list} -- _description_

    Keyword Arguments:
        bp_type {str} -- _description_ (default: {"sbp"})

    Returns:
        str -- Tabulated Data reporting percentage in cumulative error
    """
    abs_error = np.absolute(pred_arr - test_arr)
    within_5 = (abs_error <= 5).sum() / len(abs_error)
    within_10 = (abs_error <= 10).sum() / len(abs_error)
    within_15 = (abs_error <= 15).sum() / len(abs_error)
    return tabulate(
        [
            [
                bp_type.upper(),
                "{:.2%}".format(within_5),
                "{:.2%}".format(within_10),
                "{:.2%}".format(within_15),
            ]
        ],
        headers=["BP_type", "<=5 mmHg", "<=10 mmHg", "<=15 mmHg"],
    )


def early_stopping_trigger(metric_arr, patience=5, metric_goal="min"):
    """early_stopping_trigger
        Stopping model train if loss is not improved for some consecutive epochs

    Arguments:
        metric_arr {list} -- Array of losses correponding to epochs

    Keyword Arguments:
        patience {int} -- Choosing how long to stop when no improvement was done (default: {5})
        metric_goal {str} -- Which Metric does this follow (default: {"min"})

    Returns:
        Boolean -- Whether the early stopping criteria has met
    """
    metric_goal = metric_goal.lower()
    if len(metric_arr) <= patience**2:
        return False
    elif patience < 0:
        return False
    elif patience < 3:
        patience = 3
        # print('Early Stopping Trigger must have patience be greater than or equal to 3')
    if metric_goal == "min":
        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        return is_sorted(metric_arr[-patience:])
    elif metric_goal == "max":
        is_sorted = lambda a: np.all(a[:-1] >= a[1:])
        return is_sorted(metric_arr[-patience:])
    print(f"Unknown Goal:{metric_goal}, Skipping Trigger")
    return False


def add_noise_2_sequence(
    sequence: np.ndarray,
    noise_rate: float,
    noise_type="gaussian",
) -> np.ndarray:
    """
    add_noise_2_sequence add random noise to a given sequence

    Args:
        sequence (np.ndarray): sequence which gaussian noise being added to
        noise_rate (float): Multiplier of the noise rate (0~1).
        noise_type (str, optional): Type of noise wish to add. Defaults to "gaussian".

    Returns:
        np.ndarray: noise-embedded sequence
    """
    if noise_type == "gaussian":
        noise = np.around(
            np.random.normal(loc=0, scale=0.1, size=sequence.shape),
            decimals=3,
        )
        # print(f"noise = {noise}")
    return sequence + (noise_rate * noise)


def is_abp_above_threshold(abp_win: np.ndarray, bp_type: str, threshold=140) -> bool:
    """
    is_abp_above_threshold determine whehter the given ABP value is above certain threshold

    Args:
        abp_win (np.ndarray): The ABP array window being processed
        bp_type (str): blood pressure type (either sbp or dbp)
        threshold (int, optional): threshold to be measured. Defaults to 140.

    Raises:
        Exception: When bp_type is not sbp or dbp

    Returns:
        bool: Determine whehter ABP is above or belo
    """
    if bp_type == "sbp":
        thre = abp_win.max()
    elif bp_type == "dbp":
        thre = abp_win.min()
    else:
        raise Exception("unknown BP type")
    return thre > threshold


def quick_plot(arr, label=None):
    plt.close()
    plt.style.use("seaborn")
    if label:
        plt.plot(arr, label=label)
        plt.legend()
    else:
        plt.plot(arr)
    plt.savefig("asd")


def visual_pred_test(
    pred_arr,
    test_arr,
    title="default title",
    x_lab="default_x",
    y_lab="default_y",
    plot_style=".-.",
    add_text=[],
):
    plt.style.use("fivethirtyeight")
    plt.title(title)
    plt.plot(pred_arr, plot_style, label="pred", lw=2)
    plt.plot(test_arr, plot_style, label="test", lw=2)
    if len(add_text) != 0:
        for i, t in enumerate(add_text):
            x_pos = int(len(pred_arr) * 0.9)
            y_pos = int(max(pred_arr) * (1 - 0.1 * i))
            plt.text(x_pos, y_pos, add_text, color="k", fontsize=12)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.legend(loc=1)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    fig.set_dpi(100)
    plt.close()
    return fig


def get_motion_artifact_ppg(
    out_fs: int,
    file_path="/data/datasets/ppg_motion_artifact/wrist-ppg-during-exercise-1.0.0",
):
    record = wfdb.rdrecord(file_path + "/s9_walk")
    in_fs = record.fs
    wrist_ppg = record.p_signal[:, 1]
    wrist_ppg = wrist_ppg[~np.isnan(wrist_ppg)]
    new_ppg = resample(wrist_ppg, int(len(wrist_ppg) * out_fs / in_fs))
    return new_ppg


def pretty_progress_bar(msg="", style="=", total_len=100):
    if msg == "":
        return "=" * total_len
    num_skip = (total_len - len(msg)) / 2
    guard = style * int(num_skip / len(style))
    return f"{guard}{msg}{guard}{style*2}"[:total_len]


def get_subject_id_from_str(subject_str: str) -> int:
    """
    get_subject_id_from_str returns the subject id in integar from the subject string

    Args:
        subject_str (str): the subject string recorded in the directory

    Returns:
        int: subject id in integer form
    """
    return int(subject_str.split("-")[0][1:])


if __name__ == "__main__":
    # su.make_figlet("Seq2Seq_Utils Unit Testing")
    # x_test = np.load("x_test.npy")
    # orig = x_test[0, 0, :]
    # # orig = np.randn.random(0, 1, 0.025).reshape(4, 1, -1)
    # # print(f"orig: {orig}")
    # noise_rate = 0.075
    # after = add_noise_2_sequence(orig, noise_rate=noise_rate)
    # # print(f"after: {after}")
    # plt.style.use("fivethirtyeight")
    # plt.plot(orig.reshape(-1), "-", label="orig")
    # plt.plot(after.reshape(-1), "-", label="after", alpha=0.8)
    # plt.title(f"noise_rate=={noise_rate}")
    # plt.legend(loc=1)
    # new_ppg = get_motion_artifact_ppg(out_fs=125)
    flags = sicong_argparse(model="Sequnet")
    sub = get_selected_subject(flags, override_sel_subject=6)
    # pdb.set_trace()
