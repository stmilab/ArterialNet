from __future__ import print_function
import pandas as pd
import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seq2seq_utils as zu
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
import sys
import wfdb
import pdb

import phase_alignment as pa

# Compatibility layer between Python 2 and Python 3
from scipy.signal import resample
import argparse
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import torch
import wandb

ablation_list = [
    "shrink_data_size",
    "hide_bp_range",
    "domain_segmentor",
    "gaussian_noise",
    "mask_first_half",
    "mask_last_half",
    "mask_current_beat",
    "mask_prev_beat",
    "calibration_time",
    "None",
]


def plot_ablated_data(orig: torch.Tensor, after: torch.Tensor, noise_type: str):
    plt.style.use("ggplot")
    plt.plot(orig.reshape(-1), "-", label="orig", alpha=0.6)
    plt.plot(after.reshape(-1), "-", label="after", alpha=0.6)
    plt.title(f"noise_type=={noise_type}")
    plt.legend(loc=1)
    return plt


def mask_current_beat(x_data: torch.Tensor, beat_len: int = 150):
    """
    mask_current_beat masking current beat and using only previous beats

    Args:
        x_data (torch.Tensor): feature sequence to be masked
        beat_len (int, optional): the length of a single beat. Defaults to 300.
    """
    if len(x_data.shape) == 2:
        for i, _ in enumerate(x_data):
            x_data[i, -beat_len:] = x_data.min()
    elif len(x_data.shape) == 3:
        for i, _ in enumerate(x_data):
            x_data[i, :, -beat_len:] = x_data.min()
    else:
        raise ValueError("sequence must be 2D or 3D")
    return x_data


def mask_prev_beat(x_data: torch.Tensor, beat_len: int = 150):
    """
    mask_prev_beat masking current beat and using only previous beats

    Args:
        x_data (torch.Tensor): feature sequence to be masked
        beat_len (int, optional): the length of a single beat. Defaults to 300.
    """
    if len(x_data.shape) == 2:
        for i, _ in enumerate(x_data):
            x_data[i, :-beat_len] = x_data.min()
    elif len(x_data.shape) == 3:
        for i, _ in enumerate(x_data):
            x_data[i, :, :-beat_len] = x_data.min()
    else:
        raise ValueError("sequence must be 2D or 3D")
    return x_data


def shrink_data_size(x_data: torch.Tensor, y_data: torch.Tensor, train_size: float):
    """
    shrink_data_size prepares the dataset with selected number of datas

    Args:
        x_data (torch.Tensor): feature data, pre-segmented
        y_data (torch.Tensor): label data, pre-segmented
        train_size (float): size of training option (0~1)
    """
    split_idx = int(len(x_data) * train_size)
    x_train = x_data[:split_idx]
    y_train = y_data[:split_idx]
    x_test = x_data[split_idx:]
    y_test = y_data[split_idx:]
    return x_train, y_train, x_test, y_test


def apply_calibration_time(
    x_data: torch.Tensor, y_data: torch.Tensor, calibration_time: float
):
    """
    apply_calibration_time prpares the dataset with selected amount of data

    Args:
        x_data (torch.Tensor): feature data, pre-segmented
        y_data (torch.Tensor): label data, pre-segmented
        train_size (float): size of training option (0~1)
    """
    split_idx = calibration_time * 60
    x_train = x_data[:split_idx]
    y_train = y_data[:split_idx]
    x_test = x_data[split_idx:]
    y_test = y_data[split_idx:]
    return x_train, y_train, x_test, y_test


def mask_last_half(
    sequence: torch.Tensor,
    cut_off: float = 0.5,
    waveform_len=150,
):
    mask_bound = int(waveform_len * cut_off)
    mask_idx = np.array([])
    for i in range(sequence.shape[-1] // waveform_len):
        mask_idx = np.concatenate(
            (
                mask_idx,
                np.arange(
                    i * waveform_len + mask_bound, i * waveform_len + waveform_len
                ),
            ),
            axis=0,
        )
    if len(sequence.shape) == 2:
        for i, _ in enumerate(sequence):
            sequence[i, mask_idx] = sequence.min()
    elif len(sequence.shape) == 3:
        for i, _ in enumerate(sequence):
            sequence[i, :, mask_idx] = sequence.min()
    else:
        raise ValueError("sequence must be 2D or 3D")
    return sequence


def mask_first_half(
    sequence: torch.Tensor,
    cut_off: float = 0.5,
):
    mask_bound = int(300 * cut_off)
    mask_idx = np.array([])
    for i in range(sequence.shape[-1] // 300):
        mask_idx = np.concatenate(
            (mask_idx, np.arange(i * 300, i * 300 + mask_bound)), axis=0
        )
    if len(sequence.shape) == 2:
        sequence[mask_idx] = sequence.min()
    elif len(sequence.shape) == 3:
        for i, _ in enumerate(sequence):
            sequence[i, :, mask_idx] = sequence.min()
    else:
        raise ValueError("sequence must be 2D or 3D")
    return sequence


def add_gaussian_noise(
    sequence: torch.Tensor,
    noise_rate: float = 0.75,
) -> torch.Tensor:
    """
    add_gaussian_noise adds gaussian noise to the input sequence.

    Args:
        sequence (torch.Tensor): sequence which gaussian noise being added to
        noise_rate (float): Multiplier of the noise rate (0~1).

    Returns:
        torch.Tensor: sequence with gaussian noise
    """
    # generating a gaussian noise with the same shape as the input sequence (with 3 decimal places)
    noise = np.around(
        np.random.normal(
            loc=sequence.mean(),
            scale=sequence.std() / 2,
            size=sequence.shape,
        ),
        decimals=3,
    )
    return sequence + torch.from_numpy(noise_rate * noise).float()


def add_motion_artifact_ppg(
    out_fs: int,
    file_path="/data/datasets/ppg_motion_artifact/wrist-ppg-during-exercise-1.0.0",
):
    """
    add_motion_artifact_ppg Adding the motion artifact to the PPG signal (PPG only)

    Args:
        out_fs (int): output frequency
        file_path (str, optional): path to motion dataset. Defaults to "/data/datasets/ppg_motion_artifact/wrist-ppg-during-exercise-1.0.0".

    Returns:
        torch.Tensor: PPG sequence with gaussian noise
    """
    record = wfdb.rdrecord(file_path + "/s9_walk")
    in_fs = record.fs
    wrist_ppg = record.p_signal[:, 1]
    wrist_ppg = wrist_ppg[~np.isnan(wrist_ppg)]
    new_ppg = resample(wrist_ppg, int(len(wrist_ppg) * out_fs / in_fs))
    return new_ppg


# END of Imports from BioZ Ablation counterpart


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
    zu.pretty_progress_bar("Using Mutated Dataloader to Test Robustness!")
    # getting the selected subject from available list
    flags.subject = zu.get_selected_subject(
        flags=flags, override_sel_subject=override_sel_subject
    )
    flags.subject_id = zu.get_subject_id_from_str(flags.subject)
    print(f"Selected Subject Name == {flags.subject}")
    # loading selected subject's waveform into dataframe
    print(zu.pretty_progress_bar("Loading Waveforms from CSV files"))

    if os.path.exists(f"{flags.rex_torch_path}mimic_patient_{flags.subject_id}_abp.pt"):
        import torch

        X_data = torch.load(
            f"{flags.rex_torch_path}mimic_patient_{flags.subject_id}_feat.pt"
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
        print(zu.pretty_progress_bar("Filtering Waveforms"))
        cwdf = zu.abp_waveform_filter(wdf)
        # applying FIR Filter
        cwdf["PPG"] = zu.apply_FIR(cwdf["PPG"], numtaps=40)
        cwdf["ECG"] = zu.apply_FIR(cwdf["ECG"], numtaps=40)
        # applying phase matching
        print(zu.pretty_progress_bar("Phase Shifting waveforms"))
        cwdf = zu.phase_shift(cwdf)
        # applying sliding window batch or cardiac segmentation
        print(zu.pretty_progress_bar("Segmenting Waveforms into batches"))
        if flags.use_cardiac_seg:
            batched_arr = zu.form_cardiac_cycles(cwdf, flags)
            # removing inconsistent window lapping
            batched_arr = zu.drop_inconsistent_windows(batched_arr, thre=13)
            # batch into data available
            X_data, y_data = zu.make_batched_data_cardiac_cycle(batched_arr, flags)
        else:
            batched_arr = zu.form_sliding_window(
                cwdf, win_size=flags.win_size, overlap=flags.win_overlap
            )
            # removing inconsistent window lapping
            batched_arr = zu.drop_inconsistent_windows(batched_arr, thre=13)
            # batch into data available
            X_data, y_data = zu.make_batched_data(batched_arr, flags)

    # Selecting train/test strategy
    zu.make_figlet(f"Ablation: {flags.ablation_type}")
    if flags.ablation_type == "domain_segmentor":
        print(zu.pretty_progress_bar("Segmenting Domain"))
        x_train, y_train, x_test, y_test = domain_segmentor(
            x_data=X_data,
            y_data=y_data,
            domain_length=360,
            shuffle=flags.shuffle_data,
        )
    elif flags.ablation_type == "hide_bp_range":
        print(zu.pretty_progress_bar("Hiding BP Range"))
        x_train, y_train, x_test, y_test = hide_bp_range(
            x_data=X_data,
            y_data=y_data,
            bp_type=flags.hide_bp_type,
            range_list=flags.hide_bp_range,
        )
    elif flags.ablation_type == "shrink_data_size":
        x_train, y_train, x_test, y_test = shrink_data_size(
            x_data=X_data,
            y_data=y_data,
            train_size=flags.training_size,
        )
    elif flags.ablation_type == "calibration_time":
        x_train, y_train, x_test, y_test = apply_calibration_time(
            x_data=X_data,
            y_data=y_data,
            calibration_time=flags.calibration_time,
        )
    else:
        # default: When no data-split-level ablation is selected, we still use shrink data size
        x_train, y_train, x_test, y_test = shrink_data_size(
            x_data=X_data,
            y_data=y_data,
            train_size=flags.training_size,
        )
    orig_x = None
    if flags.ablation_type == "gaussian_noise":
        orig_x = x_test.clone()
        x_test = add_gaussian_noise(
            sequence=x_test,
            noise_rate=flags.gaussian_noise_rate,
        )
    elif flags.ablation_type == "mask_first_half":
        orig_x = x_test.clone()
        x_test = mask_first_half(sequence=x_test)
    elif flags.ablation_type == "mask_last_half":
        orig_x = x_test.clone()
        x_test = mask_last_half(sequence=x_test)
    elif flags.ablation_type == "mask_current_beat":
        orig_x = x_test.clone()
        x_test = mask_current_beat(x_data=x_test)
    elif flags.ablation_type == "mask_prev_beat":
        orig_x = x_test.clone()
        x_test = mask_prev_beat(x_data=x_test)
    # Visualizing the alated data
    if orig_x is not None:
        figure = plot_ablated_data(
            orig=orig_x[:3, :],
            after=x_test[:3, :],
            noise_type=flags.ablation_type,
        )
        if flags.use_wandb:
            wandb.log(
                {
                    "ablated_data": wandb.Image(
                        figure,
                        caption=f"{flags.ablation_type}",
                    )
                }
            )
        else:
            figure.savefig(f"plot_dir/ablate_{flags.ablation_type}.png")
    print(
        f"Training Data Shape: {x_train.shape}, ",
        f"Testing Data Shape: {x_test.shape},",
        f" Total Data Shape: {X_data.shape}",
    )
    return x_train, y_train, x_test, y_test, flags


def shrink_data_size(x_data: torch.Tensor, y_data: torch.Tensor, train_size: float):
    """
    shrink_data_size prpares the dataset with selected data size

    Args:
        x_data (torch.Tensor): feature data, pre-segmented
        y_data (torch.Tensor): label data, pre-segmented
        train_size (float): size of training option
    """
    zu.pretty_progress_bar("Shrinking Data Size")
    split_idx = int(len(x_data) * train_size)
    x_train = x_data[:split_idx]
    y_train = y_data[:split_idx]
    x_test = x_data[split_idx:]
    y_test = y_data[split_idx:]
    return x_train, y_train, x_test, y_test


def hide_bp_range(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    bp_type: str,
    range_list: list,
):
    """
    hide_bp_range generate dataset by hiding some bp ranges (SBP/DBP) from training

    Args:
        x_data (torch.Tensor): feature data, pre-segmented
        y_data (torch.Tensor): label data, pre-segmented
        bp_type (str): whether being sbp or dbp
        range_list (list): lower and upper range to be hidden, must be length of 2
    """
    # sanity check: raise error when range is not matched
    if len(range_list) != 2:
        raise ValueError(f"Invalid rnage list: len(range_list)={len(range_list)} != 2")
    # sanity check: raise error when bp_type is not matched
    if bp_type.lower() not in ["sbp", "dbp"]:
        raise ValueError(f"Invalid BP type: bp_type={bp_type.lower()} panicking!")
    # Identifying the index to be fed
    train_idx = []
    test_idx = []
    for idx in range(len(y_data)):
        bp_val = y_data[idx].max() if bp_type.lower() == "sbp" else y_data[idx].min()
        if bp_val < range_list[0] or bp_val > range_list[1]:
            train_idx.append(idx)
        else:
            test_idx.append(idx)
    x_train = x_data[train_idx]
    y_train = y_data[train_idx]
    x_test = x_data[test_idx]
    y_test = y_data[test_idx]
    return x_train, y_train, x_test, y_test


def domain_segmentor(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    domain_length: int,
    shuffle=False,
):
    """
    domain_segmentor Segments the data into domains with custom length

    Data assumes a unit of cardiac cycles, therefore you need to convert time into # of cardiac cycles

    Args:
        x_data (torch.Tensor): feature data, pre-segmented into cardiac cycles
        y_data (torch.Tensor): label data, pre-segmented into cardiac cycles
        domain_length (int): number of cardiac cycles per domain
        shuffle (bool, optional): whether to shuffle the domains. Defaults to False.
    """
    # Calculate number of domains
    split_size = domain_length
    # Segment feat and label tensors into domains (torch.split() returns tuple, so convert to list)
    train_idx = []
    test_idx = []
    domain_idx_range = np.arange(0, len(y_data), split_size)
    if shuffle:
        np.random.shuffle(domain_idx_range)
    for idx in domain_idx_range:
        max_idx = min(idx + split_size, len(y_data) - 1)
        if idx // split_size % 2 == 0:
            train_idx += [x for x in range(idx, max_idx)]
        else:
            test_idx += [x for x in range(idx, max_idx)]
    x_train = x_data[train_idx]
    y_train = y_data[train_idx]
    x_test = x_data[test_idx]
    y_test = y_data[test_idx]

    print(f"Number of domains: {len(y_data) // split_size}")
    return x_train, y_train, x_test, y_test


def make_histogram(y_data, hide_bp_type="sbp", range_list=[0, 0]):
    sbp_vals = []
    dbp_vals = []
    for idx in range(len(y_data)):
        sbp_vals.append(y_data[idx].max().item())
        dbp_vals.append(y_data[idx].min().item())

    # Start plotting
    plt.figure(figsize=(13, 5), dpi=100)

    plt.subplot(1, 2, 1)
    sns.histplot(
        data=sbp_vals,
        bins=np.arange(min(sbp_vals) // 10 * 10, max(sbp_vals) // 10 * 10 + 5, 5),
        stat="density",
        color="blue",
        alpha=0.7,
    )
    plt.title("Distribution of SBP Values")
    plt.xlabel("SBP Value (mmHg)")
    plt.ylabel("density")
    if hide_bp_type == "sbp":
        # Add a square and rectangles
        x1, x2 = range_list
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, 0),
                x2 - x1,
                plt.gca().get_ylim()[1],
                color="gray",
                alpha=0.4,
                label="Hidden Range",
            )
        )

    plt.legend()
    plt.subplot(1, 2, 2)
    sns.histplot(
        data=dbp_vals,
        bins=np.arange(min(dbp_vals) // 10 * 10, max(dbp_vals) // 10 * 10 + 5, 5),
        stat="density",
        color="green",
        alpha=0.7,
    )

    plt.title("Distribution of DBP Values")
    plt.xlabel("DBP Value (mmHg)")
    plt.ylabel("density")
    if hide_bp_type == "dbp":
        # Add a square and rectangles
        x1, x2 = range_list
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, 0),
                x2 - x1,
                plt.gca().get_ylim()[1],
                color="gray",
                alpha=0.4,
                label="Hidden Range",
            )
        )

    plt.legend()
    plt.savefig("sbp_dbp_hist.png")
