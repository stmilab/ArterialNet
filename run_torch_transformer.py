import argparse
import os
import sys

import numpy as np
import pdb
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks, peak_prominences
from torch import optim
from tqdm import tqdm
from scipy.stats import skew
import warnings

# importing my own utility functions
sys.path.append("utils/")
sys.path.append("models/")
from utils.arg_parser import sicong_argparse
from utils.visual_combine import MIMIC_Visual
import utils.seq2seq_utils as zu
from models.sequnet import DilatedCNN, TransformerModel, Sequnet as SeqUNet
from utils import torch_metrics

warnings.filterwarnings("ignore")


def calc_stat_labels(y_arr: torch.Tensor) -> torch.Tensor:
    """calc_stat_labels computes statistical labels for the given waveform arrays

    Arguments:
        y_arr {torch.Tensor} -- The waveform array used to extract statistical labels

    Returns:
        torch.Tensor -- Array of in the order of [skewness, mean, min, max, std]
    """
    # Making labels
    with torch.no_grad():
        stats_list = [
            [
                skew(y_arr[i, 0, :].reshape(-1)),  # skewness
                y_arr[i, 0, :].reshape(-1).mean(),  # mean
                y_arr[i, 0, :].reshape(-1).min(),  # min
                y_arr[i, 0, :].reshape(-1).max(),  # max
                y_arr[i, 0, :].reshape(-1).std(),  # std
            ]
            for i in range(len(y_arr))
        ]
    return torch.Tensor(stats_list).view(len(stats_list), 1, len(stats_list[-1]))


def train_epoch(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_norm,
    y_norm,
    net: torch.nn,
    optimizer: torch.optim,
    flags: argparse.Namespace,
):
    """
    train_epoch trains the entire training dataset for one epoch

    Actions:
        1. Iteratively train batch by batch
        2. Normalize the feature datasets in the beginning
        3. Compute predicted value and loss
        4. Backpropagate the loss

    Args:
        x_train (torch.Tensor): feature dataset to be trained
        y_train (torch.Tensor): label dataset to be evaluated
        x_norm (_type_): normalizer for feature dataset
        y_norm (_type_): normalizer for label dataset
        net (torch.nn): the neural network model to be trained
        optimizer (torch.optim): the optimizer of the model
        flags (argparse.Namespace): flags specifying the training setup

    Returns:
        torch.Tensor: Combined Loss of the entire epoch in RMSE
        torch.Tensor: Combined Pearson's R of the entire epoch
        torch.Tensor: Combined denormalized Predicted waveform of the entire epoch
        torch.optm: Optimizer of the model
    """
    entire_epoch_preds = torch.Tensor([])
    with torch.no_grad():
        normalized_x_train = x_norm.normalize(x_train)
    for i in range(0, len(x_train) - flags.batch_size, flags.batch_size):
        # getting batch
        if len(x_train) < 2 * flags.batch_size + i:
            x_batch = normalized_x_train[i:]
            y_batch = y_train[i:]
        else:
            x_batch = normalized_x_train[i : i + flags.batch_size]
            y_batch = y_train[i : i + flags.batch_size]
        # masking the data if needed
        if flags.mask_abp_threshold != 0 and zu.is_abp_above_threshold(
            abp_win=y_batch.detach().numpy(),
            bp_type="sbp",
            threshold=flags.mask_abp_threshold,
        ):
            continue
        # zero the gradints
        optimizer.zero_grad()
        # prediction
        y_batch_preds = net(x_batch.to(flags.device)).cpu()
        denormalized_y_batch_preds = y_norm.denormalize(y_batch_preds)
        entire_epoch_preds = torch.concat(
            [entire_epoch_preds, denormalized_y_batch_preds]
        )
        # compute waveform loss (RMSE)
        waveform_rmse = torch_metrics.calc_RMSE(denormalized_y_batch_preds, y_batch)
        # Compute statistical loss (RMSE)
        label_stats = calc_stat_labels(y_batch)
        pred_stats = calc_stat_labels(denormalized_y_batch_preds)
        stat_rmse = torch_metrics.calc_RMSE(label_stats, pred_stats)
        loss = waveform_rmse * 0.3 + stat_rmse * 0.7
        # compute Pearson's R
        pearson = torch_metrics.calc_Pearson(denormalized_y_batch_preds, y_batch)
        # backprop
        loss.backward()
        # adjust parameters
        optimizer.step()
        # getting best loss
    return entire_epoch_preds, optimizer


def train_sequnet(x_train, y_train, x_norm, y_norm, net, flags):
    # define optimization method
    optimizer = optim.Adam(
        net.parameters(),
        lr=flags.lr,
        weight_decay=flags.weight_decay,
    )
    # Start training
    avg_loss_per_epoch = []
    avg_pearson_per_epoch = []
    for epoch in tqdm(
        range(1, flags.epochs + 1),
        dynamic_ncols=True,
        ascii=True,
    ):
        entire_epoch_pred, optimizer = train_epoch(
            x_train,
            y_train,
            x_norm,
            y_norm,
            net,
            optimizer,
            flags,
        )
        # getting average loss of this epoch
        with torch.no_grad():
            avg_loss_per_epoch.append(
                torch_metrics.calc_RMSE(entire_epoch_pred, y_train)
            )
            avg_pearson_per_epoch.append(
                torch_metrics.calc_Pearson(entire_epoch_pred, y_train)
            )
        # early stopping metrics
        if zu.early_stopping_trigger(
            avg_loss_per_epoch,
            patience=flags.early_stopping_patience,
            metric_goal="min",
        ):
            print(f"Early Stopping Triggered, Training Stopped at epoch={epoch}")
            break
        # Printing Progress Figure
        if flags.print_eval != 0 and (epoch) % flags.print_eval == 0:
            prog_print = f"Avg RMSE={avg_loss_per_epoch[-1]:.4f}; Pearson={avg_pearson_per_epoch[-1]:.4f}"
            with torch.no_grad():
                fig = zu.visual_pred_test(
                    pred_arr=entire_epoch_pred[0][0],
                    test_arr=y_train[0][0],
                    x_lab="timestamp",
                    title=f"Seq-U-Net Progress for Train Patient {flags.sel_subject}({flags.subject})\nwith epoch={epoch}; "
                    + prog_print,
                    y_lab="ABP(mmHg)",
                )
                fig.savefig(f"pic_dir/SeqUNet_epoch{epoch}_sicong.png")
            print(f"Epoch={epoch};", prog_print)
        if flags.use_wandb:
            wandb.log(
                {
                    "RMSE_progress": avg_loss_per_epoch[-1],
                    "Pearson_progress": avg_pearson_per_epoch[-1],
                }
            )
    return net


def test_sequnet(x_test, y_test, x_norm, y_norm, net, flags):
    # Test setting
    with torch.no_grad():
        preds = torch.tensor([])
        test_loss = []
        test_pearson = []
        test_mae = []
        for i in tqdm(range(0, len(x_test), flags.batch_size)):
            # getting batch
            x_batch = x_test[i : i + flags.batch_size]
            y_batch = y_test[i : i + flags.batch_size]
            if flags.mask_abp_threshold != 0 and (
                not zu.is_abp_above_threshold(
                    abp_win=y_batch.to("cpu").numpy(),
                    bp_type="sbp",
                    threshold=flags.mask_abp_threshold,
                )
            ):
                continue
            if flags.embed_noise_rate != 0:
                x_batch = zu.add_noise_2_sequence(
                    sequence=x_batch.to("cpu").numpy(),
                    noise_rate=flags.embed_noise_rate,
                )
                x_batch = torch.tensor(x_batch).float().to(flags.device)
            y_batch_preds = net(x_norm.normalize(x_batch).to(flags.device)).to("cpu")
            preds = torch.cat((preds, y_norm.denormalize(y_batch_preds)), dim=0)
            # compute loss
            loss = torch_metrics.calc_RMSE(y_norm.denormalize(y_batch_preds), y_batch)
            mae = torch_metrics.calc_MAE(y_norm.denormalize(y_batch_preds), y_batch)
            # compute Pearson's R
            pearson = torch_metrics.calc_Pearson(
                y_norm.denormalize(y_batch_preds), y_batch
            ).to(flags.device)
            # recording the loss and correlations
            test_loss.append(loss)
            test_mae.append(mae)
            test_pearson.append(pearson)
        waveform_rmse = sum(test_loss) / len(test_loss)
        waveform_mae = sum(test_mae) / len(test_mae)
        waveform_pearson = sum(test_pearson) / len(test_pearson)
    print("Started Predicting", "=".join(["=" for _ in range(35)]))
    return net, preds, waveform_rmse, waveform_mae, waveform_pearson


def reconstruct_waveform(y_test, preds):
    with torch.no_grad():
        # getting average loss of this epoch
        y_test_reshaped = y_test.to("cpu").numpy().reshape(-1)
        preds_reshaped = preds.to("cpu").numpy().reshape(-1)
    preds_reshaped_smoothed = zu.smoothing(preds_reshaped)
    start_length_to_find = 0
    end_length_to_find = len(preds_reshaped_smoothed)
    peaks_y_test, _ = find_peaks(
        y_test_reshaped[start_length_to_find:end_length_to_find], distance=50, height=80
    )
    valleys_y_test, _ = find_peaks(
        -y_test_reshaped[start_length_to_find:end_length_to_find],
        distance=50,
        height=-100,
    )
    preds_sbp = []
    for peak in peaks_y_test:
        min_st = max(0, peak - 10)
        max_ed = min(len(preds_reshaped_smoothed), peak + 10)
        preds_sbp.append(np.max(preds_reshaped_smoothed[min_st:max_ed]))
    preds_dbp = []
    for valley in valleys_y_test:
        min_st = max(0, valley - 10)
        max_ed = min(len(preds_reshaped_smoothed), valley + 10)
        preds_dbp.append(np.min(preds_reshaped_smoothed[min_st:max_ed]))

    # Visualization stuff
    result_sbp = pd.DataFrame()
    result_sbp["SBP_GT"] = y_test_reshaped[peaks_y_test]
    result_sbp["SBP_pred"] = preds_sbp
    result_dbp = pd.DataFrame()
    result_dbp["DBP_GT"] = y_test_reshaped[valleys_y_test]
    result_dbp["DBP_pred"] = preds_dbp
    # result_sbp = result_sbp[(result_sbp["SBP_GT"] > 80) & (result_sbp["SBP_GT"] < 200)]
    # result_dbp = result_dbp[(result_dbp["DBP_GT"] > 20) & (result_dbp["DBP_GT"] < 120)]
    wf_dict = {
        "abp_pred": preds_reshaped_smoothed,
        "abp_test": y_test_reshaped,
        "sbp_pred": result_sbp["SBP_pred"].to_numpy(),
        "sbp_test": result_sbp["SBP_GT"].to_numpy(),
        "dbp_pred": result_dbp["DBP_pred"].to_numpy(),
        "dbp_test": result_dbp["DBP_GT"].to_numpy(),
    }
    return wf_dict


def visualization(
    wf_dict: dict,
    waveform_rmse,
    waveform_mae,
    waveform_pearson,
    flags,
    log_dict,
) -> None:
    # declaring Visualization Class
    MV = MIMIC_Visual(
        wf_dict,
        patient_name=f"{flags.sel_subject}({flags.subject_id})",
        model_name="Seq-U-Net",
        use_wandb=flags.use_wandb,
    )
    overall_visual_dict = MV.plot_everything()

    if flags.use_wandb:
        log_dict.update(overall_visual_dict)
        log_dict.update(
            {
                "ABP_RMSE": waveform_rmse,
                "ABP_MAE": waveform_mae,
                "ABP_Pearson": waveform_pearson,
                "trained_epoch": flags.epochs,
            }
        )
        wandb.log(log_dict)
        wandb.join()
    else:
        # testing visualization
        overall_visual_dict["SBP_Bland_Altman"].savefig("plot_dir/SBP_bland_altman.png")
        overall_visual_dict["DBP_Bland_Altman"].savefig("plot_dir/DBP_bland_altman.png")
        overall_visual_dict["SBP_Confusion_Matrix"].savefig(
            "plot_dir/SBP_confusion_matrix.png"
        )
        overall_visual_dict["DBP_Confusion_Matrix"].savefig(
            "plot_dir/DBP_confusion_matrix.png"
        )
        overall_visual_dict["ABP_Waveform"].savefig("plot_dir/ABP_visual_test.png")
        overall_visual_dict["ABP_Three_Waveform"].savefig(
            "plot_dir/ABP_Three_visual_test.png"
        )
        overall_visual_dict["SBP_Waveform"].savefig("plot_dir/SBP_visual_test.png")
        overall_visual_dict["DBP_Waveform"].savefig("plot_dir/DBP_visual_test.png")
        print(f"ABP_RMSE=={waveform_rmse}")
        print(f"ABP_MAE=={waveform_mae}")
        print(f"ABP_Pearson=={waveform_pearson}")
        print(f"SBP_RMSE=={overall_visual_dict['SBP_RMSE']}")
        print(f"SBP_MAE=={overall_visual_dict['SBP_MAE']}")
        print(f"SBP_Pearson=={overall_visual_dict['SBP_Pearson']}")
        print(f"DBP_RMSE=={overall_visual_dict['DBP_RMSE']}")
        print(f"DBP_MAE=={overall_visual_dict['DBP_MAE']}")
        print(f"DBP_Pearson=={overall_visual_dict['DBP_Pearson']}")


if __name__ == "__main__":
    # getting arguments
    flags = sicong_argparse("Sequnet")
    # extracting datasets
    print(zu.pretty_progress_bar("Loading data with MIMIC_dataloader"))
    x_train, y_train, x_test, y_test, flags = zu.MIMIC_dataloader(flags)
    # normalize data
    x_norm = zu.Sicong_Norm(x_train.detach())
    y_norm = zu.Sicong_Norm(y_train.detach())
    # define the UNet Model as the backbone of the dilated CNN
    print(zu.pretty_progress_bar("Initializing SeqUNet"))
    snet = TransformerModel(
        input_size=256,
        output_size=256,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        dropout_prob=0.1,
    ).to(flags.device)
    # define the dilated CNN Model that includes the UNet backbone
    net = DilatedCNN(
        input_size=x_train.shape[-1],
        output_size=y_train.shape[-1],
        use_norm=True,
        trained_model=snet,
    ).to(flags.device)
    # Initialize wandb if used
    if flags.use_wandb:
        wandb.init(
            project=flags.wandb_project,
            reinit=True,
            tags=[
                flags.wandb_tag,
                f"{flags.sel_subject}_{flags.subject_id}",
            ],
        )
        wandb.config.update(flags)
    log_dict = {}
    print(zu.pretty_progress_bar("Training SeqUNet"))
    net = train_sequnet(x_train, y_train, x_norm, y_norm, net, flags)
    print(zu.pretty_progress_bar("Testing SeqUNet"))
    net, preds, waveform_rmse, waveform_mae, waveform_pearson = test_sequnet(
        x_test, y_test, x_norm, y_norm, net, flags
    )
    wf_dict = reconstruct_waveform(y_test, preds)
    visualization(
        wf_dict, waveform_rmse, waveform_mae, waveform_pearson, flags, log_dict
    )
    criteria = np.around(waveform_rmse.detach().cpu().numpy(), 3)
    if flags.save_result and criteria < 10:
        model_name = (
            flags.seq_result_dir
            + f"/sequnet_subject{flags.sel_subject}_rmse_{str(criteria)}.pt"
        )
        torch.save(net, model_name)
        print("saving good results")
    if flags.email_reminder:
        su.email_func(
            subject="sequnet_task_completed",
            message=f"Sequence Task Completed with Commands\n{flags}",
        )
        print(
            zu.pretty_progress_bar("Email sent, run_torch_sequnet.py Script Completed")
        )
    else:
        print(zu.pretty_progress_bar("run_torch_sequnet.py Script Completed"))
