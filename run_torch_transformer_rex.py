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
from scipy.stats import pearsonr
import warnings

# importing my own utility functions
sys.path.append("utils/")
sys.path.append("models/")
from utils.arg_parser import sicong_argparse
from utils.visual_combine import MIMIC_Visual
import utils.seq2seq_utils as zu
from models.arterialnet import Sequnet as SeqUNet
from models.arterialnet import DilatedCNN
from models.arterialnet import TransformerModel
from utils import torch_metrics, rex_utils as ru
import run_torch_transformer as base_sequnet


warnings.filterwarnings("ignore")


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
    train_patient_list = [82574, 47874, 27172]
    envs, flags = ru.rex_preprocess(train_patient_list, flags)
    # End of REx pretraining
    x_norm, y_norm = ru.rex_normalizer(envs)
    # Initialize wandb if used
    if flags.use_wandb:
        wandb.init(
            project=flags.wandb_project,
            reinit=True,
            tags=[flags.wandb_tag, f"{flags.sel_subject}_{flags.subject}"],
        )
        wandb.config.update(flags)
    log_dict = {}
    (
        best_model,
        all_train_accs,
        all_train_pearsons,
        all_test_accs,
        all_test_pearsons,
        all_train_nlls,
        all_irmv1_penalties,
        all_rex_penalties,
    ) = ru.fit_model(envs, x_norm, y_norm, flags)
    print(zu.pretty_progress_bar("Done"))
    print(zu.pretty_progress_bar("Training SeqUNet"))
    net = base_sequnet.train_sequnet(
        x_train,
        y_train,
        x_norm,
        y_norm,
        best_model,
        flags,
    )
    print(zu.pretty_progress_bar("Testing SeqUNet"))
    (
        net,
        preds,
        waveform_rmse,
        waveform_mae,
        waveform_pearson,
    ) = base_sequnet.test_sequnet(x_test, y_test, x_norm, y_norm, net, flags)
    wf_dict = base_sequnet.reconstruct_waveform(y_test, preds)
    base_sequnet.visualization(
        wf_dict, waveform_rmse, waveform_mae, waveform_pearson, flags, log_dict
    )
    criteria = np.around(waveform_rmse.detach().cpu().numpy(), 3)
    if flags.save_result and criteria < 10:
        model_name = (
            flags.seq_result_dir
            + f"/sequnet_subject{flags.subject}_rmse_{criteria:.3f}.pt"
        )
        torch.save(net, model_name)
        print("saving good results")

    print(zu.pretty_progress_bar("run_torch_sequnet.py Script Completed"))
