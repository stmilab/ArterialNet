import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse, sys, os
import torch
import torch.nn as nn

from arg_parser import sicong_argparse
import seq2seq_utils as zu

sys.path.append("/home/grads/s/siconghuang/MimicModels/seq2seq_MIMIC/")
from models.arterialnet import Sequnet as SeqUNet
from models.arterialnet import DilatedCNN
import torch_metrics
from torch import optim


def rex_preprocess(train_patient_list: list, flags: argparse.Namespace):
    """
    rex_preprocess loads the lists of patients into environments

    Actions:
        1. load train and test patients via MIMIC_dataloader
        2. build environments (domains) from these lists of patients
        3. Return the dictionry of environments and pass the flags to the function

    Args:
        train_patient_list (list): list of REx patient ids in integar to be trained on
        train_patient_id (int): the test patient id in integar
    """
    envs = []
    train_min_length = np.Inf
    save_tensor = True
    for train_patient_id in train_patient_list:
        # load data
        try:
            # Load the corresponding torch tensors if they exist otherwise invoke MIMIC_dataloader
            if os.path.exists(
                f"{flags.rex_torch_path}mimic_patient_{train_patient_id}_ppg.pt"
            ):
                feat = torch.load(
                    f"{flags.rex_torch_path}mimic_patient_{train_patient_id}_ppg.pt"
                )
                label = torch.load(
                    f"{flags.rex_torch_path}mimic_patient_{train_patient_id}_abp.pt"
                )
            else:
                x_train, y_train, x_test, y_test, _ = zu.MIMIC_dataloader(
                    flags, override_sel_subject=train_patient_id
                )
                feat = torch.concat(
                    [x_train, x_test]
                )  # concatenate train and test data
                label = torch.concat(
                    [y_train, y_test]
                )  # concatenate train and test data
                torch.save(
                    feat,
                    f"{flags.rex_torch_path}mimic_patient_{train_patient_id}_ppg.pt",
                )
                torch.save(
                    label,
                    f"{flags.rex_torch_path}mimic_patient_{train_patient_id}_abp.pt",
                )
            # concat environment
            train_min_length = min(train_min_length, feat.shape[0])
            envs.append(
                {
                    "subject_id": train_patient_id,
                    "ppg": feat,
                    "abp": label,
                }
            )
        except Exception as e:
            print(
                zu.pretty_progress_bar(
                    f"{train_patient_id} failed to load with error {e}"
                )
            )
            continue
        print(
            zu.pretty_progress_bar(
                f"Completed Loading patient {train_patient_id}",
                style="@",
            )
        )
    flags.num_batches = train_min_length // flags.batch_size
    flags.train_min_length = train_min_length
    return envs, flags


def pretty_print(*values):
    """
    pretty_print pretty print progress
    """
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=3, floatmode="fixed")
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


def make_new_sequnet(flags: argparse.Namespace):
    snet = SeqUNet(
        num_inputs=1,
        num_channels=[1024, 512, 256, 128, 64],
        num_outputs=1,
    ).to(flags.device)
    # define the dilated CNN Model that includes the UNet backbone
    net = DilatedCNN(
        input_size=flags.num_prev_cardiac_cycle_feature * 50,
        output_size=flags.num_prev_cardiac_cycle_label * 50,
        use_norm=True,
        trained_model=snet,
    ).to(flags.device)
    return net


def fit_model(envs: dict, x_norm, y_norm, flags: argparse.Namespace):
    """
    fit_model _summary_

    Actions:
    1. Declare the variables for progress evaluation
    2.


    Args:
        envs (dict): _description_
        flags (argparse.Namespace): _description_
    """
    # Decalre the variables for progress evaluation
    all_train_nlls = -1 * np.ones((flags.n_restarts, flags.epochs))
    all_train_accs = -1 * np.ones((flags.n_restarts, flags.epochs))
    all_train_pearsons = -1 * np.ones((flags.n_restarts, flags.epochs))
    # all_train_penalties = -1*np.ones((flags.n_restarts, flags.epochs))
    all_irmv1_penalties = -1 * np.ones((flags.n_restarts, flags.epochs))
    all_rex_penalties = -1 * np.ones((flags.n_restarts, flags.epochs))
    all_test_accs = -1 * np.ones((flags.n_restarts, flags.epochs))
    all_test_pearsons = -1 * np.ones((flags.n_restarts, flags.epochs))
    final_train_accs = []
    final_train_pearsons = []
    final_test_accs = []
    final_test_pearsons = []
    best_test_accs = []
    best_model = None
    best_loss = 0.0

    # Swapping Notebook Progress Bar
    if flags.jupyter_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    for restart in range(flags.n_restarts):
        best_test_acc = 0.0
        model = make_new_sequnet(flags)
        # if flags.use_wandb:
        #     wandb.watch(model)
        optimizer = optim.Adam(
            model.parameters(),
            lr=flags.lr,
            weight_decay=flags.weight_decay,
        )
        print("Restart", restart + 1)
        i = 0
        test_index = len(envs) - 1
        steps_enumerator = np.arange(flags.epochs)
        if flags.print_eval_intervals:
            pretty_print(
                "step",
                "train nll",
                "train RMSE",
                "train Pearson",
                "rex penalty",
                "irmv1 penalty",
                "validate RMSE",
                "validate Pearson",
            )
        else:
            steps_enumerator = tqdm(steps_enumerator)
        for step in steps_enumerator:
            n = i % flags.num_batches
            val_batch_idx = np.arange(flags.train_min_length)
            np.random.shuffle(val_batch_idx)
            for edx, env in enumerate(envs):
                if edx != test_index:  # if not last env, then train, otherwise validate
                    st = n * flags.batch_size
                    ed = (n + 1) * flags.batch_size
                    norm_logits = model(
                        x_norm.normalize(env["ppg"][st:ed]).to(flags.device)
                    ).cpu()
                    logits = y_norm.denormalize(norm_logits)
                    env["nll"] = torch_metrics.mean_nll(
                        logits,
                        env["abp"][st:ed],
                    )
                    env["acc"] = torch_metrics.calc_RMSE(
                        logits,
                        env["abp"][st:ed],
                    )
                    env["penalty"] = torch_metrics.penalty(
                        logits,
                        env["abp"][st:ed],
                    )
                    env["pearson_r"] = torch_metrics.calc_Pearson(
                        logits,
                        env["abp"][st:ed],
                    )
                else:  # validate environment only randomly evaluate 4496 batches due to limited memory
                    vali_idx = val_batch_idx[:100]
                    norm_logits = model(
                        x_norm.normalize(env["ppg"][vali_idx]).to(flags.device)
                    ).cpu()
                    logits = y_norm.denormalize(norm_logits)
                    env["nll"] = torch_metrics.mean_nll(logits, env["abp"][vali_idx])
                    env["acc"] = torch_metrics.calc_RMSE(logits, env["abp"][vali_idx])
                    env["penalty"] = torch_metrics.penalty(logits, env["abp"][vali_idx])
                    env["pearson_r"] = torch_metrics.calc_Pearson(
                        logits, env["abp"][vali_idx]
                    )
            i += 1
            train_nll = torch.stack(
                [env["nll"] for edx, env in enumerate(envs) if edx != test_index]
            ).mean()
            train_acc = torch.stack(
                [env["acc"] for edx, env in enumerate(envs) if edx != test_index]
            ).mean()
            irmv1_penalty = torch.stack(
                [env["penalty"] for edx, env in enumerate(envs) if edx != test_index]
            ).mean()
            train_pearson = torch.stack(
                [env["pearson_r"] for edx, env in enumerate(envs) if edx != test_index]
            ).mean()
            weight_norm = torch.tensor(0.0)
            for w in model.parameters():
                weight_norm += w.cpu().norm().pow(2)

            loss_list = [
                env["nll"] for edx, env in enumerate(envs) if edx != test_index
            ]

            if flags.early_loss_mean:
                loss_list = [loss_unit.mean() for loss_unit in loss_list]

            loss = 0.0
            loss += flags.erm_amount * sum(loss_list)

            loss += flags.l2_regularizer_weight * weight_norm

            penalty_weight = (
                flags.penalty_weight if step >= flags.penalty_anneal_iters else 1.0
            )

            rex_penalty = torch_metrics.rex_calc(loss_list, test_index, flags.mse)
            if flags.rex:
                loss += penalty_weight * rex_penalty
            else:
                loss += penalty_weight * irmv1_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_acc = envs[test_index]["acc"]
            test_pearson = envs[test_index]["pearson_r"]

            if step % flags.eval_interval == 0:
                train_acc_scalar = train_acc.detach().numpy()
                test_acc_scalar = test_acc.detach().numpy()
                train_pearson_scalar = train_pearson.detach().numpy()
                test_pearson_scalar = test_pearson.detach().numpy()
                if flags.print_eval_intervals:
                    pretty_print(
                        np.int32(step),
                        train_nll.detach().numpy(),
                        train_acc.detach().numpy(),
                        train_pearson.detach().numpy(),
                        rex_penalty.detach().numpy(),
                        irmv1_penalty.detach().numpy(),
                        test_acc.detach().numpy(),
                        test_pearson.detach().numpy(),
                    )
                if flags.make_gif:
                    prog_print = f"Avg RMSE={test_acc_scalar:.4f}; Pearson={test_pearson_scalar:.4f}"
                    with torch.no_grad():
                        fig = zu.visual_pred_test(
                            pred_arr=model(
                                envs[test_index]["ppg"][:50].to(flags.device)
                            )
                            .cpu()
                            .numpy()
                            .reshape(-1),
                            test_arr=envs[test_index]["abp"][:50].numpy().reshape(-1),
                            x_lab="timestamp",
                            title=f"MIMIC REx Progress for Train Participant\n({envs[test_index]['subject_id']})\nwith epoch={step}; "
                            + prog_print,
                            y_lab="SBP(mmHg)",
                        )
                        fig.savefig(f"plot_dir/bioz_REx_epoch{step}_sicong.png")
                    print(f"Epoch={step};", prog_print)
                if (train_acc_scalar <= test_acc_scalar) and (
                    test_acc_scalar < best_test_acc
                ):
                    best_test_acc = test_acc_scalar

            all_train_nlls[restart, step] = train_nll.detach().numpy()
            all_train_accs[restart, step] = train_acc.detach().numpy()
            all_train_pearsons[restart, step] = train_pearson.detach().numpy()
            all_rex_penalties[restart, step] = rex_penalty.detach().numpy()
            all_irmv1_penalties[restart, step] = irmv1_penalty.detach().numpy()
            all_test_accs[restart, step] = test_acc.detach().numpy()
            all_test_pearsons[restart, step] = test_pearson.detach().numpy()
        final_train_accs.append(train_acc.detach().numpy())
        final_test_accs.append(test_acc.detach().numpy())
        final_train_pearsons.append(train_pearson.detach().numpy())
        final_test_pearsons.append(test_pearson.detach().numpy())
        best_test_accs.append(best_test_acc)
        if __name__ == "__main__":
            print("best test RMSE this run:", best_test_acc)
            print("Final train RMSE (mean/std across restarts so far):")
            print(np.mean(final_train_accs), np.std(final_train_accs))
            print("Final test RMSE (mean/std across restarts so far):")
            print(np.mean(final_test_accs), np.std(final_test_accs))
            print("best test RMSE (mean/std across restarts so far):")
            print(np.mean(best_test_accs), np.std(best_test_accs))
        if final_test_accs[-1] >= best_loss:
            best_model = model
            best_loss = final_test_accs[-1]
    print(f"best model with RMSE={best_loss} (end of function)")
    return (
        best_model,
        all_train_accs,
        all_train_pearsons,
        all_test_accs,
        all_test_pearsons,
        all_train_nlls,
        all_irmv1_penalties,
        all_rex_penalties,
    )


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


def rex_normalizer(envs):
    fea_max = -np.Inf
    fea_min = np.Inf
    lab_max = -np.Inf
    lab_min = np.Inf
    for env in envs:
        fea_max = max(fea_max, env["ppg"].max())
        fea_min = min(fea_min, env["ppg"].min())
        lab_max = max(lab_max, env["abp"].max())
        lab_min = min(lab_min, env["abp"].min())
    x_norm = Sicong_Norm(min_val=fea_min, max_val=fea_max)
    y_norm = Sicong_Norm(min_val=lab_min, max_val=lab_max)
    return x_norm, y_norm


if __name__ == "__main__":
    flags = sicong_argparse("Sequnet")
    train_patient_str_list = zu.get_mimic_subject_lists(flags=flags)
    train_patient_list = [zu.get_subject_id_from_str(x) for x in train_patient_str_list]
    # train_patient_list = [82574, 47874, 12175, 27172]
    envs, flags = rex_preprocess(train_patient_list, flags)

    flags.n_restarts = 3
    flags.print_eval_intervals = True
    flags.early_loss_mean = True
    flags.erm_amount = 1.0
    flags.l2_regularizer_weight = 0.001
    flags.penalty_anneal_iters = 100
    flags.penalty_weight = 10000.0
    flags.mse = True
    flags.rex = True
    flags.eval_interval = 10
    flags.make_gif = False
    flags.epochs = 100
    flags.lr = 0.005
    x_norm, y_norm = rex_normalizer(envs)
    (
        best_model,
        all_train_accs,
        all_train_pearsons,
        all_test_accs,
        all_test_pearsons,
        all_train_nlls,
        all_irmv1_penalties,
        all_rex_penalties,
    ) = fit_model(envs, x_norm, y_norm, flags)
    print(zu.pretty_progress_bar("Done"))
