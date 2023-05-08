import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from torchmetrics.functional import f1_score
from scipy.stats import pearsonr
import pdb

# define loss functions
def mean_accuracy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    mean_accuracy calculates mean accuracy between 2 torch tensors

    Args:
        logits (torch.Tensor): predicted tensor
        y (torch.Tensor): ground truth tensor

    Returns:
        torch.Tensor: computed mean accuracy in torch tensor format
    """
    preds = (logits > 0.0).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def calc_f1_score(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    calc_f1_score compute F1_Score between 2 torch tensors

    Args:
        logits (torch.Tensor): predicted tensor
        y (torch.Tensor): ground truth tensor

    Returns:
        torch.Tensor: computed F1 Score in torch tensor format
    """
    preds = (logits > 0.0).int()
    return f1_score(preds, y.int(), num_classes=2)


def calc_RMSE(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    calc_RMSE compute RMSE between 2 torch tensors

    Args:
        logits (torch.Tensor): predicted tensor
        y (torch.Tensor): ground truth tensor

    Returns:
        torch.Tensor: computed RMSE in torch tensor format
    """
    return torch.sqrt(F.mse_loss(logits, y))


def calc_MAE(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    calc_MAE compute MAE between 2 torch tensors

    Args:
        logits (torch.Tensor): predicted tensor
        y (torch.Tensor): ground truth tensor

    Returns:
        torch.Tensor: computed MAE value in torch tensor format
    """
    return F.l1_loss(logits, y)


def calc_Pearson(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    calc_Pearson compute Pearson's Correlation between 2 torch tensors

    Args:
        logits (torch.Tensor): predicted tensor
        y (torch.Tensor): ground truth tensor

    Returns:
        torch.Tensor: computed Pearson's R in torch tensor format
    """
    r_val, p_val = pearsonr(
        logits.detach().numpy().reshape(-1), y.detach().numpy().reshape(-1)
    )
    return torch.tensor(r_val).float()


def mean_nll(logits, y):
    """
    mean_nll compute the mean negative likelihood between 2 torch tensors

    Args:
        logits (_type_): predicted tensor
        y (_type_): ground truth tensor

    Returns:
        _type_: computed mean negative loglikelihood in torch tensor format
    """
    # return F.binary_cross_entropy_with_logits(logits, y)
    return F.mse_loss(logits, y)


def penalty(logits: torch.Tensor, y: torch.Tensor, use_cuda=True):
    """
    penalty compute penalty value between 2 torch tensors

    Args:
        logits (torch.Tensor): predicted tensor
        y (torch.Tensor): ground truth tensor
        use_cuda (bool): Whether to make it cuda

    Returns:
        _type_: computed penalty in torch tensor format
    """
    scale = torch.tensor(1.0).cpu().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def rex_calc(loss_list: list, flags_test_index: int, flag_mse: bool) -> float:
    """
    rex_calc Calculate risk extrapolation penalty

    Args:
        loss_list (list): list of env/domain losses
        flags_test_index (int): index of test index
        flag_mse (bool): whether to calculate mean squared error

    Returns:
        rex_pen (float): calculated risk extrapolation penalty
    """
    rex_pen = 0
    for edx1 in range(len(loss_list[:])):
        for edx2 in range(len(loss_list[edx1:])):
            if edx1 != edx2 and edx1 != flags_test_index and edx2 != flags_test_index:
                if flag_mse:
                    rex_pen += (loss_list[edx1].mean() - loss_list[edx2].mean()) ** 2
                else:
                    rex_pen += (loss_list[edx1].mean() - loss_list[edx2].mean()).abs()
    return torch.tensor(rex_pen).float()
