import numpy as np
import torch
import pdb


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def add_gradient_to_waveform(waveform: torch.Tensor):
    """
    add_gradient_to_waveform Adding the gradient to the input waveform

    - If input of size [100, 1, 50], then output [100, 3, 50]
    - If input of size [100, 2, 50], then output [100, 6, 50]
    - First 2 channels are original waveform
    - Middle 2 channels are first gradient
    - Last 2 channels are second gradient

    Args:
        waveform (torch.Tensor): Input waveform

    Returns:
        torch.Tensor: Gradient waveform
    """
    with torch.no_grad():
        x_grad = torch.gradient(waveform, dim=-1)[0]
        x_grad_2 = torch.gradient(x_grad, dim=-1)[0]
        return torch.cat((waveform, x_grad, x_grad_2), dim=1)


def get_maximum_slope_from_waveform(waveform, sec=1.5, freq=125) -> np.ndarray:
    """
    get_maximum_slope_from_waveform calculate maximum slope points from ppg waveform

    This function automatically calculates the maximum slope points from the waveform based on given frequency
    and window length, where sec parameter denotes a plausible window of focus, freq parameter provides the frequncy
    of the waveform


    Args:
        waveform (np.ndarray): waveform array of interest.
        sec (float, optional): defines a plausible roof of window interest (based on respiration rate). Defaults to 1.5.
        freq (int, optional): frequency of the waveform in hertz(Hz). Defaults to 125.

    Returns:
        np.ndarray: array of maximum slopes identified from the PPG waveform
    """
    # formulating a windowsize based on time
    win_size = int(freq * sec)
    olap = int(0.5 * win_size)
    ms_ind = []
    grad = np.gradient(waveform)
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


def calculate_morphology(t_dv: np.ndarray, ms_t: np.ndarray, waveform: np.ndarray):
    """
    calculate_morphology calculates morphology with the given pulsatile waveform

    Args:
        t_dv (np.ndarray): time-series array of waveform (in seconds)
        ms_t (np.ndarray): timestamps of maximum slope points (in seconds)
        waveform (np.ndarray): waveform segment of interest

    Returns:
        list: list of morphology parameters of the given waveform segment
    """
    # You can swap or add to your own morphologies or other features
    ms_amp = np.asarray([find_nearest(t_dv, t) for t in ms_t])
    zc_ch1 = np.where(np.diff(np.sign(np.gradient(waveform))) <= -1)[0]
    zc_ch1_valleys = np.where(np.diff(np.sign(np.gradient(waveform))) >= 1)[0]
    zc_2d_ch1_valleys = np.where(
        np.diff(np.sign(np.gradient(np.gradient(waveform)))) == 2
    )[0]

    return np.array(
        [
            waveform.min(),
            waveform.max(),
            waveform.mean(),
            waveform.std(),
            ms_amp,
            zc_ch1,
            zc_ch1_valleys,
            zc_2d_ch1_valleys,
        ]
    ).reshape(-1)


def extract_morphology_from_waveform(waveform: torch.Tensor):
    """
    extract_morphology_from_waveform Extracts morphologies of each segment of waveform


    Args:
        waveform (torch.Tensor): input waveform

    Returns:
        np.ndarray: array of morphologies of waveform
    """
    with torch.no_grad():
        np_waveform = waveform.cpu().numpy()
    morphology_list = []
    for i in range(np_waveform.shape[0]):
        t_dv = np.arange(0, np_waveform.shape[-1] / 125, 1 / 125)
        ms_ind = get_maximum_slope_from_waveform(waveform=np_waveform[i, 0, :])
        ms_t = t_dv[ms_ind]
        ms_t = []
        # Calculate morphology
        morphology_list.append(
            calculate_morphology(t_dv, ms_t, waveform=np_waveform[i, 0, :])
        )
    return np.array(morphology_list)


def main():
    import pdb

    print("This is morph_utils.py, only used for testing!!!")
    # X_data is a 3D torch tensor array of [batch, waveform_channel, waveform_length]
    x_data = torch.load(
        "/home/grads/s/siconghuang/REx_candidate_torch_tensors/mimic_patient_99659_ppg.pt"
    )
    # y_data is a 3D torch tensor array of [batch, waveform_channel, waveform_length]
    y_data = torch.load(
        "/home/grads/s/siconghuang/REx_candidate_torch_tensors/mimic_patient_99659_abp.pt"
    )
    # Only using first 32 samples to test arterialnet dimension
    x_data = x_data[:128, :, :]
    y_data = y_data[:128, :, :]

    morph_data = torch.from_numpy(extract_morphology_from_waveform(x_data)).float()
    x_data = add_gradient_to_waveform(x_data)
    # Loading the Model and testing the dimension
    import sys, os

    sys.path.insert(0, os.path.abspath(".."))
    from models.transformer_model import TransformerModel
    from models.arterialnet import ArterialNet

    anet = ArterialNet(
        input_size=x_data.shape[2],
        num_channels=x_data.shape[1],
        output_size=y_data.shape[2],
        use_norm="instance",
        add_morph=True,
        trained_model=TransformerModel(),
    )
    pred = anet(x_data, morph=morph_data)
    print("shape matched") if pred.shape == y_data.shape else print("Shape not matched")
    pdb.set_trace()


if __name__ == "__main__":
    # Test the model
    main()
