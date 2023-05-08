import seq2seq_utils as zu
import sys
import numpy as np
from pprint import pprint
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pdb


def form_dict(bp_key, fig, rmse, mae, mean, std, rval, pval):
    """form_dict
        Make dictionary from bland_altman and other statistics

    Arguments:
        bp_key {str} -- whether ABP, SBP, or DBP, used to make key for dictionary
        fig {pyplot_figure} -- bland altman plot
        rmse {float} -- Rooted Mean Squared Error, the lower the better
        mae {float} -- Mean Absolute Error, the lower the better
        rval {float} -- Pearon's R value, between 0~1, the higher the better

    Returns:
        {dictionary} -- dictionary containing figure and metrics of given bp_key
    """
    return {
        f"{bp_key}_RMSE": rmse,
        f"{bp_key}_MAE": mae,
        f"{bp_key}_MEAN": mean,
        f"{bp_key}_STD": std,
        f"{bp_key}_Pearson": rval,
        f"{bp_key}_Bland_Altman": fig,
    }


# Bland Altman
def plot_bland_altman(
    pred_arr, test_arr, BP_type="SBP", title="Default Title", return_plt=True
):
    """plot_bland_altman
        Plotting Bland Altman and Computing Metrics for given BP_type

    Arguments:
        pred_arr {list} -- Predicted waveform sequence of given BP_type
        test_arr {list} -- Ground Truth waveform sequence of given BP_type

    Keyword Arguments:
        BP_type {str} -- Choice of whether SBP (systolic) or DBP (diastolic) Blood Pressure (default: {"SBP"})
        title {str} -- Custom title for the given plot, usually containing info about model
                            and patient involved in the given prediction (default: {"Default Title"})
        return_plt {bool} -- whether to return the ploting component (default: {True})

    Returns:
        {list} -- plotted bland altman plot followed by metrics calculated (RMSE, MAE, and R)
    """
    rmse, mae, mean, std, rval, pval = zu.calc_metrics(pred_arr, test_arr)
    plt.style.use("fivethirtyeight")
    # print(f"{BP_type} RMSE == {rmse:.3f}")
    # print(f"{BP_type} Pearson == {rval:.3f}; p_val == {pval:.3f}")
    title += f"\n RMSE={rmse:.3f};Pearson={rval:.3f};"
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), dpi=80)
    if BP_type == "SBP":

        ######################################################
        ######################## Plot 1 ######################
        ######################################################
        f = ax[0]
        f.set_title(title)
        f.set_ylabel("Estimated SBP (mmHg)", fontsize=10)
        f.set_xlabel("Reference SBP (mmHg)", fontsize=10)
        f.scatter(test_arr, pred_arr, alpha=0.33)
        z = np.polyfit(test_arr, pred_arr, 1)
        p = np.poly1d(z)
        w = [50, 230]
        f.plot(w, p(w), "k--")
        f.set_ylim(w)
        f.set_xlim(w)
        f.tick_params(axis="x", labelsize=10)
        f.tick_params(axis="y", labelsize=10)
        ######################################################
        ######################## Plot 2 ######################
        ######################################################
        f = ax[1]
        f.set_ylabel("Systolic BP Error (mmHg)", fontsize=10)
        f.set_xlabel("Reference Systolic BP (mmHg)", fontsize=10)
        x = test_arr
        y = np.asarray(test_arr) - np.asarray(pred_arr)
        f.scatter(x, y, alpha=0.33)
        bias = y.mean()
        st = y.std()
        l_loa = bias - 1.96 * st
        h_loa = bias + 1.96 * st
        f.axhline(y=bias, ls="-.", color="k", alpha=0.5)
        f.axhline(y=l_loa, ls="-.", color="r", alpha=0.5)
        f.axhline(y=h_loa, ls="-.", color="r", alpha=0.5)
        f.text(72, h_loa + 5, f"{h_loa:.2f}".rjust(30), color="r", fontsize=12)
        f.text(72, bias + 3, f"{bias:.2f}".rjust(30), color="k", fontsize=12)
        f.text(72, l_loa - 5, f"{l_loa:.2f}".rjust(30), color="r", fontsize=12)
        f.set_ylim((-75, 75))
        f.set_xlim(w)
        f.tick_params(axis="x", labelsize=10)
        f.tick_params(axis="y", labelsize=10)

    elif BP_type == "DBP":
        ######################################################
        ######################## Plot 1 ######################
        ######################################################
        f = ax[0]
        f.set_title(title)
        f.set_ylabel("Estimated DBP (mmHg)", fontsize=10)
        f.set_xlabel("Reference DBP (mmHg)", fontsize=10)
        f.scatter(test_arr, pred_arr, alpha=0.33)
        z = np.polyfit(test_arr, pred_arr, 1)
        p = np.poly1d(z)
        w = [10, 140]
        f.plot(w, p(w), "k--")
        f.set_ylim(w)
        f.set_xlim(w)
        f.tick_params(axis="x", labelsize=10)
        f.tick_params(axis="y", labelsize=10)
        ######################################################
        ######################## Plot 2 ######################
        ######################################################
        f = ax[1]
        f.set_ylabel("Diastolic BP Error (mmHg)", fontsize=10)
        f.set_xlabel("Reference Diastolic BP (mmHg)", fontsize=10)
        x = test_arr
        y = np.asarray(test_arr) - np.asarray(pred_arr)
        f.scatter(x, y, alpha=0.33)
        bias = y.mean()
        st = y.std()
        l_loa = bias - 1.96 * st
        h_loa = bias + 1.96 * st
        f.axhline(y=bias, ls="-.", color="k", alpha=0.5)
        f.axhline(y=l_loa, ls="-.", color="r", alpha=0.5)
        f.axhline(y=h_loa, ls="-.", color="r", alpha=0.5)
        f.text(10, h_loa + 5, f"{h_loa:.2f}".rjust(30), color="r", fontsize=12)
        f.text(10, bias + 3, f"{bias:.2f}".rjust(30), color="k", fontsize=12)
        f.text(10, l_loa - 5, f"{l_loa:.2f}".rjust(30), color="r", fontsize=12)
        f.set_ylim((-65, 65))
        f.set_xlim(w)
        f.tick_params(axis="x", labelsize=10)
        f.tick_params(axis="y", labelsize=10)

    plt.close()
    # print(f"mean={bias} and std={st}")
    if not return_plt:
        plt.show()
    return fig, rmse, mae, mean, std, rval, pval


# Plot waveform
def plot_waveform(
    pred_arr,
    test_arr,
    freq,
    title="default title",
    x_lab="default_x",
    y_lab="default_y",
    plot_style=".-.",
    add_metrics=False,
):
    """plot_waveform
        Plotting a comparison plot between pred and test

    Arguments:
        pred_arr {list} -- Predicted waveform sequence of given BP_type
        test_arr {list} -- Ground Truth waveform sequence of given BP_type
        freq {int} -- frequency of the device (in Hz, MIMIC ABP has 125 Hz),
                            setting 0 then X-label will be in sample domain, otherwise in time domain

    Keyword Arguments:
        title {str} -- Custom title for the given plot, usually containing info
                            about model and patient involved in the given prediction (default: {"Default Title"})
        x_lab {str} -- name of X label (default: {"default_x"})
        y_lab {str} -- name of y label (default: {"default_y"})
        plot_style {str} -- Style of the line or dot (default: {".-."})
        add_metrics {bool} -- Whether to include metrics info on the plot
                                (metrics include RMSE, MAE, and R) (default: {False})

    Returns:
        _type_ -- _description_
    """
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.set_title(title)
    x_axis = np.arange(len(pred_arr)) / freq
    ax.plot(x_axis, pred_arr, plot_style, label="pred", lw=2, alpha=0.6)
    ax.plot(x_axis, test_arr, plot_style, label="test", lw=2, alpha=0.6)
    ax.set_ylim(20, 180)
    if add_metrics:
        rmse, mae, mean, std, rval, pval = zu.calc_metrics(pred=pred_arr, test=test_arr)
        local_metrics = [
            f"RMSE={rmse:.3f}",
            f"MAE={mae:.3f}",
            f"Pearson={rval:.3f}",
        ]
        for i, t in enumerate(local_metrics):
            x_pos = int(x_axis[-1] * 0.95)
            y_pos = 60 - (i * 5)
            ax.text(x_pos, y_pos, t, color="k", fontsize=10)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.legend(loc=1)
    fig.set_size_inches(10, 7)
    fig.set_dpi(75)
    plt.close()
    return fig, ax


def plot_three_waveform(
    pred_arr_list,
    test_arr_list,
    freq,
    title="default title",
    x_lab="default_x",
    y_lab="default_y",
    plot_style=".-.",
    add_metrics=False,
):
    """plot_three_waveforms
        Plotting 3 plots between pred and test

    Arguments:
        pred_arr_list {list} -- lists of Predicted waveform sequence of given BP_type
        test_arr_list {list} -- lists of Ground Truth waveform sequence of given BP_type
        freq {int} -- frequency of the device (in Hz, MIMIC ABP has 125 Hz),
                            setting 0 then X-label will be in sample domain, otherwise in time domain

    Keyword Arguments:
        title {str} -- Custom title for the given plot, usually containing info
                            about model and patient involved in the given prediction (default: {"Default Title"})
        x_lab {str} -- name of X label (default: {"default_x"})
        y_lab {str} -- name of y label (default: {"default_y"})
        plot_style {str} -- Style of the line or dot (default: {".-."})
        add_metrics {bool} -- Whether to include metrics info on the plot
                                (metrics include RMSE, MAE, and R) (default: {False})

    Returns:
        fig -- pyplot figure
        ax  -- pyplot axis
    """
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(len(pred_arr_list), 1, sharex=True, sharey=True)
    fig.suptitle(title, fontdict={"weight": "black"})
    for i, pred_arr in enumerate(pred_arr_list):
        x_axis = np.arange(len(pred_arr)) / freq
        ax[i].plot(x_axis, pred_arr, plot_style, label="pred", lw=1.5, alpha=0.6)
        ax[i].plot(
            x_axis, test_arr_list[i], plot_style, label="test", lw=1.5, alpha=0.6
        )
        ax[i].set_ylim(20, 180)
        if add_metrics:
            rmse, mae, mean, std, rval, pval = zu.calc_metrics(
                pred=pred_arr, test=test_arr_list[i]
            )
            local_metrics = [
                f"RMSE={rmse:.3f}",
                f"MAE={mae:.3f}",
                f"Pearson={rval:.3f}",
                f"p-value={pval:.3f}",
                f"mean={mean:.3f}",
                f"std={std:.3f}",
            ]
            for j, t in enumerate(local_metrics):
                x_pos = 0 - x_axis[int(len(x_axis) * 0.055)]
                y_pos = 180 - (j * 15)
                ax[i].text(
                    x_pos,
                    y_pos,
                    t,
                    color="k",
                    fontdict={
                        "size": 10,
                        "family": "monospace",
                    },
                )
    fig.supxlabel(x_lab)
    fig.supylabel(y_lab)
    ax[0].legend(loc=1)
    fig.set_size_inches(18, 9)
    fig.set_dpi(75)
    plt.close()
    return fig, ax


# Confusion Matrix Setup
def confusion_matrix_of_stages(pred_arr, test_arr, bp_type="sbp", pname=None):
    """confusion_matrix_of_stages
        Plotting Confusion Matrix of Accuracy by Hypertension Stages for the given BP_type

    Arguments:
        pred_arr {list} -- Predicted waveform sequence of given BP_type
        test_arr {list} -- Ground Truth waveform sequence of given BP_type

    Keyword Arguments:
        BP_type {str} -- Choice of whether SBP (systolic) or DBP (diastolic) Blood Pressure (default: {"sbp"})
        pname {str} -- Include patient info in title, if None then don't list the patient info (default: {None})

    Returns:
        pyplot_figure -- Confusion Matrix Pyplot Figure
    """
    bp_type = bp_type.upper()

    def cf_in_percent(data):
        def digit2percent(digit):
            """digit2percent
                Converting floating value into pertage string
            Arguments:
                digit {float} -- portion in percent (from 0 to 1)

            Returns:
                str -- formulated string of percentage
            """
            return str(round(digit * 100, 1)) + "%"

        new_data = []
        row_sum = np.array([np.sum(data[i, :]) for i in range(len(data))])
        col_sum = np.array([np.sum(data[:, i]) for i in range(len(data[0]))])
        for i in range(len(data)):
            tmp = []
            for j in range(len(data[i])):
                if data[i][j] == 0:
                    tmp.append("N/A")
                else:
                    tmp.append(digit2percent(data[i][j] / row_sum[i]))
            new_data.append(tmp)
        return new_data

    sbp_stage_arr = [120, 130, 140, 180]
    sbp_stage_txt = [
        "normal\n<120",
        "Elevated\n120~130",
        "Stage_1\n130~140",
        "Stage_2\n140~180",
        "Crisis\n>180",
    ]
    dbp_stage_arr = [80, 90, 120]
    dbp_stage_txt = ["normal\n<80", "Stage_1\n80~89", "Stage_2\n90~120", "Crisis\n>120"]
    if bp_type == "SBP":
        bp_stage_arr = sbp_stage_arr
        bp_stage_txt = sbp_stage_txt
    elif bp_type == "DBP":
        bp_stage_arr = dbp_stage_arr
        bp_stage_txt = dbp_stage_txt
    else:
        print("wrong selection")
    digit_pred_arr = np.digitize(pred_arr, bp_stage_arr)
    digit_test_arr = np.digitize(test_arr, bp_stage_arr)
    cf = confusion_matrix(digit_test_arr, digit_pred_arr)
    cf_percent = cf_in_percent(cf)
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots(1, 2, figsize=(25, 10), dpi=70)
    if pname is None:
        fig.suptitle(f"Confusion Matrix of {bp_type} Prediction Accuracy by Stages")
    else:
        fig.suptitle(
            f"Confusion Matrix of {bp_type} Prediction Accuracy by Stages\nfor Patient{pname}"
        )
    sns.heatmap(
        cf,
        annot=True,
        xticklabels=bp_stage_txt,
        yticklabels=bp_stage_txt,
        cmap="mako_r",
        square=True,
        fmt="g",
        ax=ax[0],
    )
    ax[0].set_xlabel("Predicted (mmHg)")
    ax[0].set_ylabel("Reference (mmHg)")
    ax[0].set_yticklabels(bp_stage_txt, rotation=90)

    sns.heatmap(
        cf,
        annot=cf_percent,
        xticklabels=bp_stage_txt,
        yticklabels=bp_stage_txt,
        cmap="mako_r",
        square=True,
        fmt="",
        ax=ax[1],
    )
    ax[1].set_xlabel("Predicted (mmHg)")
    ax[1].set_ylabel("Reference (mmHg)")
    ax[1].set_yticklabels(bp_stage_txt, rotation=90)
    plt.close()
    return fig


class MIMIC_Visual:
    """MIMIC_Visual
    A Visualization class dedicated to provide figures and metrics for MIMIC Arterial Blood Pressure waveform Prediction Tasks
    """

    def __init__(
        self,
        waveform_dict,
        patient_name=None,
        model_name=None,
        sampled_freq=125,
        use_wandb=False,
    ):
        """__init__ Initialize the MIMIC_Visual class with
            waveform data; patient information, and model information

        Arguments:
            waveform_dict {dict} -- Dictionary of both predicted and ground truth waveform of
                Arterial Blood Pressure and its extracted sequence of Systolic and Diastolic Blood Pressures

        Keyword Arguments:
            patient_name {str} -- Name of the MIMIC Patient which the waveform belong to (default: {None})
            model_name {str} -- Name/Type of the Model that produced the predicted waveform (default: {None})
            sampled_freq {int} -- Sampling frequency of the waveform in Hertz, MIMIC III default is 125Hz (default: {125})
            use_wandb {bool} -- Whether to produce Weights & Bias Image or default Pyplot Figure (default: {False})

        """
        self.pred_dict = {
            "ABP": waveform_dict["abp_pred"],
            "SBP": waveform_dict["sbp_pred"],
            "DBP": waveform_dict["dbp_pred"],
        }
        self.test_dict = {
            "ABP": waveform_dict["abp_test"],
            "SBP": waveform_dict["sbp_test"],
            "DBP": waveform_dict["dbp_test"],
        }
        self.patient_name = patient_name
        self.model_name = model_name
        self.freq = sampled_freq
        self.bp_dict = {
            "ABP": "Arterial Blood Pressure",
            "SBP": "Systolic Blood Pressure",
            "DBP": "Diastolic Blood Pressure",
        }
        self.use_wandb = use_wandb
        self.mimic_visual_dict = {}

    def fig_to_wandb_image(self, fig, caption="default caption"):
        """fig_to_wandb_image Whether to convert pyplot figures to
            Weight & Bias Image (https://docs.wandb.ai/guides/track/log/media)

        Arguments:
            fig {pyplot} -- pyplot figure to be converted

        Keyword Arguments:
            caption {str} -- caption of the Weight & Bias Image (default: {"default caption"})

        Returns:
            wandb Image -- outputted Weights & Bias Image
        """
        if self.use_wandb:
            return wandb.Image(fig, caption)
        return fig

    def plot_bland_altman(self, bp_type):
        """plot_bland_altman
            Formulating inputs based on object variable;
            Calling the external function;
            Returning the constructed dictionary of information

        Arguments:
            bp_type {str} -- whether to construct input for ABP, SBP, or DBP

        Returns:
            dict -- returning a dictionary containing desired visuliazation
        """
        bp_key = bp_type.upper()
        fig, rmse, mae, mean, std, rval, pval = plot_bland_altman(
            pred_arr=self.pred_dict[bp_key],
            test_arr=self.test_dict[bp_key],
            title=f"{self.bp_dict[bp_key]} Bland Altman for\nPatient: {self.patient_name}",
            BP_type=bp_key,
            return_plt=True,
        )
        fig = self.fig_to_wandb_image(
            fig, caption=f"{self.bp_dict[bp_key]}_Bland_Altman"
        )
        return form_dict(bp_key, fig, rmse, mae, mean, std, rval, pval)

    def plot_Prediction(
        self, bp_type, vis_st_idx=0, vis_seq_len=1250, add_metrics=False
    ):
        """plot_Prediction
            Formulating inputs based on object variable;
            Calling the external function;
            Returning the constructed dictionary of information

        Arguments:
            bp_type {str} -- whether to construct input for ABP, SBP, or DBP

        Keyword Arguments:
            vis_st_idx {int} -- starting index of the sequence of the visualization (default: {0})
            vis_seq_len {int} -- length of sequence of the visualization (default: {1250})
            add_metrics {bool} -- whether to include metrics info on the visualization (default: {False})

        Returns:
            dict -- returning a dictionary containing desired visuliazation
        """
        bp_key = bp_type.upper()
        x_lab = "Time (seconds)" if bp_key == "ABP" else "Data Points (# of Samples)"
        freq = self.freq if bp_key == "ABP" else 1
        st_idx = max(0, vis_st_idx)
        ed_idx = min(len(self.pred_dict[bp_key]), vis_st_idx + vis_seq_len)
        print(f"st_idx: {st_idx}, ed_idx: {ed_idx}, print_len = {len(self.test_dict[bp_key])}")
        wave_fig, ax = plot_waveform(
            pred_arr=self.pred_dict[bp_key][st_idx:ed_idx],
            test_arr=self.test_dict[bp_key][st_idx:ed_idx],
            freq=freq,  # Match frequency for A, else point by point
            x_lab=x_lab,
            y_lab=f"{self.bp_dict[bp_key]} (mmHg)",
            title=f"{self.model_name.upper()} {self.bp_dict[bp_key]} Prediction for\nPatient: {self.patient_name}",
            add_metrics=add_metrics,
        )
        wave_fig = self.fig_to_wandb_image(
            wave_fig, caption=f"{self.bp_dict[bp_key]}_Prediction"
        )
        return {f"{bp_key}_Waveform": wave_fig}

    def plot_three_Predictions(
        self, bp_type, vis_st_idx=0, vis_seq_len=1250, add_metrics=False
    ):
        """plot_Prediction
            Formulating inputs based on object variable;
            Calling the external function;
            Returning the constructed dictionary of information

        Arguments:
            bp_type {str} -- whether to construct input for ABP, SBP, or DBP

        Keyword Arguments:
            vis_st_idx {int} -- starting index of the sequence of the visualization (default: {0})
            vis_seq_len {int} -- length of sequence of the visualization (default: {250})
            add_metrics {bool} -- whether to include metrics info on the visualization (default: {False})

        Returns:
            dict -- returning a dictionary containing desired visuliazation
        """
        bp_key = bp_type.upper()
        x_lab = "Time (seconds)" if bp_key == "ABP" else "Data Points (# of Samples)"
        freq = self.freq if bp_key == "ABP" else 1
        pred_list = []
        test_list = []
        vis_seq_len = min(vis_seq_len, int(len(self.pred_dict[bp_key])/3))
        for i in range(3):
            st_idx = max(0, vis_st_idx) + i * vis_seq_len
            ed_idx = (
                min(len(self.pred_dict[bp_key]), vis_st_idx + vis_seq_len)
                + i * vis_seq_len
            )
            pred_list.append(self.pred_dict[bp_key][st_idx:ed_idx])
            test_list.append(self.test_dict[bp_key][st_idx:ed_idx])
            print(f"st_idx: {st_idx}, ed_idx: {ed_idx}")
        wave_fig, ax = plot_three_waveform(
            pred_arr_list=pred_list,
            test_arr_list=test_list,
            freq=freq,  # Match frequency for A, else point by point
            x_lab=x_lab,
            y_lab=f"{self.bp_dict[bp_key]} (mmHg)",
            title=f"{self.model_name.upper()} {self.bp_dict[bp_key]} Prediction for\nPatient: {self.patient_name}",
            add_metrics=add_metrics,
        )
        wave_fig = self.fig_to_wandb_image(
            wave_fig, caption=f"{self.bp_dict[bp_key]}_Prediction"
        )
        return {f"{bp_key}_Three_Waveform": wave_fig}

    def plot_confusion_matrix(self, bp_type):
        """plot_confusion_matrix
            Formulating inputs based on object variable;
            Calling the external function;
            Returning the constructed dictionary of information

        Arguments:
            bp_type {str} -- whether to construct input for ABP, SBP, or DBP

        Returns:
            dict -- returning a dictionary containing desired visuliazation
        """
        bp_key = bp_type.upper()
        cf_fig = confusion_matrix_of_stages(
            pred_arr=self.pred_dict[bp_key],
            test_arr=self.test_dict[bp_key],
            bp_type=bp_key,
            pname=f"{self.patient_name}",
        )
        cf_fig = self.fig_to_wandb_image(
            cf_fig, caption=f"{self.bp_dict[bp_key]}_Confusion_Matrix"
        )
        return {f"{bp_key}_Confusion_Matrix": cf_fig}

    def plot_everything(self):
        """plot_everything
            Plotting all of Bland Altman, Waveform, and Confusion Matrix Plots
            Returning a dictionary of all information this object does

        Returns:
            dict -- A dictionary of all MIMIC Visualization information available
        """
        # adding plot_waveform
        self.mimic_visual_dict.update(self.plot_Prediction("abp", add_metrics=True))
        self.mimic_visual_dict.update(self.plot_Prediction("sbp", add_metrics=True))
        self.mimic_visual_dict.update(self.plot_Prediction("dbp", add_metrics=True))
        self.mimic_visual_dict.update(
            self.plot_three_Predictions("abp", add_metrics=True)
        )
        # adding bland altman
        self.mimic_visual_dict.update(self.plot_bland_altman("sbp"))
        self.mimic_visual_dict.update(self.plot_bland_altman("dbp"))
        # adding confusion matrix
        self.mimic_visual_dict.update(self.plot_confusion_matrix("sbp"))
        self.mimic_visual_dict.update(self.plot_confusion_matrix("dbp"))
        return self.mimic_visual_dict


if __name__ == "__main__":
    # Declaring random numbers for testing purposes
    use_wandb = True
    import wandb

    if use_wandb:
        wandb.init(
            project="play_ground",
            reinit=True,
            tags=["visual combine"],
        )
        log_dict = {}
    np.random.seed(12)
    wf_dict = {
        "abp_pred": np.random.normal(100, 20, 4000),
        "abp_test": np.random.normal(100, 20, 4000),
        "sbp_pred": np.random.normal(120, 20, 375),
        "sbp_test": np.random.normal(120, 20, 375),
        "dbp_pred": np.random.normal(80, 20, 375),
        "dbp_test": np.random.normal(80, 20, 375),
    }
    # declaring Visual Class
    MV = MIMIC_Visual(
        wf_dict,
        patient_name="Sicong(clearloveyanzhen)",
        model_name="Deep Learning",
        use_wandb=use_wandb,
    )
    overall_visual_dict = MV.plot_everything()
    # testing visualization
    if not use_wandb:
        overall_visual_dict["DBP_Bland_Altman"].savefig(
            "../plot_dir/DBP_bland_altman.png"
        )
        overall_visual_dict["SBP_Bland_Altman"].savefig(
            "../plot_dir/SBP_bland_altman.png"
        )
        overall_visual_dict["SBP_Confusion_Matrix"].savefig(
            "../plot_dir/SBP_confusion_matrix.png"
        )
        overall_visual_dict["DBP_Confusion_Matrix"].savefig(
            "../plot_dir/DBP_confusion_matrix.png"
        )
        overall_visual_dict["ABP_Waveform"].savefig("../plot_dir/ABP_visual_test.png")
        overall_visual_dict["ABP_Three_Waveform"].savefig(
            "../plot_dir/ABP_Three_visual_test.png"
        )
        overall_visual_dict["SBP_Waveform"].savefig("../plot_dir/SBP_visual_test.png")
        overall_visual_dict["DBP_Waveform"].savefig("../plot_dir/DBP_visual_test.png")
    else:
        log_dict = overall_visual_dict
        wandb.log(log_dict)
    for each in overall_visual_dict:
        print(f"{each}\t{overall_visual_dict[each]}\t{type(overall_visual_dict[each])}")
    pprint(overall_visual_dict)
    print("This main func is used for testing purpose only")
