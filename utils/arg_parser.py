import argparse
import sys

sys.path.append("/home/grads/s/siconghuang/CufflessBP/bioz_processing")
import sicong_util as su

sys.path.append("utils/")
sys.path.append("models/")
import rex_parser as rp


def str2bool(v: str) -> bool:
    """
    str2bool
        Convert String into Boolean

    Arguments:
        v {str} --input raw string of boolean command

    Returns:
        bool -- converted boolean value
    """
    return v.lower() in ("yes", "true", "y", "t", "1")


def sicong_argparse(model: str) -> argparse.Namespace:
    """
    sicong_argparse
        parsing command line arguments with reinforced formats

    Arguments:
        model {str} -- indicates which model being used

    Raises:
        RuntimeError: When the required model is unknown

    Returns:
        argparse.Namespace -- _description_
    """
    try:
        model_desc_dict = {
            "Sequnet": "Sicong's Sequence U-Net Approach to MIMIC Arterial Blood Pressure Sequene Modeling",
            "Transformer": "Zhale's Transformer Approach to MIMIC Arterial Blood Pressure Sequene Modeling",
            "Cycle_GAN": "Mahrabadi 2022 Baseline Approach of Cycle GAN, published on EMBC 2022 ",
        }
        if model not in model_desc_dict:
            raise RuntimeError(
                "Model Unknown, only 'Sequnet' and 'Transformer' are available at this time"
            )
        parser = argparse.ArgumentParser(description=model_desc_dict[model])
        parser.add_argument(
            "--batch_size",
            type=int,
            default=512,
            help="Choosing the Batch Size, default=4 (If out of memory, try a smaller batch_size)",
        )
        parser.add_argument(
            "--data_path",
            default="/data/datasets/GuidedAttnDataset/7_days/",
            help="Please provide the Path to MIMIC Patient Waveforms",
        )
        parser.add_argument(
            "--dense_layer",
            type=int,
            default=100,
            help="[Transformer only] Hidden Layer of Transformer Model",
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            help="Choosing the patience of early stopping criteria, Setting -1 to disable it",
        )
        parser.add_argument(
            "--email_reminder",
            type=str2bool,
            default=False,
            help="Whether to send an email upon completion",
        )
        parser.add_argument(
            "--embed_noise_rate",
            type=float,
            default=0,
            help="degree of gaussian noise (0~1) to the tested feature, (default=0, no noise)",
        )
        parser.add_argument(
            "-f",
            default="random stuff",
            help="Just put it so for calling by jupyter",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=1,
            help="Choose the max number of epochs, default=3 for testing purpose",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-4,
            help="Define Learning Rate, default=1e-4, if failed try something smaller",
        )
        parser.add_argument(
            "--mask_abp_threshold",
            type=int,
            default=0,
            help="determining whether to mask out part of interval during training",
        )
        parser.add_argument(
            "--model_used",
            default=model,
            help="Specifying which pre-built model to use (default Sequnet)",
        )
        parser.add_argument(
            "--print_eval",
            type=int,
            default=0,
            help="Providing train progress insights (default=0, means disabled)",
        )
        parser.add_argument(
            "--run_portion",
            type=float,
            default=0.1,
            help="Whether to run a portion for testing purpose, setting to 1 and run every waveform",
        )
        parser.add_argument("--save_result", type=str2bool, default=True)
        parser.add_argument(
            "--sel_gpu",
            type=int,
            default=6,
            help="Choosing which GPU to use (STMI has GPU 0~7)",
        )
        parser.add_argument(
            "--sel_subject",
            type=int,
            default=2,
            help="Selecting which subject to run (in sequential order)",
        )
        parser.add_argument(
            "--seq_result_dir",
            default="./seq2seq_results/",
            help=" Model save Location, defaults in ./seq2seq_results/",
        )
        parser.add_argument(
            "--shuffle_data",
            type=str2bool,
            default=False,
            help="Whether to shuffle data before train/test split",
        )
        parser.add_argument(
            "--training_size",
            type=float,
            default=0.8,
            help="Define how much portion of data is trained (default 0.8)",
        )
        parser.add_argument(
            "--use_wandb",
            type=str2bool,
            default=False,
            help="Whether to save progress and results to Wandb",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=0,
            help="Weight Decay hyperparameter",
        )
        parser.add_argument(
            "--win_size",
            type=int,
            default=256,
            help="Windowing size of each batch, default=256, consider balance between window size and batch size",
        )
        parser.add_argument(
            "--wandb_project",
            type=str,
            default="Seq_U-Net_MIMIC",
            help="If using Wandb, this shows the location of the project, usually don't change this one",
        )
        parser.add_argument(
            "--wandb_tag",
            type=str,
            default="default_sequnet",
            help="If using Wandb, define tag to help filter results",
        )
        parser.add_argument(
            "--win_overlap",
            type=int,
            default=32,
            help="Overlapping portion between windows, setting to 0 to disable sliding window",
        )
        parser.add_argument(
            "--load_portion",
            type=float,
            default=0.001,
            help="[cycleGAN Only], setting to 0 to disable sliding window",
        )
        parser.add_argument(
            "--train_or_test",
            default="train",
            help="[cycleGAN Only], setting to 0 to disable sliding window",
        )
        # Cardiac Segmentation Addition
        parser.add_argument(
            "--use_cardiac_seg",
            type=str2bool,
            default=True,
            help="Whether to do cardiac segmentation and use cardiac cycles instead of sliding window",
        )
        parser.add_argument(
            "--num_prev_cardiac_cycle_feature",
            type=int,
            default=9,
            help="How many previous cardiac cycle to formulate a windowed batch (only used if use_cardiac_seg is True)",
        )
        parser.add_argument(
            "--num_prev_cardiac_cycle_label",
            type=int,
            default=3,
            help="How many of cardiac cycles of ABP to predict (only used if use_cardiac_seg is True)",
        )
        parser.add_argument(
            "--rex_torch_path",
            default="/home/grads/s/siconghuang/REx_candidate_torch_tensors/",
            help="The path to saved MIMIC datasets for REx in torch tensors (depends on the server)",
        )
        parser = rp.rex_argparse(parser)
        flags, _ = parser.parse_known_args()
        # setting cuda device
        flags.device = f"cuda:{flags.sel_gpu}" if flags.sel_gpu > 0 else "cpu"
        print("Flags:")
        for k, v in sorted(vars(flags).items()):
            print("\t{}: {}".format(k, v))
        return flags
    except Exception as error_msg:
        print(error_msg)
        su.email_func(subject="ArgumentParser Failed", message=f"{error_msg}")


if __name__ == "__main__":
    flags = sicong_argparse("Transformer")
    print("This main func is used for testing purpose only")
