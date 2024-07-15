import argparse
import pdb

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


def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1")


def str2list(v: str) -> list:
    return list(sorted(map(int, v.split(","))))


def environment_scope(v: str) -> str:
    scope_dict = {
        "trial": "trial2trial/",
        "session": "session2session/",
    }
    return scope_dict[v.lower()]


def lower(v: str) -> str:
    return v.lower()


def print_flags(flags: argparse.Namespace):
    for k, v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))


def ablate_argparse(parser=None) -> argparse.Namespace:
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation_type",
        type=str,
        default="None",
        help="Determine which ablation to perform (default=None)",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Determine how much training data (default=0.8)",
    )
    parser.add_argument(
        "--calibration_time",
        type=int,
        default=60,
        help="Determine how much training time in minutes (default=60)",
    )
    parser.add_argument(
        "--hide_bp_type",
        type=lower,
        default="sbp",
        help="Determine which range of BP type to hide (default=sbp)",
    )
    parser.add_argument(
        "--hide_bp_range",
        type=str2list,
        default="100,105",
        help="Determine which range of BP type to hide (default=120,130)",
    )
    parser.add_argument(
        "--domain_length",
        type=int,
        default=30,
        help="Determine the length of a domain segment (default=30)",
    )
    parser.add_argument(
        "--gaussian_noise_rate",
        type=float,
        default=0.5,
    )
    return parser


if __name__ == "__main__":
    flags = ablate_argparse()
