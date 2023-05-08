import argparse


def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1")


def print_flags(flags: argparse.Namespace):
    for k, v in sorted(vars(flags).items()):
        print("\t{}: {}".format(k, v))


def lower_case(v: str) -> str:
    return v.lower()


def rex_argparse(parser=None) -> argparse.Namespace:
    if parser is None:
        parser = argparse.ArgumentParser()
    # Starting to evaluate the model
    parser.add_argument(
        "--l2_regularizer_weight",
        type=float,
        default=0.001,
        help="set l2 regularizer weight (default=0.001)",
    )
    parser.add_argument(
        "--n_restarts",
        type=int,
        default=1,
        help="determine how many times the experiments to be repeated",
    )
    parser.add_argument(
        "--penalty_anneal_iters",
        type=int,
        default=100,
        help="set penalty anneal iterations (default=100)",
    )
    parser.add_argument(
        "--penalty_weight",
        type=float,
        default=10000.0,
        help="set penalty weight (default=10000.0)",
    )
    parser.add_argument(
        "--eval_interval",
        type=float,
        default=0.1,
        help="Frequency of evaluation interval during training (proportion to overall training epochs, default=0.1, yielding 10 eval intervals)",
    )
    parser.add_argument(
        "--print_eval_intervals",
        type=str2bool,
        default=False,
        help="Choose whether to spress evalaution interval info (default True, prints intervals)",
    )
    parser.add_argument("--erm_amount", type=float, default=1.0)
    parser.add_argument("--early_loss_mean", type=str2bool, default=True)
    parser.add_argument(
        "--rex",
        type=str2bool,
        default=True,
        help="Whether to use Risk Extrapolation (default=True)",
    )
    parser.add_argument(
        "--mse",
        type=str2bool,
        default=True,
        help="whether to choose mean squuare error as rex penalty (true), or just absolute error(false), defaults=True",
    )
    parser.add_argument(
        "--test_index",
        type=int,
        default=2,
        help="Select which index of the data to be test index (default=2)",
    )

    parser.add_argument(
        "--jupyter_notebook",
        type=str2bool,
        default=False,
        help="Passed to identify which tqdm to use for train iteration visualization",
    )
    parser.add_argument(
        "--make_gif",
        type=str2bool,
        default=False,
        help="Whether to save screenshots of training progress to make a GIF (default=False)",
    )
    parser.add_argument(
        "--use_batchnorm",
        type=str2bool,
        default=True,
        help="Whether to use batch norm layer",
    )
    return parser


if __name__ == "__main__":
    parser = rex_argparse()
    flags = parser.parse_args()
    print_flags(flags)
