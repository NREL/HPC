import argparse


def create_parser():

    parser = argparse.ArgumentParser()

    # Training Setup
    parser.add_argument(
        "--run", type=str, default="PPO",
        help="The RLlib-registered algorithm to use."
    )
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--checkpoint-frequency", type=int, default=20,
        help="How frequent the policy will be saved."
    )
    parser.add_argument(
        "--checkpoint-to-save", type=int, default=3,
        help="Number of best policies to save."
    )
    parser.add_argument(
        "--worker-num", type=int, default=102,
        help="Number of parallel workers."
    )
    parser.add_argument(
        "--exp-note", type=str, default=None,
        help="The experiment note will be used in the path for saving models."
    )

    # Ray Configuration
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
        "be achieved within --stop-timesteps AND --stop-iters.",
    )
    parser.add_argument(
        "--ip-head", type=str, default=None,
        help="The IP address of the head node of the ray cluster."
    )
    parser.add_argument(
        "--redis-password", type=str, default=None,
        help="The password to connect to the ray cluster."
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Run without Tune using a manual train loop instead. In this case,"
        "use PPO without grid search and no TensorBoard.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    # Training stopping criteria
    parser.add_argument(
        '--run-hour', type=int, default=1
    )
    parser.add_argument(
        "--stop-timesteps", type=int, default=8e7,
        help="Number of timesteps to train."
    )
    parser.add_argument(
        "--stop-reward", type=float, default=210,
        help="Reward at which we stop training."
    )
    parser.add_argument(
        "--stop-iters", type=int, default=80000,
        help="Number of iterations to train."
    )

    # Algorithm hyperparameters (PPO)
    parser.add_argument(
        '--lr', type=float, default=5e-5
    )
    parser.add_argument(
        '--train-batch-size', type=int, default=30000
    )
    parser.add_argument(
        '--entropy-coeff', type=float, default=0.0
    )
    parser.add_argument(
        '--policy-layers', nargs="+", type=int, default=None
    )

    return parser
