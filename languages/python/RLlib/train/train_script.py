"""
This example is modified from the following repo.
https://github.com/NREL/rlc4clr/blob/main/train/train_stg1.py
"""


import os
import sys

import ray

from ray import air, tune
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env


CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(CURRENT_FILE_PATH, 'results')

torch, nn = try_import_torch()

if __name__ == "__main__":

    from config_parser import create_parser
    parser = create_parser()
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    # Ray cluster initialization
    if args.ip_head is not None:
        ray.init(address=args.ip_head, 
                 _redis_password=args.redis_password,
                 local_mode=False)
    else:
        ray.init(local_mode=args.local_mode)

    env_name = 'CarPass-v0'
    def env_creator(config):

        custom_env_dir = CURRENT_FILE_PATH.replace('train', 'custom_gym_env')
        sys.path.append(os.path.dirname(custom_env_dir))

        from custom_gym_env.custom_env import CarPassEnv
        env = CarPassEnv()
        return env
    register_env(env_name, env_creator)

    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(env_name, env_config={})
        .framework(args.framework)
        .rollouts(num_rollout_workers=args.worker_num)
        .training(train_batch_size=args.train_batch_size, 
                  lr=args.lr,
                  model={"fcnet_hiddens": [256, 256]})
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    stop = {
        # "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    tuner = tune.Tuner(
        args.run,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop, local_dir=LOG_PATH,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_frequency,
                num_to_keep=args.checkpoint_to_save,
                checkpoint_score_attribute='sampler_results/episode_reward_mean'
            )),
    )
    results = tuner.fit()

    if args.as_test:
        print("Checking if learning goals were achieved")
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
