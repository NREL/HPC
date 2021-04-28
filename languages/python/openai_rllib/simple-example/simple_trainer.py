import argparse

import ray
from ray import tune

###############################################
## Command line args
###############################################
parser = argparse.ArgumentParser(description="Script for training RLLIB agents")
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--name-env", type=str, default="CartPole-v0")
parser.add_argument("--run", type=str, default="DQN")
args = parser.parse_args()

################################################
## RLLIB SETUP
################################################
ray.init()


######################################
## Run TUNE Experiments!
######################################
tune.run(
    args.run,
    name=args.name_env,
    stop={"training_iteration": 10000},
    config={
        "env": args.name_env,
        "num_workers": args.num_cpus, 
        "num_gpus": args.num_gpus,
        "ignore_worker_failures": True
        }
    )
