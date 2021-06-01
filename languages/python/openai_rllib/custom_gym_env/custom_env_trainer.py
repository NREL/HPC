import argparse
import os

import ray
from ray import tune
from ray.tune.registry import register_env
from custom_env import BasicEnv

###############################################
## Command line args
###############################################
parser = argparse.ArgumentParser(description="Script for training RLLIB agents")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--redis-password", type=str, default=None)
parser.add_argument("--local-mode", action="store_true")
args = parser.parse_args()
print(args)

################################################
## RLLIB SETUP
################################################
if args.redis_password is None:
    # Single node
    ray.init(local_mode=args.local_mode)
    num_cpus = args.num_cpus - 1
else:
    # On a cluster
    ray.init(_redis_password=args.redis_password, address=os.environ["ip_head"])
    num_cpus = args.num_cpus - 1

################################################
## Register custom environment
################################################
env_name = "custom-env"
register_env(env_name, lambda config: BasicEnv())

######################################
## Run TUNE Experiments!
######################################
tune.run(
    args.run,
    name=env_name,
    local_dir="/scratch/$USER/ray_results",
    stop={"training_iteration": 100},
    config={
        "env": env_name,
        "num_workers": num_cpus, 
        "num_gpus": args.num_gpus,
        "ignore_worker_failures": True
        }
    )
