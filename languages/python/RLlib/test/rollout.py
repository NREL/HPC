# Followed the example below.
# https://github.com/ray-project/ray/blob/master/rllib/examples/sb2rllib_rllib_example.py

import os
import sys

from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import get_trainable_cls, register_env

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

env_name = 'CarPass-v0'
def env_creator(config):

    custom_env_dir = CURRENT_FILE_DIR.replace('test', 'custom_gym_env')
    sys.path.append(os.path.dirname(custom_env_dir))

    from custom_gym_env.custom_env import CarPassEnv
    env = CarPassEnv()
    return env
register_env(env_name, env_creator)

# Use the same configuration as the one for training except the rollout worker
# number, which can be limited at 1.
config = (
    get_trainable_cls("PPO")
    .get_default_config()
    .framework("torch")
    .training(model={"fcnet_hiddens": [256, 256]})  # Same as the one in training
    .rollouts(num_rollout_workers=1)
)

# Loading trained RL agent
agent = PPO(config, env=env_name)
checkpoint_folder = CURRENT_FILE_DIR.replace('test', 'train/results')
trial_id = "PPO_2024-03-03_07-07-34/PPO_CarPass-v0_623f2_00000_0_2024-03-03_07-07-35/checkpoint_000001"
checkpoint_path = os.path.join(checkpoint_folder, trial_id)
agent.restore(checkpoint_path)
print("agent restored.")

# Rollout the trained agent.
env = env_creator(None)
state, _ = env.reset()
terminated = False
truncated = False
total_rew = 0.0

while not (terminated or truncated):
    
    act = agent.compute_single_action(state)
    # print(act)
    state, reward, terminated, truncated, info = env.step(act)
    env.render()
    total_rew += reward

print(total_rew)
