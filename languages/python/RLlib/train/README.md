## Training RL policy on Kestrel

This directory contains examples of using RLlib for RL policy training.

### Files in this directory

`train_script.py`: The main python file that sets up and initiates the training process. **It not only works on Kestrel, but can be used for local training if you want to test on your local computer.**

`config_parser.py`: Use to create an argparser with the options to customize training hyperparameters. Default values are also set within the file.

`train_single_node.sh`: The SLURM sbatch script for job submission. This script shows an example of using a single computing node for training.

`train_multi_nodes.sh`: The SLURM sbatch script for job submission. This script shows an example of using multiple Kestrel nodes for training.

### Main training script

Here we walk though the code in `train_script.py` to help the readers understand what each part of the code is doing.

Firstly, loading configuration arguments for this training experiment.

```
from config_parser import create_parser
parser = create_parser()
args = parser.parse_args()

print(f"Running with following CLI options: {args}")
```

This allows setting different configuration values in the SLURM sbatch script for the training: `python train_script.py --worker-num 100`. See [`config_parser.py`](config_parser.py) for different configuration options.

RLlib training relies on the distributed computing framework called [Ray](https://www.ray.io/), so a Ray cluster needs to be initialized.

When the `ip_head` configuration is provided, we will connect to the Ray server at that address. This usually happens when using multiple nodes for training and the Ray cluser is started in [train_multi_nodes.sh](train_multi_nodes.sh). Otherwise, we start the Ray server using `ray.init(local_mode=args.local_mode)`, and this happens in single node training.

```
# Ray cluster initialization
if args.ip_head is not None:
    ray.init(address=args.ip_head, 
             _redis_password=args.redis_password,
             local_mode=False)
else:
    ray.init(local_mode=args.local_mode)
```

Register the Gym environment:
```
env_name = 'CarPass-v0'

def env_creator(config):
    ...
    return env

register_env(env_name, env_creator)
```

Setting up the learning configuration and stopping criteria for the training task.
```
config = (
    get_trainable_cls(args.run)
    .get_default_config()
    .environment(env_name, env_config={})
    .framework(args.framework)
    .rollouts(num_rollout_workers=args.worker_num)
    .training(train_batch_size=args.train_batch_size, 
              lr=args.lr,
              model={"fcnet_hiddens": [256, 256]})
    # Use GPUs iff `RLlib_NUM_GPUS` env var set to > 0.
    .resources(num_gpus=int(os.environ.get("RLlib_NUM_GPUS", "0")))
)

# Stopping criteria
stop = {
    # "training_iteration": args.stop_iters,
    "timesteps_total": args.stop_timesteps,
    "episode_reward_mean": args.stop_reward,
}
```

Here, `args.run` could be a string such as 'PPO', 'A3C', etc., indicating which RL algorithms to be used, and `get_trainable_cls` goes and fetches the right trainable class according to `args.run`. `.environment(env_name, ...)` specifies which environment to train on. `args.framework` defines the ML framework to use, either Tensorflow or PyTorch, and we use PyTorch as an example here. `.rollouts()` allows setting the number of parallel workers. This number is usually limited to the number of CPU cores, since a Kestrel node has 104 cores, we can have 103 rollout workers (leaving one core for other tasks.) When using multiple nodes for training, the rollout worker number is calculated [here](train_multi_nodes.sh#L17). In `.training(...)`, hyperparameters of the RL algorithm can be set. The neural network architecture can also be defined: we are using a [256, 256] fully connected network, but RLlib allows the user to set a customized network as well, see an example of a pre-trained network [here](https://github.com/NREL/rlc4clr/blob/main/train/train_stg2.py#L178).

The actual training starts with the following code snippet, by putting everything, e.g., RL algorithm, stopping criteria and other configurations, together.

```
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
```
There is an option to set up how model checkpoints are to be saved in the `checkpoint_config`. Within it, `checkpoint_frequency=x` is how frequent the policy model needs to be saved, e.g., every x iterations; `num_to_keep=y` indicates how many best checkpoints to be saved, i.e., the Top y performing policies; and `checkpoint_score_attribute` specifies using which attribute to evaluate the policies (and determine which ones are the "best".)

### SLURM Sbatch scripts

**NOTE: You will need to change the $HPC_HANDLE and $USER variable in the script to be able to run on Kestrel.**

There are two files for submitting training jobs: one for _single node job_ and the other for _multi-node job_, they follow the following two or three steps.

#### Step 1 Loading the conda env

```
module purge
module load anaconda3
conda activate /projects/$HPC_HANDLE/$USER/conda_envs/rl_hpc
```

#### Step 2 Starting Ray cluster (Only needed for multi-node case)

Code snippet omitted here, see the [multi-node train script](train_multi_nodes.sh) for details. In general, this step will calculate the worker number, obtain node name list, and start the Ray cluster on the head node and all worker nodes.

#### Step 3 Starting RL training

Start RL training by calling the main training script. This is where some hyperparameters or configurations can be set: e.g., `--run PPO` as shown below.

```
python -u train_script.py --run PPO --redis-password $redis_password --worker-num $worker_num --ip-head $ip_head
```
