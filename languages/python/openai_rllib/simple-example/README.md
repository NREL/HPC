# Reinforcement Learning with Ray/RLlib

Reinforcement learning algorithms are notorious for the amount of data they need to collect in order to perform adequate agent training. The more data collected, the better the training will be. However, we also need to collect massive amounts of data in reasonable time. That is where RLlib can assist us. 

[RLlib](https://docs.ray.io/en/master/rllib.html) is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. It supports all known deep learning frameworks such as Tensorflow, Pytorch, although most parts are framework-agnostic and can be used by either one.

To demonstrate RLlib's capabilities, we provide here a simple example of training an RL agent for one of the standard OpenAI Gym environments, the CartPole. The example, which can be found in the `simple_trainer.py` file, utilizes the power of RLlib in running multiple experiments in parallel by exploiting as many CPUs and/or GPUs are available on your machine. Below, you will find a detailed description of how this example works.

Note here that the Ray version used is the latest at the time this tutorial is written (*version 1.3.0*)

## Import packages

You begin by importing the most basic packages:
```python
import ray
from ray import tune
```
`Ray` consists of an API readily available for building distributed applications, hence its importance for parallel RL training. On top of it, there are several problem-solving libraries, one of which is RLlib.

`Tune` is another one of `Ray`'s libraries for scalable hyperparameter tuning. All RLlib trainers (scripts for RL agent training) are compatible with Tune API, making experimenting in RL quite easy. All the trainer examples posted in this repo utilize Tune for hyperparameter tuning and agent training.

We also import the `argparse` package with which you can setup a number of flags. These flags will allow you to control certain hyperparameters, such as:
* RL algorithm (e.g. PPO, DQN)
* Number of CPUs/GPUs
* ...and others
```python
import argparse
```

## Create flags
Using the `argparse` package, you can define the following flags:
```python
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--name-env", type=str, default="CartPole-v0")
parser.add_argument("--run", type=str, default="DQN")
```
All of them are self-explanatory, however let's see each one separately.
1. `--num-cpus`: Define how many CPU cores you want to utilize.
2. `--num-gpus`: If you allocate a GPU node, then you can set this flag equal to 1. It also accepts partial values, in case you don't want 100% of the GPU utilized.
3. `--name-env`: The name of the OpenAI Gym environment (later you will see how to register your own environment).
4. `--run`: Specify the RL algorithm for agent training.

There are also other flags that can be used in the trainer, but they will be explained in later examples.

## Initialize Ray

This is an easy step, you just have to do
```python
ray.init()
```
You have just initialized a Ray session!

## Run experiments with Tune

This is the final step in this basic trainer. Using Tune's `tune.run` function, you will initiate the agent training. This function takes three basic arguments:
* RL algorithm (string): It is defined in the `--run` flag.
* `stop` (dictionary): Provide a criterion to stop training (in this example is the number of training iterations, stop training when they reach 10,000).
* `config` (dictionary): Basic information for training, contains the OpenAI Gym environment name, number of CPUs/GPUs, and possible others.
```python
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
```

# Run experiments on Eagle

## Login on Eagle

Begin by logging in on Eagle:
```
ssh <username>@eagle.hpc.nrel.gov
```

## Activate Conda environment

After logging in on Eagle, move to the repo directory:
```
cd /scratch/$USER/git-repos/openai_gym_tutorial/simple-example/
```

## Allocate an interactive Eagle node

Now you can allocate an interactive node. For this example, let's start by allocating a `debug` node since it is faster. Debug nodes have a maximum allocation time of one hour (60 minutes):
```
srun -n1 -t10 -<project_name> --partition debug --pty $SHELL
```
and activate the environment you created:
```
module purge
conda activate env_example
```
## Run single-node/multi-core experiments

The example in the previous section, keeping the default values, is designed to run on a local machine with a single CPU nodes and no GPUs are utilized. It can also run on Eagle, after allocating a single node and you decide to run the experiments on a single CPU of this node. However, as explained above, RL training is highly benefited by running experiments, that is evaluations of OpenAI Gym environments. A single node on Eagle has 36 CPU cores, therefore it is prudent to utilize all of them for faster agent training. 

In order to exploit the Eagle node 100%, you need to adjust the `--num-cpus` hyperparameter to reflect to all CPUs on the node. Therefore, you can run the following:
```
python simple_trainer.py --num-cpus 35
```

You may wonder why set up the number of CPUs to 35 when there are 36 cores on an Eagle node. That happens because RLlib always sets a minimum of one core in `num_workers` key of the `config` dictionary, even if you don't. In the current setting of the aforementioned example (`--num-cpus`: 1), RLlib will actually utilize 2 cores. So, by setting the `--num-cpus` hyperparameter to 35, RLlib will actually allocate 36 cores, which means 100% utilization of the Eagle node. Such is not the case with the `num_gpus` key, where zero means no GPU allocation is permitted. This is because GPUs are used for training the policy network and not running the OpenAI Gym environment instances, and thus they are not mandatory (although having a GPU node can assist the agent training by reducing training time).

## Run multi-node/multi-core experiments

Problems that involve highly complex environments with very large observation and/or action state spaces will probably require running experiments utilizing more than one Eagle nodes. In this case it is better to work with slurm scripts. You can submit such scripts as 
```
sbatch <name_of_your_batch_script>
```
The results are exported in an `slurm-<job_id>.out` file. This file can be accesssed:
 * During training (`tail -f slurm-<job_id>.out`) 
 * Open it using a standard text editor (e.g. `nano`) after training is finished.

An example of a `slurm-<job_id>.out` file is also included in the repo for reference.