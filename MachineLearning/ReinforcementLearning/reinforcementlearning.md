---
layout: default
title: Reinforcement Learning   
parent: Machine Learning
---

# Reinforcement Learning on Eagle

Here, we give directions for using Ray and OpenAI Gym environments to run experiments on Eagle. Please note that this tutorial is accompanied by a dedicated GitHub repo, which you will find [here](https://github.com/erskordi/HPC/tree/HPC-RL/languages/python/openai_rllib).

## Create Anaconda environment

As a first step, you need to create an Anaconda environment that you will use for all your experiments. Below there is a list of the steps you need to follow:

### 1<sup>st</sup> step: Log in on Eagle

Login on Eagle with:
```
ssh eagle
```
or
```
ssh <username>@el1.hpc.nrel.gov
```

### 2<sup>nd</sup> step: Set up Anaconda environment

Use the `env_example.yml` file that can be found in the aforementioned repo to create the new Anaconda environment. You can do it to a directory of your choosing. There are three main directories on Eagle where you can install the new environment, namely `/home`, `/scratch`, and `/projects`. Depending on your needs, you have to choose one of these three. For more information regarding installing your new environment and the different Eagle directories, please see [here](https://nrel.github.io/HPC/languages/python/NREL_python.html) and [here](https://nrel.github.io/HPC/languages/python/conda.html).

For example: 

Create a directory `/scratch/$USER/github-repos/` if you don't have one already. Clone the repo there, and `cd` to the repo directory. You can also create a directory where all your Anaconda environments will reside, e.g. `/scratch/$USER/conda-envs/`. Assuming you want to install the environment on your `scratch` directory, you can do the following:
```
conda env create --prefix=/scratch/$USER/conda-envs/myenv -f env_example.yml
```

### 3<sup>rd</sup> step: Run OpenAI Gym on a single node/single core

After the environment is created, you need to make sure everything is working correctly. For OpenAI Gym, you can test your installation by running a small example using one of the standard Gym environments (e.g. `CartPole-v0`).

Begin by activating the enironment and start a Python session
```
module purge
conda activate /scratch/$USER/conda-envs/myenv
python
```
and run the following:
```python
import gym

env = gym.ens.make("CartPole-v0")
env.reset()

done = False

while not done:
    action = env.action_space.sample()
    obs, rew, done, _ = env.step(action)
    print(action, obs, rew, done)
```
If everything works correctly, you will see an output similar to:
```
0 [-0.04506794 -0.22440939 -0.00831435  0.26149667] 1.0 False
1 [-0.04955613 -0.02916975 -0.00308441 -0.03379707] 1.0 False
0 [-0.05013952 -0.22424733 -0.00376036  0.2579111 ] 1.0 False
0 [-0.05462447 -0.4193154   0.00139787  0.54940559] 1.0 False
0 [-0.06301078 -0.61445696  0.01238598  0.84252861] 1.0 False
1 [-0.07529992 -0.41950623  0.02923655  0.55376634] 1.0 False
0 [-0.08369004 -0.61502627  0.04031188  0.85551538] 1.0 False
0 [-0.09599057 -0.8106737   0.05742218  1.16059658] 1.0 False
0 [-0.11220404 -1.00649474  0.08063412  1.47071687] 1.0 False
1 [-0.13233393 -0.81244634  0.11004845  1.20427076] 1.0 False
1 [-0.14858286 -0.61890536  0.13413387  0.94800442] 1.0 False
0 [-0.16096097 -0.8155534   0.15309396  1.27964413] 1.0 False
1 [-0.17727204 -0.62267747  0.17868684  1.03854806] 1.0 False
0 [-0.18972559 -0.81966549  0.1994578   1.38158021] 1.0 False
0 [-0.2061189  -1.0166379   0.22708941  1.72943365] 1.0 True
```
Note that the above process does not involve any training, it works only as a sanity check.

# Agent training with Ray/RLlib

Reinforcement learning algorithms are notorious for the amount of data they need to collect in order to perform adequate agent training. The more data collected, the better the training will be. However, we also need to collect massive amounts of data in reasonable time. That is where RLlib can assist us. 

RLlib is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications ([source](https://docs.ray.io/en/master/rllib.html)). It supports all known deep learning frameworks such as Tensorflow, Pytorch, although most parts are framework-agnostic and can be used by either one.

To demonstrate RLlib's capabilities, this page describes a simple example of training an RL agent. As above, the `CartPole-v0` OpenAI Gym environment will be used.

## Import packages

Begin by importing the most basic packages:
```python
import ray
from ray import tune
```
`Ray` consists an API readily available for building distributed applications, hence its importance for parallel RL training. On top of it, there are several problem-solving libraries, one of which is RLlib.

`Tune` is another one of `Ray`'s libraries for scalable hyperparameter tuning. All RLlib trainers (scripts for RL agent training) are compatible with Tune API, making experimenting quite easy.

We also import the `argparse` package with which you can setup a number of flags. These flags will allow you to control certain hyperparameters, such as:
* RL algorithm (e.g. PPO, DQN)
* Number of CPUs/GPUs
* ...and others
```python
import argparse
```

## Create flags
Let's define the following flags:
```python
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--name-env", type=str, default="CartPole-v0")
parser.add_argument("--run", type=str, default="DQN")
```
All of them are self-explanatory, however let's see each one separately.
1. `--num-cpus`: Define how many CPU cores you want to utilize (Default value 0 means allocation of a single CPU core).
2. `--num-gpus`: If you allocate a GPU node, then you can set this flag equal to 1. It also accepts partial values, in case you don't want 100% of the GPU utilized.
3. `--name-env`: The name of the OpenAI Gym environment (later you will see how to register your own environment).
4. `--run`: Specify the RL algorithm for agent training.

## Initialize Ray

Initialize Ray using the following command:
```python
ray.init()
```

## Run experiments with Tune

This is the final step in this basic trainer. Using Tune's `tune.run` function, you will initiate the agent training. This function takes three basic arguments:
* RL algorithm (string): It is defined in the `--run` flag.
* `stop` (dictionary): Provide a criterion to stop training (in this example is the number of training iterations, stop training when iterations reach 10,000).
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

That's it! Your first RLlib trainer for running reinforcement learning experiments on Eagle is ready!

You can find the code of this example in the repo (`simple_trainer.py`), under the `simple-example` directory.


# Run experiments on Eagle

Here we give the necessary steps to succesfully run the `simple_trainer.py` example on Eagle. For any kind of experiment you want to run later, you have to follow the same steps.

## Allocate an interactive Eagle node

Firstly, allocate an interactive node. For this example, let's start by allocating a `debug` node. Debug nodes have a maximum allocation time of one hour (60 minutes):
```
srun -n1 -t60 -<project_name> --partition debug --pty $SHELL
```
and activate the environment you created:
```
module purge
conda activate env_example
```
**VERY IMPORTANT!!** 
Before running your experiment, run
```
unset LD_PRELOAD
```
For communication between cores in a node, RLlib uses a Redis server. However, there is some kind of process running on Eagle causing Redis server to malfunction. Therefore, make sure you unset variable `LD_PRELOAD`, which disables that process and lets your experiment to run smoothly.

## Run multi-core experiments

The example in the previous section, using default flag values, is designed to run utilizing only a single CPU core. It can also run on Eagle, after allocating a single node and you decide to run the experiments on a single CPU of this node. However, as explained above, RL training is highly benefited by running multiple concurrent experiments, that is evaluations of OpenAI Gym environments. A single node on Eagle has 36 CPU cores, therefore it is prudent to utilize all of them for faster agent training. 

In order to exploit the Eagle node 100%, you need to adjust the `--num-cpus` hyperparameter to reflect to all CPUs on the node. Therefore, you can run the following:
```
python simple_trainer.py --num-cpus 35
```

You may wonder why set up the number of CPUs to 35 when there are 36 cores on an Eagle node. That happens because RLlib always sets a minimum of one core in `num_workers` key of the `config` dictionary, even if you don't (remember the default `--num-cpus` flag value of zero). In the current setting of the aforementioned example (`--num-cpus`: 0), RLlib will actually utilize 1 core. So, by setting the `--num-cpus` hyperparameter to 35, RLlib will actually allocate 36 cores, which means 100% utilization of the Eagle node.

Such is not the case with the `num_gpus` key, where zero means no GPU allocation is permitted. This is because GPUs are used for training the policy network and not running the OpenAI Gym environment instances, and thus they are not mandatory (although having a GPU node can assist the agent training by reducing training time).
