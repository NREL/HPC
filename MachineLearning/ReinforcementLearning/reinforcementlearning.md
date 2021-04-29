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

# Outputs

RLlib produces outputs of the following form:
```
Resources requested: 1/36 CPUs, 0/0 GPUs, 0.0/119.73 GiB heap, 0.0/38.13 GiB objects
Result logdir: /home/eskordil/ray_results/CartPole-v0
Number of trials: 1 (1 RUNNING)
+-----------------------------+----------+-------+
| Trial name                  | status   | loc   |
|-----------------------------+----------+-------|
| DQN_CartPole-v0_380be_00000 | RUNNING  |       |
+-----------------------------+----------+-------+


(pid=200639) 2021-04-29 10:44:55,555	INFO trainer.py:585 -- Tip: set framework=tfe or the --eager flag to enable TensorFlow eager execution
(pid=200639) 2021-04-29 10:44:55,555	INFO trainer.py:612 -- Current log_level is WARN. For more information, set 'log_level': 'INFO' / 'DEBUG' or use the -v and -vv flags.
(pid=200639) 2021-04-29 10:44:56,504	WARNING util.py:37 -- Install gputil for GPU system monitoring.
Result for DQN_CartPole-v0_380be_00000:
  custom_metrics: {}
  date: 2021-04-29_10-44-57
  done: false
  episode_len_mean: 22.906976744186046
  episode_reward_max: 86.0
  episode_reward_mean: 22.906976744186046
  episode_reward_min: 8.0
  episodes_this_iter: 43
  episodes_total: 43
  experiment_id: 66ccec197de8447fb178c8abebd26107
  experiment_tag: '0'
  hostname: r7i7n35
  info:
    last_target_update_ts: 1000
    learner:
      default_policy:
        cur_lr: 0.0005000000237487257
        max_q: 0.3239544630050659
        mean_q: -0.08293987810611725
        mean_td_error: -1.207885503768921
        min_q: -1.2971210479736328
        model: {}
    num_steps_sampled: 1000
    num_steps_trained: 32
    num_target_updates: 1
  iterations_since_restore: 1
  node_ip: 10.148.7.231
  num_healthy_workers: 0
  off_policy_estimator: {}
  perf:
    cpu_util_percent: 2.85
    ram_util_percent: 5.0
  pid: 200639
  policy_reward_max: {}
  policy_reward_mean: {}
  policy_reward_min: {}
  sampler_perf:
    mean_env_wait_ms: 0.042502815787727896
    mean_inference_ms: 0.49455801804701643
    mean_processing_ms: 0.1114586611965915
  time_since_restore: 0.9389147758483887
  time_this_iter_s: 0.9389147758483887
  time_total_s: 0.9389147758483887
  timers:
    learn_throughput: 197.465
    learn_time_ms: 162.054
  timestamp: 1619714697
  timesteps_since_restore: 0
  timesteps_total: 1000
  training_iteration: 1
  trial_id: 380be_00000
```
Most probably you won't need much of this information, however there are some parts of it that can give you an idea of the quality of results that you get.

One important piece of information is the utilization of resources achieved by RLlib during training, as well as the RL algorithm used for the experiment:
```
Resources requested: 1/36 CPUs, 0/0 GPUs, 0.0/119.73 GiB heap, 0.0/38.13 GiB objects
Result logdir: /home/eskordil/ray_results/CartPole-v0
Number of trials: 1 (1 RUNNING)
+-----------------------------+----------+---------------------+--------+------------------+------+----------+
| Trial name                  | status   | loc                 |   iter |   total time (s) |   ts |   reward |
|-----------------------------+----------+---------------------+--------+------------------+------+----------|
| DQN_CartPole-v0_380be_00000 | RUNNING  | 10.148.7.231:200639 |      2 |          2.55838 | 2000 |    19.69 |
+-----------------------------+----------+---------------------+--------+------------------+------+----------+
```
These lines inform us about the following:
* Number of CPUs utilized: Since `--num-cpus` was defined as equal to zero, then by default RLlib acquired a single CPU core to run the experiment.
* No GPU resources were utilized: Expected since `--num-gpus` was set to zero.
* Trial name: Generated automatically, it gives information regarding the specific RL algorithm used (here DQN).
* status: either RUNNING or FAILED (in case there was an error during training).
* iter: Here you see the number of training iterations (remember, we set the maximum value for it to 10,000)
* total time (seconds): time spend for one iteration
* reward: The reward returned after the end of this iteration. Succesfull agent training will be shown through increase of that value during the process.

These are the most straightforward, but in the same time important, information you will get after every training iteration. The rest of this snippet contains more specialized information, but for most of it you will never have to be concerned about. 
