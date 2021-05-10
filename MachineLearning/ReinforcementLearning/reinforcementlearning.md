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
ssh <username>@eagle.hpc.nrel.gov
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

Ray version used here: *1.3*

## Import packages

Begin by importing the most basic packages:
```python
import ray
from ray import tune
```
`Ray` consists of an API readily available for building distributed applications, hence its importance for parallel RL training. On top of it, there are several problem-solving libraries, one of which is RLlib.

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
parser.add_argument("--local-mode", action="store_true")
```
All of them are self-explanatory, however let's see each one separately.
1. `--num-cpus`: Define how many CPU cores you want to utilize (Default value 0 means allocation of a single CPU core).
2. `--num-gpus`: If you allocate a GPU node, then you can set this flag equal to 1. It also accepts partial values, in case you don't want 100% of the GPU utilized.
3. `--name-env`: The name of the OpenAI Gym environment (later you will see how to register your own environment).
4. `--run`: Specify the RL algorithm for agent training.
5. `--local-mode`: This flag, set on True, is necessary to show that experiments run on a single core/single node.

### Extra flags

In the script you will also find a flag that is not necessary for now:
```batch
parser.add_argument("--redis-password", type=str, default=None)
```
This flag will become essential for when you need to deploy your experiments on multiple Eagle nodes, so let's skip it for now.

## Initialize Ray

You can setup Ray to run either on a single node (local mode), or on a cluster. For convenience, we put an `if-else` statement on the `simple-trainer.py` script, which will automatically switch between modes, depending on your needs. Therefore, you won't have to have two separate scripts:
```python
if args.redis_password is None:
    # Single node
    ray.init(local_mode=args.local_mode)
    num_cpus = args.num_cpus - 1
else:
    # On a cluster
    ray.init(_redis_password=args.redis_password, address=os.environ["ip_head"])
    num_cpus = args.num_cpus - 1
```

Focus on the first statement. Since on local mode you don't need a server for communication between nodes, you only need to setup ray to run on a local mode: `ray.init(local_mode=args.local_mode)`. The next line denotes the number of CPU cores you want to use. Remember that RLlib always allocates one CPU core, even if you put `--num-cpus=0`, hence you subtract one from your total number of cores.

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

Note here that, except default hyperparameters like those above, every RL algorithm provided by RLlib has its own hyperparameters and their default values that need to be tuned beforehand, if necessary. Please refere [here](https://docs.ray.io/en/master/rllib-algorithms.html#available-algorithms-overview) for more details.

You can find the code of this example in the repo (`simple_trainer.py`), under the `simple-example` directory.


# Run experiments on Eagle

Here we give the necessary steps to succesfully run the `simple_trainer.py` example on Eagle. For any kind of experiment you want to run later, you have to follow the same steps.

## Allocate an interactive Eagle node

Firstly, allocate an interactive node. For this example, let's start by allocating a `debug` node. Debug nodes have a maximum allocation time of one hour (60 minutes):
```
srun -n1 -t10 -<project_name> --partition debug --pty $SHELL
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

<sup>**</sup> **Supplemental notes: **
As you noticed, when using RLlib for RL traning, there is no need to `import gym`, as we did in the non-training example, because RLlib recognizes automatically all benchmark OpenAI Gym environments. Even when you create your own custom-made Gym environments, RLlib provides proper functions with which you can register your environment before training.

# Outputs

RLlib produces outputs of the following form:
```
== Status ==
Memory usage on this node: 9.1/187.3 GiB
Using FIFO scheduling algorithm.
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

When you set `--num-cpus` equal to 35, then the aforementioned printout will be like this:
```
== Status ==
Memory usage on this node: 10.6/92.8 GiB
Using FIFO scheduling algorithm.
Resources requested: 36/36 CPUs, 0/0 GPUs, 0.0/54.25 GiB heap, 0.0/18.7 GiB objects
Result logdir: /home/eskordil/ray_results/CartPole-v0
Number of trials: 1 (1 RUNNING)
+-----------------------------+----------+-----------------+--------+------------------+------+----------+
| Trial name                  | status   | loc             |   iter |   total time (s) |   ts |   reward |
|-----------------------------+----------+-----------------+--------+------------------+------+----------|
| DQN_CartPole-v0_b665a_00000 | RUNNING  | 127.0.0.1:92844 |      3 |          6.30516 | 3360 |    19.07 |
+-----------------------------+----------+-----------------+--------+------------------+------+----------+


Result for DQN_CartPole-v0_b665a_00000:
custom_metrics: {}
date: 2021-04-29_09-16-00
done: false
episode_len_mean: 17.37
episode_reward_max: 97.0
episode_reward_mean: 17.37
episode_reward_min: 8.0
episodes_this_iter: 75
episodes_total: 233
experiment_id: 573f3799c81946439a598d6d633df7d4
experiment_tag: '0'
hostname: r4i7n35
info:
last_target_update_ts: 4480
learner:
  default_policy:
    cur_lr: 0.0005000000237487257
    max_q: 12.753413200378418
    mean_q: 5.389695167541504
    mean_td_error: -1.9424524307250977
    min_q: 0.2038756161928177
    model: {}
num_steps_sampled: 4480
num_steps_trained: 800
num_target_updates: 7
iterations_since_restore: 4
node_ip: 127.0.0.1
num_healthy_workers: 35
off_policy_estimator: {}
perf:
cpu_util_percent: 7.4
ram_util_percent: 11.4
pid: 92844
policy_reward_max: {}
policy_reward_mean: {}
policy_reward_min: {}
sampler_perf:
mean_env_wait_ms: 0.04555504737231483
mean_inference_ms: 0.8442203457250892
mean_processing_ms: 0.1175659340957767
time_since_restore: 8.04318118095398
time_this_iter_s: 1.7380259037017822
time_total_s: 8.04318118095398
timers:
learn_throughput: 19616.739
learn_time_ms: 1.631
update_time_ms: 89.14
timestamp: 1619709360
timesteps_since_restore: 0
timesteps_total: 4480
training_iteration: 4
trial_id: b665a_00000
```
Obviously, RLlib here utilized the cardinality of CPU cores on the node (36/36). 

You may consider odd the fact that the `total time(s)` here is more than when using a single CPU core, but this happens because in the latter case the algorithm runs 36 instances of the OpenAI Gym environment concurrently, rather than a single instance. Therefore, more data is collected for policy training, which can lead to faster reward convergence. In any case, these times are not absolute, and may decrease during training, especially if the agent achieves reward convergence.

## Metadata

When you run RLlib experiments, a directory named `ray_results` will automatically appear on your `home` directory. There you can find subdirectories for all your experiments that contain metadata distilled from all this information you see in the training printouts, and later use for evaluating the training process. 

After your experiment with `CartPole-v0` is finished, go to your home directory:
```
cd ~/
```
where 
Then, do `cd ray_results`. There, you will see directories named after the OpenAI Gym environment you used for running experiments. Hence, for CartPole you will see a directory named `CartPole-v0`. Within this directory, you will find subdirectories with names being combinations of the RL algorithm that you used for training, the OpenAI Gym environment's name, the datetime when the experiment took place, and a unique string. 

So, if for example you ran an experiment for CartPole, using Deep Q-Network (DQN), and the experiment started on April 29, 2021, at 9:14:57AM, the subdirectory containing the metadata will have a name like this:
```
DQN_CartPole-v0_0_2021-04-29_09-14-573vmq2rio
```
You `cd` in that directory, where you will find various text, JSON, and CSV files. One of them, named `progress.csv` contains a dataframe with columns such as `episode_reward_mean`, that help you evaluate the quality of the training process.

## Comparisons

Let us now compare the outcomes from when running experiments on a single core versus on all cores on a single Eagle node. A first approach to do that is the values column `episode_reward_mean` in file `progress.csv`. These values will show you how fast (or not) the reward converged to the optimal value during agent training. The faster the convergence, the better.

The following image shows the agent training progress, in terms of reward convergence, for the `CartPole-v0` environment. The RL algorithm used was the Proximal Policy Optimization (for more information see [here](https://arxiv.org/pdf/1707.06347.pdf)), and training was conducted for 100 iterations.
![](images/ppo_rew_comparison.png)
As you can see, training using the cardinality of CPU cores on a node led to faster convergence to the optimal value. 

It is necessary to say here that CartPole is a simple example where the optimal reward value (200) can be easily reached even when using a single CPU core on a local machine. The power of using multiple cores becomes more apparent in cases of more complex environments (such as the [Atari environments](https://gym.openai.com/envs/#atari)). RLlib website also gives examples of the scalability benefits for many RL algorithms ([here](https://docs.ray.io/en/master/rllib-algorithms.html#ppo)).

# Run experiments on multiple nodes

There are some cases where the problem under consideration is highly complex and requires vast amounts of training data for the policy network to train in a reasonable amount of time. It could be then, that you will require more than one nodes to run your experiments. In this case, you need to write a batch script file, where you will include all the necessary commands to train your agents on multiple CPUs and multiple nodes.

## Example: CartPole-v0

As explained above, CartPole is a rather simple environment and solving it using multiple cores on a single node feels like an overkill, let alone multiple nodes! However, it is a good example for giving you an experience on running RL experiments on Eagle.

For multiple nodes it is more convenient to use a batch script instead of an interactive node, which you will submit as `sbatch <name_of_your_batch_script>`. The results will be exported in an `slurm-<job_id>.out` file, which you can dynamically access during training using the `tail -f slurm-<job_id>.out` command. Otherwise, you can open it using a standard text editor (e.g. `nano`) after training is finished.
This tutorial will give you the basic parts of the batch script file. You can find the complete script [here](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/multi_node_trainer.sh).

You begin by defining some basic `SBATCH` options, including the desired training time, number of nodes, tasks per node, etc.

```bash
#!/bin/bash --login

#SBATCH --job-name=cartpole-multiple-nodes
#SBATCH --time=00:10:00
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --account=<your_account>
env
```

We want to run our agent training for 20 minutes (`SBATCH --time=00:20:00`), and on three Eagle CPU nodes (`SBATCH --nodes=3`). Every node will execute a single task (`SBATCH --tasks-per-node=1`), which will be executed on all 36 cores (`SBATCH --cpus-per-task=36`). Then, you need to set is the project account. You can always add more options, such as whether you want your experiment prioritized (`--qos=high`).

Afterwards, activate your environment. Do not forget to `unset LD_PRELOAD`.
```batch
module purge
conda activate /scratch/$USER/conda-envs/env_example
unset LD_PRELOAD
```

Now comes the part where you have to set up the Redis server that will allow all the nodes you requested to communicate with each other. For that, you have to set a Redis password:
```batch
ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address)
port=6379
ip_head=$ip_prefix:$port
redis_password=$(uuidgen)
```
Then, you submit your jobs one at a time at your workers, starting with the head node and moving on to the rest of them.
```batch
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head \
--node-ip-address="$ip_prefix" --port=$port --redis-password=$redis_password &
sleep 10

echo "starting workers"
for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  echo "i=${i}, node2=${node2}"
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address "$ip_head" --redis-password=$redis_password &
  sleep 5
done
```
Finally, you set your Python script to run. Since this experiment will run on a cluster, Ray will be initialized as:
```python
ray.init(_redis_password=args.redis_password, address=os.environ["ip_head"])
num_cpus = args.num_cpus - 1
```
Therefore, you need to activate the `--redis-password` option from your input arguments, along with the total number of CPUs. You do this as:
```batch
python -u simple_trainer.py --redis-password $redis_password --num-cpus $total_cpus
```
You are ready to start your experiment! Just run:
```
sbatch <your_slurm_file>
```

# Experimenting using GPUs (under development)

It is now time to learn running experiments utilizing also GPU nodes on Eagle. This can boost your training times considerably. GPU nodes however is better to be utilized only on cases of highly complex environments with very large observation and/or action spaces. In this tutorial we will continue with CartPole for establishing a template which you can later use for your own experiments.

## Creating Anaconda Environment

As expected, you first have to create a new environment, this time installing `Tensorflow-gpu`. This is the specialized Tensorflow distribution that is able to recognize and utilize GPU hardware in your system. For your convenience, we provide a sample [yaml file](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/simple-example-gpu/env_example_gpu.yml) that is tuned to create an Anaconda environment on Eagle with Tensorflow-gpu in it. For installing the new environment, follow the same process as before.

## Allocate GPU node

After the environment is successfuly created, you need to allocate a GPU node on Eagle. For this example, you will use an interactive node:
```
srun -n1 -t20 --gres=gpu:2 -<account_name> --pty $SHELL
```
