# Agent training with Ray/RLlib

RL algorithms are notorious for the amount of data they need to collect in order to learn policies. The more data collected, the better the training will be. The best way to do it is to run many Gym instances in parallel and collecting experience, and this is where RLlib assists.

[RLlib](https://docs.ray.io/en/master/rllib.html) is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. It supports all known deep learning frameworks such as Tensorflow, Pytorch, although most parts are framework-agnostic and can be used by either one.

The RL policy learning examples provided in this tutorial demonstrate the RLlib abilities. For convenience, the `CartPole-v0` OpenAI Gym environment will be used.

The most straightforward way is to create a Python "trainer" script. It will call the necessary packages, setup flags, and run the experiments, all nicely put in a few lines of Python code.

## Import packages

Begin trainer by importing the `ray` package:
```python
import ray
from ray import tune
```
`Ray` consists of an API readily available for building [distributed applications](https://docs.ray.io/en/master/index.html). On top of it, there are several problem-solving libraries, one of which is RLlib.

`Tune` is another one of `Ray`'s libraries for scalable hyperparameter tuning. All RLlib trainers (scripts for RL agent training) are compatible with Tune API, making experimenting easy and streamlined.

Import also the `argparse` package and setup some flags. Although that step is not mandatory, these flags will allow controlling of certain hyperparameters, such as:
* RL algorithm utilized (e.g. PPO, DQN)
* Number of CPUs/GPUs
* ...and others
```python
import argparse
```

## Create flags
Begin by defining the following flags:
```python
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--name-env", type=str, default="CartPole-v0")
parser.add_argument("--run", type=str, default="DQN")
parser.add_argument("--local-mode", action="store_true")
```
All of them are self-explanatory, however let's see each one separately.
1. `--num-cpus`: Defines the number of CPU cores used for experience collection (Default value 0 means allocation of a single CPU core).
2. `--num-gpus`: Allocates a GPU node for policy learning (works only for Tensorflow-GPU). Except whole values (1,2,etc.), it also accepts partial values, in case 100% of the GPU is not necessary.
3. `--name-env`: The name of the OpenAI Gym environment.
4. `--run`: Specifies the RL algorithm for agent training.
5. `--local-mode`: Helps defining whether experiments running on a single core or multiple cores.

## Initialize Ray

Ray is able to run either on a local mode (e.g. laptop, personal computer), or on a cluster. Depending on the nature of the experiment, distinguish between the two modes using an `if-else` statement:
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
*Do not worry about the `redis_password` for now, it will be explained later in the tutorial*

For the first experiment, only a single core is needed. Therefore, setup ray to run on a local mode: `ray.init(local_mode=args.local_mode)`. The next line shows the number of CPU cores to be used. Remember that RLlib always allocates one CPU core, even when `--num-cpus=0`. Hence, always subtract one from the total number of cores.

## Run experiments with Tune

This is the final step in this basic trainer. Tune's `tune.run` function initiates the agent training process. There are three main arguments in this function:
* RL algorithm (string): It is defined in the `--run` flag (PPO, DQN, etc.).
* `stop` (dictionary): Provides a criterion to stop training (in this example is the number of training iterations; stop training when iterations reach 10,000).
* `config` (dictionary): Basic information for training, contains the OpenAI Gym environment name, number of CPUs/GPUs, and others.
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
That's it! The RLlib trainer is ready!

Note here that, except default hyperparameters like those above, [every RL algorithm](https://docs.ray.io/en/master/rllib-algorithms.html#available-algorithms-overview) provided by RLlib has its own hyperparameters and their default values that can be tuned in advance.

The code of the trainer in this example can be found [in the repo](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/simple-example/simple_trainer.py).


# Run experiments on Eagle

The next step calls for running the trainer script on Eagle.

## Allocate an interactive Eagle node

First, allocate an interactive node. Let's start by allocating a `debug` node. Debug nodes have a maximum allocation time of one hour (60 minutes), but they are easier to allocate than regular nodes:
```
srun -n1 -t10 -<project_name> --partition debug --pty $SHELL
```
Successful node allocation is shown as:
```
srun: job 6896853 queued and waiting for resources
srun: job 6896853 has been allocated resources
```
Activate the Anaconda environment:
```
module purge
conda activate env_example
```
**VERY IMPORTANT!!** 
Before the experiment, run
```
unset LD_PRELOAD
```
For communication between cores in a node (and between nodes in multi-node experiments), RLlib uses a Redis server. However, there is some kind of process running on Eagle causing Redis server to malfunction. Therefore, make sure you unset variable `LD_PRELOAD`, which disables that process and lets your experiment run smoothly.

## Run multi-core experiments

The previous example is designed to run on a single CPU core. However, as explained above, RL training is highly benefited from running multiple concurrent OpenAI Gym rollouts. A single node on Eagle has 36 CPU cores, therefore use any number of those in order to speed up your agent training. 

For all 36 cores, adjust the `--num-cpus` hyperparameter to reflect to all CPUs on the node:
```
python simple_trainer.py --num-cpus 35
```
Again, RLlib by default utilizes a single CPU core, therefore by putting `--num-cpus` equal to 35 means that all 36 cores are requested.

Such is not the case with the `num_gpus` key, where zero means no GPU allocation is permitted. This is because GPUs are used for policy training and not running the OpenAI Gym environment instances, thus they are not mandatory (although having a GPU node can assist the agent training by reducing training time).


## Run multi-node/multi-core experiments

Problems that involve highly complex environments with very large observation and/or action state spaces will probably require running experiments utilizing more than one Eagle nodes. In this case it is better to work with slurm scripts. You can submit such scripts as 
```
sbatch <name_of_your_batch_script>
```
The results are exported in an `slurm-<job_id>.out` file. This file can be accesssed:
 * During training (`tail -f slurm-<job_id>.out`) 
 * Open it using a standard text editor (e.g. `nano`) after training is finished.

An example of a `slurm-<job_id>.out` file is also included in the repo for reference.

## Example: CartPole-v0

As explained above, CartPole is a rather simple environment and solving it using multiple cores on a single node feels like an overkill, let alone multiple nodes! However, it is a good example for giving you an experience on running RL experiments using RLlib.

For multiple nodes it is more convenient to use a slurm script instead of an interactive node. Slurm files are submitted as `sbatch <name_of_your_batch_script>`, and the results are exported in an `slurm-<job_id>.out` file. The `.out` file can be interactively accessed during training using the `tail -f slurm-<job_id>.out` command. Otherwise, after training, open it using a standard text editor (e.g. `nano`).
Next, the basic parts of the slurm script file are given. The repo also provides [the complete script](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/multi_node_trainer.sh).

The slurm file begins with defining some basic `SBATCH` options, including the desired training time, number of nodes, tasks per node, etc.

```bash
#!/bin/bash --login

#SBATCH --job-name=cartpole-multiple-nodes
#SBATCH --time=00:10:00
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --account=A<account>
env
```
Allocating multiple nodes means creating a Ray cluster. A Ray cluster consists of a head node and a set of worker nodes. The head node needs to be started first, and the worker nodes are given the address of the head node to form the cluster. 

The agent training will run for 20 minutes (`SBATCH --time=00:20:00`), and on three Eagle CPU nodes (`SBATCH --nodes=3`). Every node will execute a single task (`SBATCH --tasks-per-node=1`), which will be executed on all 36 cores (`SBATCH --cpus-per-task=36`). Then, define the project account. Other options are also available, such as whether to prioritize the experiment (`--qos=high`).

Afterwards, use the commands to activate the Anaconda environment. Do not forget to `unset LD_PRELOAD`.
```batch
module purge
conda activate /scratch/$USER/conda-envs/env_example
unset LD_PRELOAD
```
Next, set up the Redis server that will allow all the nodes you requested to communicate with each other. For that, set a Redis password:
```batch
ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address)
port=6379
ip_head=$ip_prefix:$port
redis_password=$(uuidgen)
```
Then, submit the jobs one at a time at the workers, starting with the head node and moving on to the rest of them.
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
Finally, set the Python script to run. Since this experiment will run on a cluster, Ray will be initialized as:
```python
ray.init(_redis_password=args.redis_password, address=os.environ["ip_head"])
num_cpus = args.num_cpus - 1
```
Therefore, `--redis-password` option must be active, along with the total number of CPUs:
```batch
python -u simple_trainer.py --redis-password $redis_password --num-cpus $total_cpus
```
The experiment is ready to begin, simply run:
```
sbatch <your_slurm_file>
```
If the trainer script is on a different directory, make sure to `cd` to this directory in the slurm script before executing it.
```
# Example where the trainer is on scratch:
cd /scratch/$USER/path_to_specific_directory
python -u simple_trainer.py --redis-password $redis_password --num-cpus $total_cpus
```

# Outputs (single-core/multiple-core)

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
Most of this information will not be necessary, however there are some parts that give an idea of the results quality.

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
These lines show the following:
* Number of CPUs utilized: If `--num-cpus` was defined as equal to zero, then by default RLlib acquired a single CPU core to run the experiment.
* No GPU resources were utilized: Expected since `--num-gpus` was set to zero.
* Trial name: Generated automatically, it gives information regarding the RL algorithm used (here DQN) and the OpenAI Gym environment.
* status: either RUNNING or FAILED.
* iter: Current training iteration.
* total time (seconds): time spend per iteration.
* reward: The reward returned after the end of this iteration. Succesfull agent training will be shown through increase of that value during training.


This is the most basic, but in the same time important, information after every training iteration. The rest of this snippet contains more specialized information, but most of it will not be relevant in subsequent analysis. 

When `--num-cpus` equals 35, then the aforementioned printout will be:
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
Obviously, RLlib here utilized all CPU cores on the node (36/36). 

It might seem odd that the `total time(s)` here is slightly more than when using a single CPU core, but this happens because when the algorithm runs 36 instances of the OpenAI Gym environment concurrently, more data are collected for policy training leading to faster reward convergence. In any case, these times are not absolute, and will decrease during training, especially as the agent approaches the optimal reward value.

**Supplemental notes:**
As you noticed, when using RLlib for RL traning, there is no need to `import gym`, as we did in the non-training example, because RLlib recognizes automatically all benchmark OpenAI Gym environments. Even when you create your own custom-made Gym environments, RLlib provides proper functions with which you can register your environment before training.
