# Experimenting using GPUs

It is now time to learn running experiments using GPU nodes on Eagle that can boost training times considerably. GPU nodes however is better to be utilized only in cases of environments with very large observation and/or action spaces. CartPole will be used again for establishing a template.

## Creating Anaconda environment

First thing to do is to create a new environment, this time installing `Tensorflow-GPU`. This is the specialized Tensorflow distribution that is able to recognize and utilize GPU hardware in a computer system. For convenience, the repo provides a validated sample [yaml file](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/simple-example-gpu/env_example_gpu.yml) that is tuned to create an Anaconda environment on Eagle with Tensorflow-GPU in it. For installing the new environment, follow the same process as before:
```
conda env create --prefix=/<path_to_chosen_directory>/env_example_gpu -f env_example_gpu.yml 
```

### **Only for Eagle users:** Creating Anaconda environment using Optimized Tensorflow

NREL's HPC group has recently created [a set of optimized Tensorflow drivers](https://github.com/NREL/HPC/tree/code-examples/workshops/Optimized_TF) that maximize the efficiency of utilizing Eagle's Tesla V100 GPU units. The drivers are created for various Python 3 and Tensorflow 2.x.x versions. 

The repo provides an [Anaconda environment version](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/simple-example-gpu/env_example_optimized_tf.yml) for using these drivers. This environment is based on one of the [example environments](https://github.com/NREL/HPC/blob/code-examples/workshops/Optimized_TF/py37tf22.yml) provided in the [Optimized TF repo](https://github.com/NREL/HPC/tree/code-examples/workshops/Optimized_TF).

**The provided Anaconda environment currently works for Python 3.7, Tensorflow 2.2, and the latest Ray version**

*Make sure to follow the [instructions for installing this particular environment](https://github.com/NREL/HPC/tree/code-examples/workshops/Optimized_TF) explicitly!*

## Allocate GPU node 

*The following instructions are the same for both regular and Optimized TF versions of the Anaconda environments*

Running experiments with combined CPU and GPU nodes is not so straightforward as running them using only CPU nodes (either single or multiple nodes). Particularly, heterogenous jobs using slurm have to be submitted.

Begin at first by specifying some basic options, similarly to previous section:
```batch
#!/bin/bash  --login

#SBATCH --account=A<account>
#SBATCH --job-name=cartpole-gpus
#SBATCH --time=00:10:00
```
The slurm script will clearly define the various jobs. These jobs include the CPU nodes that will carry the environment rollouts, and the GPU node for policy learning. Eagle has 44 GPU nodes and each node has 2 GPUs. Either request one GPU per node (`--gres=gpu:1`), or both of them (`--gres=gpu:2`). For the purposes of this tutorial, one GPU core on a single node is utilized.

In total, slurm nodes can be categorized as: 
 * A head node, and multiple rollout nodes (as before)
 * A policy training node (GPU)

Include the `hetjob` header for both the rollout nodes and the policy training node. Three CPU nodes are requested to be used for rollouts and a single GPU node is requested for policy learning:
```batch
# Ray head node
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

# Rollout nodes - Nodes with multiple runs of OpenAI Gym 
#SBATCH hetjob
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36

# Policy training node - This is the GPU node
#SBATCH hetjob
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
```
Of course, any number of CPU/GPU nodes can be requested, depending on problem complexity. 

As an example, a single node and perhaps just a single CPU core may be requested. Now, it is more reasonable to request GPUs for an OpenAI Gym environment that utilizes high-dimensional observation and/or action spaces. Hence, the first priority would be to start with multiple CPU nodes, and request GPUs only if they are needed.

For the three types of nodes (head, rollouts, training), define three separate groups:
```batch
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)
rollout_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1)
rollout_nodes_array=( $rollout_nodes )
learner_node=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_2)
echo "head node    : "$head_node
echo "rollout nodes: "$rollout_nodes
echo "learner node : "$learner_node
```
Each group of nodes requires its separate `srun` command so that they will run independently of each other.
```batch
echo "starting head node at $head_node"
srun --pack-group=0 --nodes=1 --ntasks=1 -w $head_node ray start --block --head \
--node-ip-address="$ip_prefix" --port=$port --redis-password=$redis_password & # Starting the head
sleep 10

echo "starting rollout workers"
for ((  i=0; i<$rollout_node_num; i++ ))
do
  rollout_node=${rollout_nodes_array[$i]}
  echo "i=${i}, rollout_node=${rollout_node}"
  srun --pack-group=1 --nodes=1 --ntasks=1 -w $rollout_node \
   ray start --block --address "$ip_head" --redis-password=$redis_password & # Starting the workers
  sleep 5
done

echo "starting learning on GPU"
srun --pack-group=2 --nodes=1 --gres=gpu:1 -w $learner_node ray start --block --address "$ip_head" --redis-password=$redis_password &
```
The slurm commands for the head and rollout nodes are identical to those from the previous section. A third command is also added for engaging the GPU node.

Finally, call
```batch
python -u simple_trainer.py --redis-password $redis_password --num-cpus $rollout_num_cpus --num-gpus 1
```
to begin training. Add the `---num-gpus` argument to include the requested GPU node (or nodes in case of `--gres=gpu:2`) for policy training. There is no need to manually declare the GPU for policy training in the `simple_trainer.py`, RLlib will automatically recognize the available GPU and use it accordingly.

The repo contains the complete slurm file versions for both [`env_example_gpu`](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/simple-example-gpu/gpu_trainer.sh) and [`env_gpu_optimized_tf`](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/simple-example-gpu/env_example_optimized_tf.yml), and they can be used as templates for future projects.

## Outputs

The output presents the same data as when running CPU-only experiments.

The difference is that now a GPU is also utilized (`1.0/1 GPUs`). This is the easiest way to make sure that a GPU was indeed allocated:
```
== Status ==
Memory usage on this node: 6.5/92.8 GiB
Using FIFO scheduling algorithm.
Resources requested: 108.0/180 CPUs, 1.0/1 GPUs, 0.0/807.51 GiB heap, 0.0/294.94 GiB objects (0.0/1.0 accelerator_type:V100)
Result logdir: /scratch/eskordil/ray_results/CartPole-v0
Number of trials: 1/1 (1 RUNNING)
+-----------------------------+----------+------------------+--------+------------------+-------+----------+----------------------+----------------------+--------------------+
| Trial name                  | status   | loc              |   iter |   total time (s) |    ts |   reward |   episode_reward_max |   episode_reward_min |   episode_len_mean |
|-----------------------------+----------+------------------+--------+------------------+-------+----------+----------------------+----------------------+--------------------|
| PPO_CartPole-v0_0339b_00000 | RUNNING  | 10.148.8.36:3651 |	   1 |          11.9292 | 21400 |  22.1782 |                   84 |                    8 |            22.1782 |
+-----------------------------+----------+------------------+--------+------------------+-------+----------+----------------------+----------------------+--------------------+


Result for PPO_CartPole-v0_0339b_00000:
  agent_timesteps_total: 42800
  custom_metrics: {}
  date: 2021-05-12_10-00-28
  done: false
  episode_len_mean: 47.06188118811881
  episode_media: {}
  episode_reward_max: 200.0
  episode_reward_mean: 47.06188118811881
  episode_reward_min: 9.0
  episodes_this_iter: 404
  episodes_total: 1302
  experiment_id: f5a33b1e020c4ef19dd38f1ab425b16d
  hostname: r103u23
  info:
    learner:
      default_policy:
        learner_stats:
          cur_kl_coeff: 0.30000001192092896
          cur_lr: 4.999999873689376e-05
          entropy: 0.5984720587730408
          entropy_coeff: 0.0
          kl: 0.01920320838689804
          model: {}
          policy_loss: -0.030758701264858246
          total_loss: 394.77838134765625
          vf_explained_var: 0.14716656506061554
          vf_loss: 394.8033447265625
    num_agent_steps_sampled: 42800
    num_steps_sampled: 42800
    num_steps_trained: 42800
  iterations_since_restore: 2
  node_ip: 10.148.8.36
  num_healthy_workers: 107
  off_policy_estimator: {}
  perf:
    cpu_util_percent: 6.346666666666667
    ram_util_percent: 3.200000000000001
  pid: 3651
  policy_reward_max: {}
  policy_reward_mean: {}
  policy_reward_min: {}
  sampler_perf:
    mean_action_processing_ms: 0.04437368377214986
    mean_env_render_ms: 0.0
    mean_env_wait_ms: 0.056431942853082055
    mean_inference_ms: 0.7302830909252791
    mean_raw_obs_processing_ms: 0.06735290716566901
  time_since_restore: 22.6750385761261
  time_this_iter_s: 10.745835304260254
  time_total_s: 22.6750385761261
  timers:
    learn_throughput: 1949.331
    learn_time_ms: 10978.124
    sample_throughput: 92888.12
    sample_time_ms: 230.385
    update_time_ms: 9.555
  timestamp: 1620835228
  timesteps_since_restore: 0
  timesteps_total: 42800
  training_iteration: 2
  trial_id: 0339b_00000
```