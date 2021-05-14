#!/bin/bash --login

# Basic options
#SBATCH --account=rlldrd
#SBATCH --job-name=cartpole-gpus
#SBATCH --time=00:10:00

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

env

## Never forget to unset LD_PRELOAD before executing your code
module purge
conda activate /projects/rlldrd/eskordil/envs/env_example_gpu_2
unset LD_PRELOAD

# Get nodes 
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_0)
rollout_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_1)
rollout_nodes_array=( $rollout_nodes )
learner_node=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_2)
echo "head node    : "$head_node
echo "rollout nodes: "$rollout_nodes
echo "learner node : "$learner_node

rollout_node_num=$(( $SLURM_JOB_NUM_NODES_HET_GROUP_1 ))
rollout_num_cpus=$(( $rollout_node_num * $SLURM_CPUS_ON_NODE ))
echo "rollout num cpus: "$rollout_num_cpus

ip_prefix=$(srun --pack-group=0 --nodes=1 --ntasks=1 -w $head_node hostname --ip-address) # Making address
port=6379
ip_head=$ip_prefix:$port
redis_password=$(uuidgen)
echo "ip_prefix: "$ip_prefix
echo "suffix: "$port
echo "ip_head: "$ip_head
echo "redis_password: "$redis_password

export ip_head # Exporting for latter access by trainer.py

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

sleep 10

echo "calling simple_trainer.py"
python -u simple_trainer.py --redis-password $redis_password --num-cpus $rollout_num_cpus --num-gpus 1
