#!/bin/bash
#SBATCH --account=$HPC_HANDLE
#SBATCH --time=1:00:00
#SBATCH --job-name=rl_train
#SBATCH --nodes=10
#SBATCH --partition=short
#SBATCH --tasks-per-node=1

# Step 1. Loading conda env
module purge
module load anaconda3
conda activate /projects/$HPC_HANDLE/$USER/conda_envs/rl_hpc

# Step 2. Starting Ray Cluster
#node_num=2 # Must be one less that the total number of nodes
node_num=$(( $SLURM_JOB_NUM_NODES - 1 ))
worker_num=$(( $SLURM_JOB_NUM_NODES * $SLURM_CPUS_ON_NODE - 1 ))
echo "node_num="$node_num
echo "worker_num="$worker_num

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

echo "nodes: "$nodes
echo "nodes_array: "$nodes_array

node1=${nodes_array[0]}
echo "node1: "$node1

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)
echo "ip_prefix: "$ip_prefix
echo "suffix: "$suffix
echo "ip_head: "$ip_head
echo "redis_password: "$redis_password

# export ip_head # Exporting for latter access by trainer.py

echo "starting head"
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --port=6379 --redis-password=$redis_password --temp-dir="/tmp/scratch/ray"& # Starting the head
sleep 30

echo "starting workers"
for ((  i=1; i<=$node_num; i++ ))
do
  node2=${nodes_array[$i]}
  echo "i=${i}, node2=${node2}"
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password --temp-dir="/tmp/scratch/ray"& # Starting the workers
  sleep 5
done

sleep 20

# Step 3. Start Training
echo "Start training"

TIME=$(squeue -j $SLURM_JOB_ID -h --Format TimeLimit)

python -u train_script.py --run PPO --redis-password $redis_password --worker-num $worker_num --ip-head $ip_head
