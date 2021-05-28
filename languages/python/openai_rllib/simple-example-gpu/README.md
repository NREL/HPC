# Running experiments on GPU

**Before proceeding:** Generally, RLlib is designed to use CPUs for running OpenAI Gym instances in order to collect experience, and GPUs for policy learning using the collected experience.

## Create new Ananconda environment with Tensorflow-GPU

You need to create a new environment, but this time you will use [Tensorflow-GPU](https://www.tensorflow.org/install/gpu), along with the proper CUDA and CUDNN drivers. This repo provides two `yaml` files for installing a GPU-oriented Anaconda environment on Eagle:
 * `env_example_gpu.yml`: Simple GPU-based environment.
 * `env_example_optimized_tf.yml`: Similar environment, this time using [Optimized TF drivers](https://github.com/NREL/HPC/tree/master/workshops/Optimized_TF).

**NOTE: Due to possible incosistencies between package versions, make sure that when you use updated versions of Tensorflow-GPU, Numpy, Pandas, etc. that their versions work well together. Both aforementioned environments generally perform bug-free.**

## Experiments using GPU for policy training

The process is similar to the one without using GPUs, you can use a slurm script with heterogenous jobs to distinguish between rollouts/experience collection (Gym iterations - CPUs) and policy learning (GPUs), see the following example:

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
You can find complete slurm file examples to use as template in this subdirectory.
