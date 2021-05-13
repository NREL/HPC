# Running experiments on GPU

Before proceeding, RLlib is not designed to use GPUs to run OpenAI Gym instances, but rather use it for policy network training, while at the same time allocating CPU cores (single-node/single-core, single-node/multi-core, multi-node/multi-core) for collecting training data through multiple iterations of the `reset`-`step` Gym functions.

## Create new Ananconda environment with Tensorflow-GPU

You need to create a new environment, this time including the [Tensorflow-GPU](https://www.tensorflow.org/install/gpu) package. Please use the provided `yaml` file for installing a GPU-oriented Anaconda environment on Eagle.

**NOTE: Due to possible incosistencies between package versions, make sure that when you use updated versions of Tensorflow-GPU, Numpy, Pandas, etc. that their versions work well together. The `env_example_gpu.yml` as it is now gives a combination that generally performs bug-free (unfortunately this can only be done through trial-and-error).**

## Experiments using GPU for policy training

The process is similar to the one without using GPUs, you can use a slurm script with heterogenous jobs to distinguish between rollouts (Gym iterations - CPUs) and policy training (GPUs), see the following example:

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
You can find a complete slurm file to use as template in this subdirectory.