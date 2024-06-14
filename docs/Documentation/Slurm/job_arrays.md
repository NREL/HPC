---
layout: default
title: Submitting Job Arrays
has_children: false
---
# Submitting Job Arrays
Job arrays are typically used when a user wants to submit many similar jobs with different inputs. Job arrays are capable of submitting hundreds, and even thousands, of similar jobs together. Here, we will describe how to submit job arrays on Slurm and walk through an example from the [NREL HPC Slurm Examples](https://github.com/NREL/HPC/tree/master/slurm) directory. More details on job arrays can be found in the [Slurm documentation](https://slurm.schedmd.com/job_array.html).

To run through the example on your own, you will need to download [uselist.sh](https://github.com/NREL/HPC/blob/master/slurm/uselist.sh) from the main slurm directory along with [doarray.py](https://github.com/NREL/HPC/blob/master/slurm/source/doarray.py) and [invertc.c](https://github.com/NREL/HPC/blob/master/slurm/source/invertc.c) from the source folder.

## SBATCH Directives for Job Arrays
In order to submit a job array to Slurm, the SBATCH directives at the top of your script or sbatch command line submission must contain the flag `--array={ARRAY_VALS}`, where `ARRAY_VALS` is a list or range of numbers that will represent the index values of your job array. For example:
```
# SBATCH --array=0-12  # Submits a job array with index values between 0 and 12
...

# SBATCH --array=2,4,6,10  # Submits a job array with index values 2, 4, 6, and 10
...

# SBATCH --array=1-43:2  # Submits a job array with index values between 1 and 43 with a step size of 2

```
