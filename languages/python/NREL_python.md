---
layout: default
title: NREL HPC Python
parent: Python
grand_parent: Programming Languages
---


# Python on NREL HPC
By design, the HPC is a time-shared multi-machine system which necessarily warrants some nuanced consideration for how environments are managed relative to a single machine with a single user. Sometimes, the default workflow for environment creation and usage is not the most optimal for some use-cases.

Below is a list of common pitfalls that users have encountered historically while using Python and Anaconda on NREL HPC. 

* Running a SLURM job that uses a `conda` environment which is stored in `$HOME`.
* Exhausting the `$HOME` storage quota (50GB on the current HPC system) usually because of conda's package cache combined with their user environments.
* Trying to share a `conda` environment from another user's `/home` directory.
* Forgetting to install `jupyter` in a new conda environment, resulting in using the `base` installation's version which doesn't have your dependencies installed.

Let's discuss strategies to mitigate or avoid these kinds of problems

## Installing Conda Environments in Different Directories
By default, `conda` will install new environments in `$HOME/.conda`. Generally speaking, this a sensible default&mdash;it just happens to be the starting point to frequent issues that users have experienced historically. Something to consider is that `conda` has a `--prefix` flag which allows one to arbitrate where a conda environment gets installed to, notably allowing you to place environments on other file-systems and block devices besides the `/home` network-storage that is mounted on NREL HPC systems.

For example, here is how one might create a project in their `/scratch` directory:


```python
ENV_PREFIX="/scratch/$USER/demo_scratch_env"

import os ; os.environ['ENV_PREFIX']=ENV_PREFIX  # Export this variable for cells below 
```


```python
!conda create --quiet --use-local --yes \
    --prefix $ENV_PREFIX   # `--prefix` in action \
    python=3.7
```

    Collecting package metadata: ...working... done
    Solving environment: ...working... done
    
    ## Package Plan ##
    
      environment location: /scratch/mbartlet/demo_scratch_env
    
    
    
    Preparing transaction: ...working... done
    Verifying transaction: ...working... done
    Executing transaction: ...working... done



```python
!ls -ld $ENV_PREFIX
```

    drwxr-xr-x. 3 mbartlet mbartlet 4096 Dec  3 11:10 /scratch/mbartlet/demo_scratch_env



```python
# Delete the demo environment for cleanliness
!conda-env remove --yes --quiet --prefix $ENV_PREFIX &>/dev/null
```

Below is a table which discusses the pros and cons of each block-device mount on NREL HPC as a location for storing your software environments.

| Block-device mounts | Situations where you would want to use this block device for your conda environments                                                                                                                                                  | Caveats to consider when using this mount                                                                                                                                                                                                                                                                                                                                           |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| /home                     | `$HOME/.conda` is the default location for environments. For one-off environments,  or if you don't create environments often, this is a reasonable location for your environments and doesn't require any extra flags or parameters. | Files in <span>\$</span>HOME will not be purged so long as you have an active NREL HPC account. However, <span>\$</span>HOME is limited to a 50GB storage quota so you may have to take care to monitor your storage footprint.                                                                                                                                                                                        |
| /scratch                  | `/scratch` or `/projects` is ultimately where you want your environment to end up if your jobs have more than 1 node&mdash;if your environment is in `/home` then every node in your job will be competing for read-access over a non-parallel network fabric to the source files of your environment. `/scratch` provides simultaneous access to all the nodes. A sensible approach is copying your environments from `/home` to `/scratch` as part of your job's initialization. | `/scratch` storage is unlimited. `/scratch` is a parallel filesystem, meaning simultaneous filesystem operations by several nodes is possible and performant. However, the contents of `/scratch` are subject to purge after 28 days of inactivity.                                                                                                                                        |
| /projects                 | This is a great place to put a conda environment that you anticipate sharing with your colleagues who are also working on the project. You can structure the permissions such that others in the project have read-only, write-only, or no access (we also encourage restoring these permissions at a later date so others on the project can manage your files without a hassle). | `/projects` is also a parallel filesystem which reaps the same benefits as mentioned above. However, access to projects is contingent on having access to an HPC project allocation. Moreover, the storage quota allotted to each project is relative to the reasonableness of its requested needs, although a conda environment is very unlikely to have a significant storage footprint. |

As mentioned above, let's demonstrate one might go about copying an environment from `/home` to `/scratch` in a SLURM job. The below cell will generate a nice code block based on variables used earlier in this notebook, as well as environment variables within your user account:


```python
# Acquire a default project handle to procedurally generate a SLURM job
import subprocess

command = "/nopt/nrel/utils/bin/hours_report | tail -1 | awk '{print $1}'" # Grab a valid project handle

command_array = [
    '/bin/bash',
    '-c',
    command
]

project_handle = subprocess.run(command_array, stdout=subprocess.PIPE).stdout.decode('utf-8')[:-1]

import os ; os.environ['DEFAULT_HANDLE'] = project_handle  # Export handle for cells below
```


```python
!echo $DEFAULT_HANDLE
```

    wks



```python
conda_home_env="py3"
```


```python
# Acquire info about the default conda environment
import subprocess

command = f"module load conda && . activate {conda_home_env} && echo $CONDA_PREFIX"

command_array = [
    '/bin/bash',
    '-lc',         # Have to run this from a login-shell
    command
]

conda_home_env_prefix = subprocess.run(command_array, stdout=subprocess.PIPE).stdout.decode('utf-8')[:-1]

import os ; os.environ['CONDA_HOME_ENV_PREFIX'] = conda_home_env_prefix  # Export handle for cells below
```


```python
!echo $CONDA_HOME_ENV_PREFIX
```

    /home/mbartlet/.conda/envs/py3



```python
from IPython.display import Markdown as md
from os import environ as env

SCRATCH_ENV=f"/scratch/{env['USER']}/home_conda_clone"

body=f"""
```bash
#!/usr/bin/env bash
#SBATCH --account {env['DEFAULT_HANDLE']}
#SBATCH --time 5
#SBATCH --partition debug
#SBATCH --nodes 2

export SCRATCH_ENV="{SCRATCH_ENV}"
rsync -avz --ignore-existing "{env['CONDA_HOME_ENV_PREFIX']}" "$SCRATCH_ENV" &>/dev/null

srun bash -l <<EOF
module purge
module load conda
. activate "$SCRATCH_ENV"
which python
EOF

rm -rf "$SCRATCH_ENV"  # Optional clean-up
```
"""

md(body)
```





```bash
#!/usr/bin/env bash
#SBATCH --account wks
#SBATCH --time 5
#SBATCH --partition debug
#SBATCH --nodes 2

export SCRATCH_ENV="/scratch/mbartlet/home_conda_clone"
rsync -avz --ignore-existing "/home/mbartlet/.conda/envs/py3" "$SCRATCH_ENV" &>/dev/null

srun bash -l <<EOF
module purge
module load conda
. activate "$SCRATCH_ENV"
which python
EOF

rm -rf "$SCRATCH_ENV"  # Optional clean-up
```




And after running what was generated above:
```bash
[mbartlet@el1 ~] $ cat slurm-1845968.out
/scratch/mbartlet/home_conda_clone/bin/python
/scratch/mbartlet/home_conda_clone/bin/python
```
Which shows both nodes sourced the environment from `/scratch`
