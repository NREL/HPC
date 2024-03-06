## Create Anaconda environment

Follow the following steps to create an Anaconda environment for this experiment:

### 1st step: Log in on Kestrel (Can be skipped if work on local computer)

Login on Kestrel with:
```
ssh kestrel
```
if you have hostname configured, or
```
ssh <username>@kestrel.hpc.nrel.gov
```

### 2nd step: Set up Anaconda environment

To use `conda` on Kestrel (different from Eagle), the Anaconda module needs to be loaded.
```
module purge
module load anaconda3
```

We suggest creating a conda environment on your `\projects` rather than `\home` or `\scratch`. (#TODO: Check this with HPC team.)

***Example:***

Use the following script to create a conda environment:
```
conda create --prefix=/projects/$HPC_HANDLE/$USER/conda_envs/rl_hpc python=3.10
```

Here, `$HPC_HANDLE` is the project handle and `$USER` is your HPC user name.

Activate the conda environment and install packages:

```
conda activate /projects/$HPC_HANDLE/$USER/conda_envs/rl_hpc

pip install -r requirements.txt
```

### 3rd step: Test OpenAI Gym API

After installation is complete, make sure everything is working correctly. You can test your installation by running a small example using one of the standard Gym environments (e.g. `CartPole-v1`).

Activate the enironment and start a Python session
```
module purge
module load anaconda3
conda activate /projects/$HPC_HANDLE/$USER/conda_envs/rl_hpc
python
```
Request an interactive session on Kestrel, and then, run the following:
```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    done = (terminated or truncated)
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

### 4th step: Test other libraries
The following libraries should also be imported without an error.

```
import ray
import torch
```
