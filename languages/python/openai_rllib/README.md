## Create Anaconda environment

At first, create an Anaconda environment that you will use for all your experiments. For your convenience, simply follow the next steps:

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

The repo [provides](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/env_example.yml) the `env_example.yml` file. Use it to create a new Anaconda environment at a directory of your choosing. There are three main directories on Eagle where you can install the new environment, namely `/home`, `/scratch`, and `/projects`. Please go to [NREL HPC resources page](https://nrel.github.io/HPC/) to find more information about [the various Eagle directories](https://nrel.github.io/HPC/languages/python/NREL_python.html) and [how to create new Anaconda environments](https://nrel.github.io/HPC/languages/python/conda.html).

***Example:***

Begin by createing a subdirectory `/scratch/$USER/github-repos/`, `cd` there and clone the repo. Assuming you want to install your new environment in your `scratch` directory, you may want to create a directory that will contain all your Anaconda environments, e.g. `/scratch/$USER/conda-envs/`:
```
conda env create --prefix=/scratch/$USER/conda-envs/myenv -f env_example.yml
```

### 3<sup>rd</sup> step: Run OpenAI Gym on a single node/single core

After installation is complete, make sure everything is working correctly. You can test your installation by running a small example using one of the standard Gym environments (e.g. `CartPole-v0`).

Activate the enironment and start a Python session
```
module purge
conda activate /scratch/$USER/conda-envs/myenv
python
```
Then, run the following:
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
*Note that the above process does not involve any training.*

### Install more packages

Later, when you will start running reinforcement learning examples on Eagle, you will need to install other packages, most important of which the `Ray RLlib` library. This will enable you to run multiple instances of Gym in parallel over multiple cores per node, or even multiple nodes. You can always install new packages via:

```
conda install -c conda-forge <package_name>
pip install <package_name>
```
