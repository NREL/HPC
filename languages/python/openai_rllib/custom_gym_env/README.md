# Create your own Gym environment

If you want to create your own Gym environment and use it with RLlib, you begin with creating a Python `class` that contains the three main Gym functions:
 * `__init__`: Necessary to define `self.action_space` and `self.observation_space`. They represent the dimensionality of the observation and action spaces of your environment
 * `reset`: Resets the environment on a starting state.
 * `step`: Contains the environment mechanics. It takes as input the sampled action, and returns the new state, the current reward, and other information.

See the `custom_env.py` file for a (very) simple custom-made Gym environment. Use it as template for your own environments.

# Run experiments on RLlib

For running experiments using RLlib, you need to register your new environment first. Ray provides specialized functionality for that:
```python
from ray.tune.registry import register_env
```
The only addition to the trainer script is
```python
env_name = "custom-env"
register_env(env_name, lambda config: BasicEnv())
```

It requires a name of your choosing for the environment. For registering the environment, the two arguments required are its name and a configuration that involves the actual environment (here is the `BasicEnv`), as well as any environment hyperparameters (if they exist).

Finally, make sure the arguments `name` and `config["env"]` of the `tune.run` function are equal to the `env_name`.