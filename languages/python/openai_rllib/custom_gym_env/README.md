# Create Gym environments from scratch

So far, only benchmark Gym environments were used in order to demonstrate the processes for running experiments. It is time now to see how one can create their own Gym environment, carefully tailor-made to one's needs. OpenAI Gym functionality allows the creation of custom-made environments using the same structure as the benchmark ones. 

Custom-made environments can become extremely complex due to the mechanics involved and may require many subscripts that perform parts of the simulation. Nevertheless, the basis of all environments is simply a Python class that inherits the `gym.Env` class, where the user can implement the three main Gym functions and define any hyperapameters necessary:
 * `def __init__(self)`: Initializes the environment. It defines initial values for variables/hyperparameters and may contain other necessary information. It also defines the dimensionality of the problem. Dimensionality is expressed at the sizes of the observation and action spaces, which are given using the parameters `self.observation_space` and `self.action_space`, respectively. Depending on their nature, they can take discrete, continuous, or a combination of values. OpenAI provides [detailed examples](https://gym.openai.com/docs/) of each one of these types of spaces.
 * `def reset(self)`: When called, it *resets* the environment on a previous state (hence the name). This state can either be a user-defined initial state or it may be a random initial position. The latter can be found on environments that describe locomotion like `CartPole`, where the initial state can be any possible position of the pole on the cart.
 * `def step(self, action)`: The heart of the class. It defines the inner mechanics of the environment, hence it can be seen as some kind of simulator. Its main input is the sampled action, which when acted upon moves the environment into a new state and calculates the new reward. The new state and reward are two of the function's output and they are necessary for policy training since they are also inputs to the policy network. Other outputs include a boolean variable `done` that is **True** when the environment reaches its final state (if it exists), and **False** otherwise<sup>*</sup>, as well as a dictionary (`info`) with user-defined key-value objects that contain further information from the inner workings of the environment. 
 
<sup>*</sup>*Many environments do not consider a final state, since it might not make sense (e.g. a traffic simulator for fleets of autonomous ridesharing vehicles that reposition themselves based on a certain criterion. In this case the reward will get better every time, but there is no notion of a final vehicle position).*

Directions of how to create and register a custom-made OpenAI Gym environment are given below.

## Create an environment class

As stated above, the basis of any Gym environment is a Python class that inherits the `gym.Env` class. After importing the gym package, define the class as:
```python
import gym

class BasicEnv(gym.Env):(...)
```
The example environment is very simple and is represented by two possible states (0, 1) and 5 possible actions (0-4). For the purposes of this tutorial, consider state 0 as the initial state, and state 1 as the final state.

Define the dimensions of observation and action spaces in the `def __init__(self)` function:
```python
def __init__(self):
    self.action_space = gym.spaces.Discrete(5) # --> Actions take values in the 0-4 interval
    self.observation_space = gym.spaces.Discrete(2) # --> Two possible states [0,1]
```
Both spaces take discrete values, therefore they are defined using Gym's `Discrete` function. Other possible functions are `Box` for continuous single- or multi-dimensional observations and states, `MultiDiscrete` for vectors of discrete values, etc. OpenAi provides [detailed explanation](https://gym.openai.com/docs/) for all different space forms.

Next, define the `def reset(self)` function:
```python
def reset(self):
    state = 0
    return state
```
In this example, the reset function simply returns the environment to the initial state.

Finally, define the `def step(self, action)` function, which takes as input the sampled action. Here the step function takes the environment at state 1 and based on the action, returns a reward of 1 or -1:
```python
def step(self, action):
    state = 1

    if action == 2:
        reward = 1
    else:
        reward = -1

    done = True
    info = {}

    return state, reward, done, info
```
That's it, the new Gym environment is ready! Make note that there is one more function usually found on Gym environments. This is the `def render(self)` function, and is called in random intervals throughout training returning a "snapshot" of the environment at that time. While this is helpful for evaluating the agent training process, it is not necessary for the actual training process. OpenAI documentation [provides](https://gym.openai.com/docs/#environments) details for every one of these functions.

You can find the [full script](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/custom_gym_env/custom_env.py) of this environment in the repo.

## Run experiments on RLlib
Let's now train the agent with RLlib. The [full trainer script](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/custom_gym_env/custom_env_trainer.py) is given at the repo.

The trainer is almost identical to [the one used before](https://github.com/erskordi/HPC/blob/HPC-RL/languages/python/openai_rllib/simple-example/simple_trainer.py), with few additions that are necessary to register the new environment.

At first, along with `ray` and `tune`, import:
```python
from ray.tune.registry import register_env
from custom_env import BasicEnv
```
The `register_env` function is used to register the new environment, which is imported from the `custom_env.py`.

Function `register_env` takes two arguments:
* Training name of the environment, chosen by the developer.
* Actual name of the environment (`BasicEnv`) in a `lambda config:` function.
```python
env_name = "custom-env"
register_env(env_name, lambda config: BasicEnv())
```
Once again, RLlib provides [detailed explanation](https://docs.ray.io/en/code-examples/rllib-env.html) of how `register_env` works.

The `tune.run` function, instead of `args.name_env`, it uses the `env_name` defined above.

That's all! Proceed with agent training using any of the slurm scripts provided by the repo.

As a final note, creating custom-made OpenAI Gym environment is more like an art than science. The main issue is to really clarify what the environment represents and how it works, and then define this functionality in Python.