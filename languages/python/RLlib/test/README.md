## Rollout trained policy

This section explains once an RL agent (or policy) is trained, how do we use it to rollout the environment, i.e., examine how the control is implemented.

### Set up configuration

```
# Use the same configuration as the one for training except the rollout worker
# number, which can be limited at 1.
config = (
    get_trainable_cls("PPO")
    .get_default_config()
    .framework("torch")
    .training(model={"fcnet_hiddens": [256, 256]})  # Same as the one in training
    .rollouts(num_rollout_workers=1)
)
```

Note, the number of rollout workers does not have to be the same as the value used for training, and it can be as low as 1. If setting to a high value, Ray will start that many rollout workers, which is unnecessary when rollout.

### Load trained RL agent

The trained RL agent can be restored as shown below:

```
agent = PPO(config, env=env_name)
checkpoint_path = ...
agent.restore(checkpoint_path)
```
See [the code](rollout.py) for detailed example of setting checkpoint path.

### Obtain action from the trained policy

The action can be retrieved by calling `agent.compute_single_action(state)`, here we used state and observation interchangably. By passing in the environment state/observation, the agent returns the action according to the trained policy.
