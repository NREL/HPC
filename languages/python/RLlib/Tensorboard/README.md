# Validating results using Tensorboard

Another way of visualizing the performance of agent training is with [**Tensorboard**](https://www.tensorflow.org/tensorboard). TensorBoard provides visualization and tooling needed for machine learning, deep learning, and reinforcement learning experimentation, for tracking and visualizing metrics such as loss and accuracy. 

Specifically for RL it is useful to visualize metrics such as:
 * Mean, min, and max reward values.
 * Episodes/iteration.
 * Estimated Q-values.
 * Algorithm-specific metrics (e.g. entropy for PPO).
 
To visualize results from Tensorboard, first `cd` to the `ray_results` directory:
```
cd ~/ray_results/
```
Every RL experiment generates a subdirectory named from the OpenAI Gym environment used in the experiment. 

E.g., after running all the examples previously shown in this tutorial, `ray_results` will have a subdirectory named `CartPole-v0`. Within, every experiment using CartPole generates a new subdirectory.

For the purpose of this tutorial, `cd` to the `CartPole-v0` subdirectory and activate one of the environments:
```
module purge
conda activate <your_environment>
```
Then, initialize Tensorboard as:
```
tensorboard --logdir=. --port 6006
```
For a specific training instance, e.g. the `DQN_CartPole-v0_0_2021-04-29_13-49-56gv0j3u93`, do instead:
```
tensorboard --logdir=DQN_CartPole-v0_0_2021-04-29_13-49-56gv0j3u93 --port 6006
```
If everything works properly, terminal will show:
```
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.5.0 at http://localhost:6006/ (Press CTRL+C to quit)
```
Open a new Terminal tab and create a tunnel:
```
ssh -NfL 6006:localhost:6006 $USER@el1.hpc.nrel.gov
```
Finally, open the above localhost url (`http://localhost:6006/`) in a browser.

The `tune/episode_reward_mean` plot is essentialy the same as the figure plotted from data in the `progress.csv` file. The difference in the x-axis scale has a simple explanation. The `episode_reward_mean` column on the `progress.csv` file shows the reward progress on every training iteration, while the `tune/episode_reward_mean` plot on Tensorboard shows reward progress on every training episode (a single RLlib training iteration consists of thousands of episodes).