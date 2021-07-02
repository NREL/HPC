---
layout: default
title: TensorBoard 
parent: Machine Learning
---


# Validating ML results using Tensorboard

[**Tensorboard**](https://www.tensorflow.org/tensorboard) provides visualization and tooling needed for machine learning, deep learning, and reinforcement learning experimentation:
 * Tracking and visualizing metrics such as loss and accuracy.
 * Visualizing the model graph (ops and layers).
 * Viewing histograms of weights, biases, or other tensors as they change over time.
 * Projecting embeddings to a lower dimensional space.
 * Displaying images, text, and audio data.
 * Profiling TensorFlow programs.

For RL it is useful to visualize metrics such as:
 * Mean, min, and max reward values.
 * Episodes/iteration.
 * Estimated Q-values.
 * Algorithm-specific metrics (e.g. entropy for PPO).
 
To visualize results from Tensorboard, first `cd` to the directory where your results reside. E.g., if you ran experiments using `ray`, then do the following:
```
cd ~/ray_results/
```

There are three main methods for activating Tensorboard:
* If you included Tensorboard installation in an Anaconda environment, simply activate it:
   ```
   module purge
   conda activate <your_environment>
   ```
* You can also install Tensorboard in userspace using `pip install`:
   ```
   pip install tensorboard --user
   ```
* Or, install using container images:
   ```
   ml singularity-container
   singularity pull docker://tensorflow/tensorflow
   singularity run tensorflow_latest.sif
   ```

Then, initialize Tensorboard using a pre-specified port number of your choosing (e.g. 6006, 8008):
```
tensorboard --logdir=. --port 6006 --bind_all
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
Finally, open the above localhost url (`http://localhost:6006/`) in a browser, where all the aforementioned plots will be shown.
