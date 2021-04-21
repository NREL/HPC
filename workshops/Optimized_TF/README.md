# Install instructions for GPU use

1. Navigate to your eagle home directory or scratch directory using
```
cd ~/
```
or
```
cd /scratch/$USER/
```
2. Clone the github repo into the directory you chose
```
git clone https://github.nrel.gov/hpc-apps/Optimized_TF
```
3. Navigate to the repo
```
cd ./Optimized_TF/
```

4. To install TensorFlow 2.4.0 with Python 3.8 for GPUS run the following

  * a) load the appropriate modules
      ```
      module purge
      module use /nopt/nrel/apps/modules/test/modulefiles/
      module load conda
      module load gcc/7.4.0
      module load cudnn/8.0.5/cuda-10.2
      ```
  * b) build a predefined conda environment
      ```
      conda env create -f py38tf24.yml
      ```
  * c) Active the conda environment
      ```
      source activate py38tf24
      ```
  * d) Install the precompiled TensorFlow installation from a wheel
      ```
      pip install --upgrade --no-deps --force-reinstall /nopt/nrel/apps/wheels/tensorflow-2.4.0-cp38-cp38-linux_x86_64.whl
      ```
  * e) If you are an allocated or interactive node you can test the install by running
      ```
      python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
      ```

5. To install TensorFlow 2.3.2 with Python 3.8 for GPUS run the following

  * a) load the appropriate modules
      ```
      module purge
      module use /nopt/nrel/apps/modules/centos74/modulefiles/
      module load gcc/7.4.0
      module load cuda/10.0.130
      module load cudnn/7.4.2/cuda-10.0
      module load conda
      ```
  * b) build a predefined conda environment
      ```
      conda env create -f py38tf23.yml
      ```
  * c) Active the conda environment
      ```
      source activate py38tf23
      ```
  * d) Install the precompiled TensorFlow installation from a wheel
      ```
      pip install --upgrade --no-deps --force-reinstall /nopt/nrel/apps/wheels/tensorflow-2.3.2-cp38-cp38-linux_x86_64.whl
      ```
  * e) If you are an allocated or interactive node you can test the install by running
      ```
      python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
      ```

6. To install TensorFlow 2.2.x with Python 3.7 for GPUs run the following

  * a) load the appropriate modules
      ```
      module purge
      module use /nopt/nrel/apps/modules/centos74/modulefiles/
      module load gcc/7.4.0
      module load cuda/10.0.130
      module load cudnn/7.4.2/cuda-10.0
      module load conda
      ```
  * b) build a predefined conda environment
      ```
      conda env create -f py37tf22.yml
      ```
  * c)  Active the conda environment
      ```
      source activate py37tf22
      ```
  * d) Install the precompiled TensorFlow installation from a wheel
      ```
      pip install --upgrade --no-deps --force-reinstall /nopt/nrel/apps/wheels/tensorflow-2.2.1-cp37-cp37m-linux_x86_64.whl
      ```
      or
      ```
      pip install --upgrade --no-deps --force-reinstall /nopt/nrel/apps/wheels/tensorflow-2.2.1-cp37-cp37m-linux_x86_64.whl
      ```
  * e) If you are an allocated or interactive node you can test the install by running
      ```
      python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
      ```

7. To install TensorFlow 2.0.0 with Python 3.7 for GPUs run the following

  * a) load the appropriate modules
      ```
      module purge
      module use /nopt/nrel/apps/modules/centos74/modulefiles/
      module load gcc/7.3.0
      module load cuda/10.0.130
      module load cudnn/7.4.2/cuda-10.0
      module load conda
      ```
  * b) build a predefined conda environment
      ```
      conda env create -f py37tf20.yml
      ```
  * c)  Active the conda environment
      ```
      source activate py37tf20
      ```
  * d) Install the precompiled TensorFlow installation from a wheel
      ```
      pip install --upgrade --no-deps --force-reinstall /nopt/nrel/apps/wheels/tensorflow-2.0.0-cp37-cp37m-linux_x86_64.whl
      ```

  * e) If you are an allocated or interactive node you can test the install by running
      ```
      python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
      ```
