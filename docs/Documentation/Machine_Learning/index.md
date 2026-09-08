# Machine Learning

*Machine learning refers to a set of techniques and algorithms that enable computers to automatically learn from data and improve their performance on a specific task over time. Types of machine learning methods include, but are not limited to, supervised learning (algorithms trained on labeled datasets), unsupervised learning (algorithms trained on unlabeled datasets), and reinforcement learning (learning by trial and error). The Computational Science Center at NLR conducts research in these types of machine learning, and also supports the use of machine learning software on Kestrel.*

## Getting Started

This page covers running the two most popular machine learning libraries on
Kestrel: [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/).
The examples use [Anaconda environments](https://www.anaconda.com/), so if you
are not familiar with their use please refer to the
[NLR HPC page on using Conda environments](../Environment/Customization/conda.md)
and also the Conda guide to
[managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

The page is organized by framework. Each framework section covers CPU
installation, GPU installation, the pre-built GPU module, and a worked
example. Shared [job scripts](#job-scripts) and [installing on Gila](#installing-pytorch-on-gila)
are covered at the end.

!!! note
	This page is only scratching the surface of ML libraries and resources that can be used on Kestrel. Tools such as LightGBM, XGBoost, and scikit-learn work well with conda environments, and other tools such as Flux for the Julia Language can be used on Kestrel as well.

!!! Note
	We recommend installing software for GPU jobs using the GPU nodes. There are two [GPU login nodes](../Systems/Kestrel/index.md) available on Kestrel.

To install either PyTorch or TensorFlow for use with GPUs on Kestrel, the first step is to load the anaconda module on the GPU node using ```module load conda```. Once the anaconda module has been loaded, create a new environment in which to install PyTorch or TensorFlow, e.g.,

??? example "Creating and activating a new conda environment"
        conda create --prefix /projects/<your-project-name>/<your-user-name>/<conda-env-dir>/pt python=3.9
        conda activate /projects/<your-project-name>/<your-user-name>/<conda-env-dir>/<pt or tf>

!!! Note
	If you are not familiar with using [Anaconda environments](https://www.anaconda.com/) please refer to the [NLR HPC page on using Conda environments](../Environment/Customization/conda.md) and also the Conda guide to [managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

---

## PyTorch

### Installing PyTorch (CPU)

To begin, we will outline basic steps for building a simple CPU-based conda environment for PyTorch. First, load the anaconda module and create a new conda environment:
```
module load anaconda3

conda create -p /projects/YOUR_PROJECT/YOUR_USER_NAME_HERE/FOLDER_FOR_CONDA_ENVIRONMENTS/pt python=3.9
```
Answer yes to proceed, and you should end up with directions for starting your conda environment pt. Note that these instructions place your environment in the specified /projects folder. This is advisable, as opposed to installing conda environments in their default location in your home directory. See our [Conda documentation](../Environment/Customization/conda.md#where-to-store-conda-environments) for more information.

Activate the pt conda environment and install PyTorch into the active conda environment:
```
conda activate /projects/YOUR_PROJECT/YOUR_USER_NAME_HERE/FOLDER_FOR_CONDA_ENVIRONMENTS/pt

conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
Answer yes to proceed, and you should be up and running with PyTorch! The [PyTorch](https://pytorch.org/) webpage has great resources for getting started, including resources on [learning the basics](https://pytorch.org/tutorials/beginner/basics/intro.html) and [PyTorch recipes](https://pytorch.org/tutorials/recipes/recipes_index.html).

### Installing PyTorch (GPU)

In an activated python environment, you can install PyTorch using the standard approach found under the Get Started tab of the [PyTorch](https://pytorch.org/) website, e.g., using ```pip```,

??? example "Installing PyTorch using pip"
	```pip3 install torch torchvision torchaudio```

or using ```conda,```

??? example "Installing PyTorch using conda specifying CUDA 12.4"
	```conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia```

### Pre-built PyTorch GPU module

??? example "Experimental: Pre-built pytorch/2.10.0 module"

    A pre-built `pytorch` module is available on Kestrel as an experimental alternative to the conda-unpack based approach below. It provides PyTorch 2.10.0 with CUDA 12.4, NCCL 2.23.4, and Python 3.11. It is for **GPUs nodes only**.

    Load it with:
    ```
    module load pytorch
    ```

    which will print usage instructions:
    ```
    PyTorch 2.10.0 loaded (CUDA 12.4 | NCCL 2.23.4 | Python 3.11 | H100/sm_90 only)

    No extra packages needed? Use directly:
      python3 your_script.py

    Option 1 — venv (recommended when you need extra pip packages):
      python3 -m venv /scratch/$USER/myenv --system-site-packages
      source /scratch/$USER/myenv/bin/activate
      pip install <your-packages>

    Option 2 — conda env (only when you need non-Python deps like compiled libs or tools):
      conda create -p ~/myenv <non-python-deps>
      conda activate ~/myenv
      $PYTORCH_PYTHON \
          -m venv ~/myenv/pyenv --system-site-packages
      source ~/myenv/pyenv/bin/activate
      pip install <your-packages>
      NOTE: do NOT 'conda install' Python packages — use the venv pip instead.
    ```

### Pre-built PyTorch environment with multi-node and GPU support

For training large datasets on multiple GPUs with NCCL and MPI support, please use our pre-built environment. The environment can be downloaded and installed in a directory of your choice by executing the following instructions after logging into a GPU node on Kestrel.

??? example "Installing pre-built PyTorch MPI NCCL environment"
	```
       $ cp /nopt/nrel/apps/examples/python_envs/torchParallel.tar.gz . 
       $ mkdir -p my_torch_MPI_NCCL
       $ tar -xzf torchParallel.tar.gz -C my_torch_MPI_NCCL
       $ source my_torch_MPI_NCCL/bin/activate
       $ conda-unpack
	   $ echo "import numpy; numpy.version.version" > ${CONDA_PREFIX}/lib/python3.13/site-packages/00-preload-numpy.pth
    ```

Once the environment has been installed, it can be tested with the following steps:

??? example "Testing the pre-built PyTorch MPI NCCL environment"
	```
       $ wget https://raw.githubusercontent.com/NatLabRockies/HPC/gh-pages/docs/Documentation/Machine_Learning/metadata/testPytorchMPI.py
       $ wget https://raw.githubusercontent.com/NatLabRockies/HPC/gh-pages/docs/Documentation/Machine_Learning/metadata/testNCCL.py
       $ salloc -A <projectname> -t 00:15:00 --nodes=2 --ntasks-per-node=1 --gres=gpu:1
       $ source my_torch_MPI_NCCL/bin/activate
       $ srun -n 2 python testPytorchMPI.py
       Hello from process 0 (out of 2)!
       Hello from process 1 (out of 2)!
       $ srun -n 2 python testNCCL.py
       Successfully initialized process group with NCCL backend.
       Successfully initialized process group with NCCL backend.
    ```

Users can also install additional packages on top of this environment. When installing additional packages, please be informed that this enviroment was produced by compiling ```pytorch v2.7.0``` from source using ```PrgEnv-gnu/8.5.0```, ```anaconda3/2024.06.1```, ```cuda/12.3```, ```gcc-native/11.2```, ```cray-mpich/8.1.28``` and the ```nccl/2.21.5``` modules. Loading these same modules before installing additional python packages is less likely to lead to conflicts. 

If another version of pytorch is desired, users may compile and install it on their own by using the following steps used in building v2.7.0 as a guideline:

??? example "Building PyTorch MPI NCCL from source"
	```
       $ module load PrgEnv-gnu/8.5.0
       $ module load cuda/12.3
       $ module load anaconda3/2024.06.1
       $ conda create --prefix ./torchMPI
       $ conda activate ./torchMPI
       $ module load gcc-native/11.2
       $ conda install python
       $ conda install yaml
       $ conda install pyyaml
       $ conda install typing_extensions
       $ conda install numactl
       $ conda install scipy
       $ module load nccl
       $ export USE_SYSTEM_NCCL=1
       $ export NCCL_ROOT_DIR=/nopt/nrel/apps/gpu_stack/software/nccl/2.21.5/install/
       $ git clone --branch v2.7.0 https://github.com/pytorch/pytorch.git
       $ cd pytorch
       $ MAX_JOBS=20 python setup.py install
    ```

### PyTorch Example
Below we present a simple convolutional neural network example for getting started using PyTorch with Kestrel GPUs. The original, more detailed version of this example can be found in the pytorch tutorials repo [here](https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py).

??? example "CIFAR10 example"
    ```
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    # Check if there are GPUs. If so, use the first one in the list
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load data and normalize
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    # Define the CNN
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    # send the network to the device
    # If you want to use data parallelism across multiple GPUs, uncomment if statement below
    #if torch.cuda.device_count() > 1:
    #    net = nn.DataParallel(net)
    
    net.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data # setup without device
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    ```

!!! Note
	Currently, this code will run on a single GPU, specifically the GPU denoted ```cuda:0```. To use multiple GPUs via data parallelism, uncomment the two lines above the ```net.to(device)``` command. Furthermore, use of multiple GPUs require requesting multiple GPUs for the batch or interactive job.

!!! Note
	To better observe the multi-GPU peformance of the above example, you can change the size of the CNN. For example, by increasing the size of the second argument in the definition of ```self.conv1``` and the first argument in ```self.conv2```, you can increase the size of the network and use more resources for training.

---

## TensorFlow

### Installing TensorFlow (CPU)

Getting started with TensorFlow on CPU is similar to the process for PyTorch. The first step is to construct an empty conda environment to work in:
```
module load anaconda3

conda create -p /projects/YOUR_PROJECT/YOUR_USER_NAME_HERE/FOLDER_FOR_CONDA_ENVIRONMENTS/tf python=3.9
```
Subsequently, activate the tf conda environment, ensure you are running the latest version of pip in your environment, and install the CPU only version of TensorFlow using pip:
```
conda activate /projects/YOUR_PROJECT/YOUR_USER_NAME_HERE/FOLDER_FOR_CONDA_ENVIRONMENTS/tf
pip install --upgrade pip
pip install tensorflow-cpu
```
You should now be up and running with a TensorFlow! Similar to PyTorch, the [TensorFlow webpage](https://www.tensorflow.org/learn) has lots of great resources for getting started, including tutorials, basic examples, and more! 

### Installing TensorFlow (GPU)

Presented below are instructions for installing TensorFlow following in the ```pip``` install instructions found here: [TensorFlow](https://www.tensorflow.org/install). For optimized TensorFlow performance, we recommend using a [containerized version of TensorFlow](Containerized_TensorFlow/index.md).

You can install TensorFlow using the ```pip``` based approach described in [TensorFlow](https://www.tensorflow.org/install/pip), but with a couple modifications. Instead of using the ```cudatoolkit```, we recommend using the default gnu programming environment accessed using the module ```PrgEnv-gnu/8.5.0```. The ```nccl/2.21.5_cuda124``` module loads its corresponding ```cuda/12.4```. Using these modules, we install TensorFlow with the following commands: 

??? example "Installing TensorFlow using pip"
	```
	ml PrgEnv-gnu/8.5.0
	ml anaconda3 nccl/2.21.5_cuda124
	conda create -p ./tf-env python=3.11 pip -y
	conda activate ./tf-env
	pip install tensorflow[and-cuda]==2.18.0
	```

### Pre-built TensorFlow GPU module

??? example "Experimental: Pre-built tensorflow/2.21.0 module (RHEL 9 GPU)"

    A pre-built `tensorflow` module is available on Kestrel's **RHEL 9 GPU
    nodes** as an alternative to building your own environment. It provides
    TensorFlow 2.21.0 with Python 3.12 and GPU support for H100 (sm_90).
    CUDA 12.9 and cuDNN come from the `tensorflow[and-cuda]` pip
    wheels, while the on-node Slingshot-tuned NCCL 2.27.7 (libfabric-CXI) and
    system cuDNN 9.17 are layered in for correct multi-node collectives and to
    avoid an H100 conv-backprop regression under `MirroredStrategy`.

    The module is reachable from the **RHEL 9 GPU login node `kl5`** (log in
    there, or `ssh kl5` from a RHEL 9 login). Load it with:

    ```
    module load tensorflow
    ```

    which prints usage instructions:

    ```
    TensorFlow 2.21.0 loaded (CUDA 12.9-bundled | Python 3.12 | H100/sm_90 only)
    NCCL:  Kestrel Slingshot 2.27.7 (overrides bundled 2.30.7 for fabric perf)
    cuDNN: system 9.17 (overrides bundled 9.24 - avoids "No algorithm worked"
           regression on H100 conv-backprop under MirroredStrategy)
    ```

    Use it directly, or build a lightweight personal venv on top when you need
    extra packages:

    ```
    # Use directly
    python3 your_script.py

    # Personal venv on top (recommended for extra pip packages)
    python3 -m venv /scratch/$USER/tf_env --system-site-packages
    source /scratch/$USER/tf_env/bin/activate
    pip install <your-extras>
    ```

    !!! Note
        Do **not** combine `--system-site-packages` with a conda environment,
        and do not `conda install` Python packages on top of the module — use
        the venv `pip` instead.

    For multi-node MPI (the bundled `mpi4py` is built against Kestrel's MPICH
    4.3.2), launch with `srun`:

    ```
    srun --mpi=pmi2 -n <N> python3 your_mpi_script.py
    ```

    Slurm auto-allocates a Slingshot VNI for `pmi2` jobs when `nodes >= 2`. To
    force one explicitly (e.g. single-node MPI), add `--network=job_vni` (or
    `--network=single_node_vni`) to `salloc`/`srun`. TensorFlow's own
    `MultiWorkerMirroredStrategy` uses gRPC + NCCL and is unaffected.

    Full details are in the guide at
    `/nopt/nlr/apps/kestrel-gpu/software/tensorflow/tensorflow-2.21.0-guide.md`.

### TensorFlow Example
Find below a simple neural network example using the MNIST data set for getting started using TensorFlow with Kestrel GPUs. This example was based on TensorFlow's quick start documentation found [here](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb).

??? example "MNIST example"
    ```

	import tensorflow as tf

	# Select a standard data set and normalize
	mnist = tf.keras.datasets.mnist    
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	# Set up and compile a model 
	model = tf.keras.models.Sequential([
    	tf.keras.layers.Flatten(input_shape=(28, 28)),  tf.keras.layers.Dense(128, activation='relu'), 
    	tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10, activation='softmax')]) 

	model.compile(optimizer='adam', 
	loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	# Fit model to training data and evaluate on test data
	model.fit(x_train, y_train, epochs=5)

	model.evaluate(x_test, y_test)
    ```

---

## Installing PyTorch on Gila

The [Gila](../Systems/Gila/index.md) cluster hosts two types of accelerator nodes. Although both types use GPU drivers compatible with `cuda/13.1`, critically, they differ in their overall chip architecture: 

1. NVIDIA A100 GPUs with Intel Xeon Icelake CPUs nodes (`x86`)
2. NVIDIA Grace Hopper nodes (`arm`)

!!! Note 
    Python/conda environments built on Gila will only work for **either** `x86` or `arm` architectures depending on which node was used to create them. In other words, an environment created on an `x86`-based node hosting A100s would *not* be expected to work on an `arm`-based Grace Hopper node (and vice versa). This is generally true for all software managed on Gila. As such, always ensure you are using an environment that was created on the same node architecture you plan to run it on.

Regardless of architecture, installing an accelerated version PyTorch on Gila is straightforward; simply load the `cuda` and `miniforge3` modules and pass the appropriate `index-url` to use cuda 13+ wheels. This example reflects a minimal arm-based environment that starts with `python` and the `numpy` package. Note that in this example, a project's `.conda-envs` folder is assumed to be organized to include `arm` and `x86` subfolders to distinguish environments created for the two different types of architectures:

??? example "Creating an arm-based PyTorch environment for Grace Hopper nodes on Gila"
    ```
    # Connect to the arm login node
    ssh gila-arm.hpc.nlr.gov
    
    # Request partial Grace Hopper node for 15 minutes to create arm-based environment
    # Replace <allocation handle> accordingly
    salloc -A <allocation handle> -p gh -t 00:15:00 --mem-per-cpu=2G -n 1 -c 8 --gres=gpu:1

    # Define location for PyTorch environment (don't forget to replace <allocation handle>!)
    EXAMPLE_TORCH_ENV=/projects/<allocation handle>/.conda-envs/arm/torch-test-env

    # Load modules, create env, and use pip to install torch into it
    ml cuda/13.1 miniforge3
    mamba create --prefix=$EXAMPLE_TORCH_ENV python numpy -y
    conda activate $EXAMPLE_TORCH_ENV
    pip install torch --index-url https://download.pytorch.org/whl/cu130
    ```

---

## Job Scripts

The following job scripts work for either PyTorch or TensorFlow — activate the
appropriate environment (or `module load pytorch` / `module load tensorflow`)
and point `srun` at your script.

??? example "PyTorch or TensorFlow shared partition CPU example"
      ```
      #!/bin/bash
      #SBATCH --nodes=1
      #SBATCH --ntasks-per-node=1
      #SBATCH --mem=64G
      #SBATCH --cpus-per-task=26
      #SBATCH --partition=shared
      #SBATCH --account=<your account>
      #SBATCH --time=00:30:00

      module load conda
      conda activate /projects/<your_project>/<conda_envs_dir>/<pt_or_tf>

      srun python <your_code>.py
      ```

Once you have completed your batch file, submit using
```
sbatch <your_batch_file_name>.sb
```

??? example "Sample job script: Kestrel - Shared (partial) GPU node"

    ```
    #!/bin/bash
    #SBATCH --account=<your-account-name> 
    #SBATCH --nodes=1
    #SBATCH --gpus=1 
    #SBATCH --ntasks-per-node=1
    #SBATCH --mem=96G
    #SBATCH --cpus-per-task=32
    #SBATCH --time=00:30:00
    #SBATCH --job-name=<your-job-name>

    module load conda
    conda activate /projects/<your-project-name>/<your-user-name>/<conda-env-dir>/<pt or tf>

    srun python <your-pytorch or tensorflow-code>.py
    ```
