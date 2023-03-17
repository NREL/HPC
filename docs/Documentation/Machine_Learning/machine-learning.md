# Machine Learning

**Documentation:** [ link to documentation](https://nrel.gov)

*Machine learning refers to a set of techniques and algorithms that enable computers to automatically learn from data and improve their performance on a specific task over time. Types of machine learning methods include, but are not limited to, supervised learning (algorithms trained on labeled datasets), unsupervised learning (algorithms trained on unlabeled datasets), and reinforcement learning (learning by trial and error). The Computational Science Center at NREL conducts research in these types of machine learning, and also supports the use of machine learning software on Kestrel.*

## Getting Started

This section provides basic examples for getting started with two popular machine learning libraries: [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/). Both examples use [Anaconda environments](https://www.anaconda.com/), so if you are not familiar with their use please refer to the NREL HPC page on using Anaconda environments on Kestrel (INCLUDE LINK HERE!) and also the Anaconda guide to [managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). 

###Getting started with PyTorch

To begin, we will outline basic steps for building a simple CPU-based conda environment for PyTorch. First, load the anaconda module and create a new conda environment:
```
module load conda

conda create -p /projects/YOUR_PROJECT/YOUR_USER_NAME_HERE/FOLDER_FOR_CONDA_ENVIRONMENTS/pt python=3.9
```
Answer yes to proceed, and you should end up with directions for starting your conda environment pt. Note that these instructions place your environment in the projects folder. This is advisable, as opposed to installing conda environments in their default location in your home directory. See the using Anaconda environment on Kestrel instructions on Kestrel (INCLUDE LINK HERE!) for more information.

Activate the pt conda environment and install PyTorch into the active conda environment:
```
conda activate /projects/YOUR_PROJECT/YOUR_USER_NAME_HERE/FOLDER_FOR_CONDA_ENVIRONMENTS/pt

conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
Answer yes to proceed, and you should be up and running with PyTorch! The [PyTorch](https://pytorch.org/) webpage has great resources for getting started, including resources on [learning the basics](https://pytorch.org/tutorials/beginner/basics/intro.html) and [PyTorch recipies](https://pytorch.org/tutorials/recipes/recipes_index.html).

###Getting started with TensorFlow

Getting started with TensorFlow is similar to process for PyTorch. The first step is to construct anempty conda environment to work in:
```
module load conda

conda create -p /projects/YOUR_PROJECT/YOUR_USER_NAME_HERE/FOLDER_FOR_CONDA_ENVIRONMENTS/tf python=3.9
```
Subsequently, activate the tf conda environment, ensure you are running the latest version of pip in your environment, and install the CPU only version of TensorFlow using pip:
```
conda activate /projects/YOUR_PROJECT/YOUR_USER_NAME_HERE/FOLDER_FOR_CONDA_ENVIRONMENTS/tf
pip install --upgrade pip
pip install tensorflow-cpu
```
You should now be up and running with a TensorFlow! Similar to PyTorch, the [TensorFlow webpage](https://www.tensorflow.org/learn) has lots of great resources for getting started, including turotials, basic examples, and more! 

Include a section on how to run the job, e.g., with job script examples or commands for an interactive session.

### Example Job Script

??? example "PyTorch or TensorFlow CPU"
	```
	#!/bin/bash 
	#SBATCH --nodes=1			# Run the tasks on the same node
	#SBATCH --time=1:00:00			# Required, estimate 1 hour
	#SBATCH --account=<your_account>
	#SBATCH --partition=debug
	#SBATCH --exclusive			# if you want to use the whole node

	module load conda 

	cd /projects/<your_project_here>/<your_code_directory>

	conda activate /projects/<your_project_here>/<folder_for_conda_envs>/pt #or tf

	srun python your_pt_code.py

	```
!!! note
	This Getting Started section is only scratching the surface of ML libraries and resources that can be used on Kestrel. Tools such as LightGBM, XGBoost, and scikit-learn work well with conda environments, and other tools such as Flux for the Julia Language can be used on Kestrel as well.

Once you have completed your batch file, submit using
```
sbatch <your_batch_file_name>.sb
```

## Supported Versions

| Kestrel | Swift | Vermillion |
|:-------:|:-----:|:----------:|
| 0.0.0   | 0.0.0 | 0.0.0      |

## Advanced

The above examples are simple CPU-based computing environments. To build conda environments for GPUs we refer you to the [PyTorch](https://pytorch.org/get-started/locally/) and [TensorFlow](https://www.tensorflow.org/install) install directions. 

For optimized TensorFlow performance, we recommend using a [containerized version of TensorFlow](https://). 

### Building From Source

Here, give detailed and step-by-step instructions on how to build the code, if this step is necessary. Include detailed instructions for how to do it on each applicable HPC system. Be explicit in your instructions. Ideally a user reading one of the build sections can follow along step-by-step
and have a functioning build by the end.

If building from source is not something anyone would reasonably want to do, remove this section.

Be sure to include where the user can download the source code

??? example "Building on Kestrel"

	Include here, for example, a Kestrel-specific makefile (see berkeleygw example page). This template assumes that we build the code with only one toolchain, which may not be the case. If someone might reasonably want to build with multiple toolchains, use the "Multiple toolchain instructions on Kestrel" template instead.
	
	```
	Include relevant commands in blocks.
	```
	or as in-line `blocks`

	Be sure to state how to set-up the necessary environment, e.g.:

	```
	module load gcc/8.4.0
	module load openmpi/3.1.6/gcc-8.4.0
	module load hdf5/1.10.6/gcc-ompi
	```

	Give instructions on compile commands. E.g., to view the available make targets, type `make`. To compile all program executables, type:

	```
	make cleanall
	make all
	```
	
??? example "Building on Vermillion"

	information on how to build on Vermillion

??? example "Building on Swift"

	information on how to build on Swift


## Troubleshooting

If you run into any issues with the installation instructions above, please reach out to User and Apps Support...
