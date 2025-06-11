# Jupyter Notebook in VS Code
Before proceeding with this document, please read [Connecting With VS Code](./vscode.md) and [Interactive Parallel Python with Jupyter](../Languages/Python/KestrelParallelPythonJupyter/pyEnvsAndLaunchingJobs.md) as this document will be using both as a reference.

The aim of this document is to set up Jupyter in VSCode on a compute node. This allows for VSCode extensions and tools to be used in a Jupyter coding environment to great effect. This document will be making use of example code that demonstrates the use of multiple nodes.

## Setting Up VS Code
To begin, proceed to VSCode and install the following extensions, if you have not already: [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh), [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

## Setting Up SSH-Connect

Before work can begin, the first step is to setup VS Code to be able to connect to a compute node. Like in the [Connecting With VS Code](./vscode.md#ssh-key-setup), you must have an SSH key linked to the Kestrel and have the SSH config properly set up. Please see the aforementioned documentation for further details.

## Setting Up Conda Environment.
In addition, you will also need to set up a Python Environment on Kestrel as per the [Interactive Parallel Python with Jupyter](../Languages/Python/KestrelParallelPythonJupyter/pyEnvsAndLaunchingJobs.md#install-packages) documentation. Please note that you do not need a python environment on your local machine for this. 

## VS Code Jupyter on Kestrel

With all setup finalized, the process will be following [multinode Jupyter job example](../Languages/Python/KestrelParallelPythonJupyter/pyEnvsAndLaunchingJobs.md#multinode-capable-job-eg-mpi4py-through-ipyparallel) through VS Code on the compute. Download the [example Jupyter script](../Languages/Python/KestrelParallelPythonJupyter/exampleNotebooks/cupyAndIpyparallel.ipynb) onto Kestrel to load it into VS Code later.

### Intializing Environment

Before using VS Code to connect to a compute, the compute node has to be allocated. Log onto Kestrel via a terminal and allocate 2 nodes by using this line:
```
salloc -A <projectname> -t 00:30:00 --nodes=2 --ntasks-per-node=1 --partition=short
```

Note the node you have connected so you can connect to it via VS Code in a moment.

### Running the Code

First, open VS Code and hit F1 to run a command. Step through the process of connecting to a compute node. Once connected to the compute node, hit F1 again to search for Jupyter. Select the "Select Interpreter to Start Jupyter Server" option and select the Python Environment made for this example

Once an interpreter has been selected, the code will be able to run via Jupyter in VS Code. If all is working, both nodes should be listed in the code, and it will run to completion swiftly.

