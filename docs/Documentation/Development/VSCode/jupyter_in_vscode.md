# Jupyter Notebook in VSCode
Before proceeding with this document, please read [Connecting With VS Code](./vscode.md) and [Interactive Parallel Python with Jupyter](../Languages/Python/KestrelParallelPythonJupyter/pyEnvsAndLaunchingJobs.md) as this document will be using both as a reference.

The aim of this document is to set up Jupyter in VSCode on a compute node. This allows for VSCode extensions and tools to be used in a Jupyter coding environment to great effect. This document will be making use of example code that demonstrates the use of multiple nodes.

## Setting Up VS Code
To begin, proceed to VSCode and install the following extensions, if you have not already: [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh), [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).

## Setting Up SSH-Connect

Before work can begin, the first step must be to connect to a compute node to run the code. Like in the [Connecting With VS Code](./vscode.md#ssh-key-setup), you must have an SSH key linked to the Kestrel and have the SSH config properly set up. Please see the aforementioned documentation for further details.

# Setting Up Conda Environment.
In addition, you will also need to set up a Python Environment as per the [Interactive Parallel Python with Jupyter](../Languages/Python/KestrelParallelPythonJupyter/pyEnvsAndLaunchingJobs.md#install-packages) documentation.
