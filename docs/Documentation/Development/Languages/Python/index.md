---
layout: default
title: Python
has_children: true
parent: Programming Languages
---

# Python

## NREL HPC Documentation
* [Anaconda virtual environments](../../../Environment/Customization/conda.md): Utilize a specific version of Python and install packages within a conda environment.
* [Dask](dask.md): Parallelize your Python code using the Python-native package [Dask](https://www.dask.org).
* [Intro to Jupyter notebooks](../../Jupyter/index.md): Run interactive Jupyter notebooks on NREL HPC systems.
* [Interactive Parallel Python with Jupyter](./KestrelParallelPythonJupyter/pyEnvsAndLaunchingJobs.md): Examples of launching parallel Python code from Jupyter notebooks on NREL HPC systems.

## Interactive Tutorials

The [Interactive Parallel Python with Jupyter](./KestrelParallelPythonJupyter/pyEnvsAndLaunchingJobs.md) page demonstrates various examples of using popular parallel Python packages from a Jupyter notebook.

* Example notebooks to download and test:
    * [cupy](./KestrelParallelPythonJupyter/exampleNotebooks/cupyOnly.ipynb)
    * [numbaCUDA](./KestrelParallelPythonJupyter/exampleNotebooks/numbaCUDA.ipynb)
    * [cupy ipyparallel](./KestrelParallelPythonJupyter/exampleNotebooks/cupyAndIpyparallel.ipynb)
    * [dask](./KestrelParallelPythonJupyter/exampleNotebooks/dask.ipynb)

## HPC Python
Links to External resources:

* [MPI4PY](https://mpi4py.readthedocs.io/en/stable/): Python bindings to use MPI to distribute computations across cluster nodes
* [Dask](https://docs.dask.org/en/latest/): Easily launch Dask workers on one node or across nodes
* [Numba](https://numba.pydata.org/numba-doc/latest/index.html): Optimize your Python code to run faster
* [PyCUDA](https://documen.tician.de/pycuda/): Utilize GPUs to accelerate computations
