---
layout: default
title: Python
has_children: true
parent: Programming Languages
---

# Python

## NREL HPC Tutorials
* [Python environments](../../../Environment/Customization/conda.md): Utilize a specific version of Python and install packages
* [Dask](dask.md): Parallelize your Python code
* [Jupyter notebooks](../../Jupyter/index.md): Run interactive notebooks

## Parallel Interactive Tutorials
Tututials for utilising Kestrel for parallel interactive python scripts.

* [Setting up modules and allocating compute resources for parallel jobs](./KestrelParallelPytonJupyter/README.md)
* Example jupyter notebooks to download and test:
    * [cupy](./KestrelParallelPytonJupyter/exampleNotebooks/cupyOnly.ipynb)
    * [numbaCUDA](./KestrelParallelPytonJupyter/exampleNotebooks/numbaCUDA.ipynb)
    * [cupy ipyparallel](./KestrelParallelPytonJupyter/exampleNotebooks/cupyAndIpyparallel.ipynb)
    * [dask](./KestrelParallelPytonJupyter/exampleNotebooks/dask.ipynb)

## HPC Python
Links to External resources:

* [MPI4PY](https://mpi4py.readthedocs.io/en/stable/): Python bindings to use MPI to distribute computations across cluster nodes
* [Dask](https://docs.dask.org/en/latest/): Easily launch Dask workers on one node or across nodes
* [Numba](https://numba.pydata.org/numba-doc/latest/index.html): Optimize your Python code to run faster
* [PyCUDA](https://documen.tician.de/pycuda/): Utilize GPUs to accelerate computations
