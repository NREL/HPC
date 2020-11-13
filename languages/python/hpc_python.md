# HPC Python
A collection of modules and tools for developing HPC Python scripts. These resources enable python to utilize a GPU or multi-node execution. 

## Resources
### Concurrency/Parallelism
* [Dask](https://github.com/dask/dask) 
    * [Dask-MPI](https://github.com/dask/dask-mpi): deploy workers across MPI ranks
* [MPI4Py](https://github.com/mpi4py/mpi4py): Python MPI bindings
* [Ray](https://github.com/ray-project/ray): Distributed Python

### GPU
* [CuPy](https://github.com/cupy/cupy): GPU accelerated NumPy implementation
* [cuDF](https://github.com/rapidsai/cudf): GPU accelerated dataframes, efficient for use with strings and numerics
* [CuML](https://github.com/NREL/HPC/blob/hpc-python/languages/python/hpc_python.md): Scikit-learn and other ML on GPU

### Speed
* [UCX-Py](https://ucx-py.readthedocs.io/en/latest/): UCX bindings to take advantage of NVLINK or Infiniband
* [Numba](http://numba.pydata.org/): Numba makes Python code fast
* [Modin](https://github.com/modin-project/modin) Drop in replacement for Pandas and Dask, can distribute operations 


### Integrate
* [pvpython](https://kitware.github.io/paraview-docs/latest/python/)
   * [pvpython examples](https://www.paraview.org/Wiki/ParaView/PythonRecipes): ParaView/Python Recipes
