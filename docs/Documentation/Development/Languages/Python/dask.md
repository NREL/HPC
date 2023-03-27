---
layout: default
title: Dask
parent: Python
nav_order: 2
has_children: true
grand_parent: Programming Languages
---


# Dask

Dask provides a framework for parallelizing Python code.  The most common use case is to enable Python programmers to scale scientific and machine learning analyses to run on distributed hardware.  Dask has similarities to Apache Spark (see [FAQ](https://docs.dask.org/en/stable/faq.html#how-does-dask-compare-with-apache-spark) for comparison), but Dask is more Python native and interfaces with common scientific libraries such as numpy and pandas.

## Dask single node

Dask can be used locally on your laptop or an individual node. Additionally, it provides wrappers for multiprocessing and threadpools. The advantage of using `LocalCluster` though is you can easily drop in another cluster configuration to further parallelize. 

```python
from distributed import Client, LocalCluster
import dask
import time
import random 

@dask.delayed
def inc(x):
    time.sleep(random.random())
    return x + 1

@dask.delayed
def dec(x):
    time.sleep(random.random())
    return x - 1

@dask.delayed
def add(x, y):
    time.sleep(random.random())
    return x + y

def main ():
   cluster = LocalCluster(n_workers=2)
   client = Client(cluster)
   zs = []
   for i in range(256):
      x = inc(i)
      y = dec(x)
      z = add(x, y)
      zs.append(z)
   
   result = dask.compute(*zs)
   print (result)


if __name__ == "__main__":
   main()
```

## Dask MPI
Dask-MPI can be used to parallelize calculations across a number of nodes as part of a batch job submitted to slurm. Dask will automatically create a scheduler on rank 0 and workers will be created on all other ranks. 

### Install
**Note:** The version of dask-mpi installed via Conda may be incompatible with the MPI libaries on Eagle. Use the pip install instead. 

```
conda create -n daskmpi python=3.7
conda activate daskmpi
pip install dask-mpi
```

**Python script**: This script holds the calculation to be performed in the test function. The script relies on the Dask cluster setup on MPI which is created in the 
```python
from distributed import Client, LocalCluster
import dask
import time
from dask_mpi import initialize
import random 

@dask.delayed
def inc(x):
    time.sleep(random.random())
    return x + 1

@dask.delayed
def dec(x):
    time.sleep(random.random())
    return x - 1

@dask.delayed
def add(x, y):
    time.sleep(random.random())
    return x + y

def main ():
   initialize(nanny=False,
      interface='ib0',
      protocol='tcp',
      memory_limit=0.8,
      local_directory='/tmp/scratch/dask',
      nthreads=1)

   client = Client()
   zs = []
   for i in range(256):
      x = inc(i)
      y = dec(x)
      z = add(x, y)
      zs.append(z)
   
   result = dask.compute(*zs)
   print (result)


if __name__ == "__main__":
   main()
```

Running the above script with MPI will automatically set a Dask worker on each MPI rank. 
```shell
mpiexec -np 30 python dask_mpi.py
```

## Dask jobqueue
Dask can also run using the Slurm scheduler already installed on Eagle. The Jobqueue library can handle submission of a computation to the cluster. This is particularly useful when running an interactive notebook or similar and you need to scale workers. 

```python
import dask
import time
from dask_jobqueue import SLURMCluster
from distributed import Client
import random 

@dask.delayed
def inc(x):
    time.sleep(random.random())
    return x + 1

@dask.delayed
def dec(x):
    time.sleep(random.random())
    return x - 1

@dask.delayed
def add(x, y):
    time.sleep(random.random())
    return x + y

def main ():
   cluster = SLURMCluster(
      cores=18,
      memory='24GB',
      queue='short',
      project='hpcapps',
      walltime='00:30:00',
      interface='ib0',
      processes=18,
   )
   cluster.scale(jobs=2)

   client = Client(cluster)
   zs = []
   for i in range(256):
      x = inc(i)
      y = dec(x)
      z = add(x, y)
      zs.append(z)
   
  
   result = dask.compute(*zs)
   print (result)


if __name__ == "__main__":
   main()

```

## References
[Dask documentation](https://docs.dask.org/en/latest/)

[Dask Jobqueue](https://jobqueue.dask.org/en/latest/)

[Dask MPI](http://mpi.dask.org/en/latest/)
