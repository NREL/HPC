---
layout: default
title: Dask
parent: Python
nav_order: 2
has_children: true
grand_parent: Programming Languages
---


# Dask
Dask provides a way to parallelize Python code either on a single node or across the cluster. It is similar to the functionality provided by Apache Spark, with easier setup. It provides a similar API to other common Python packages such as NumPY, Pandas, and others. 

## Dask single node
Dask can be used locally on your laptop or an individual node. Additionally, it provides wrappers for multiprocessing and threadpools. The advantage of using `LocalCluster` though is you can easily drop in another cluster configuration to further parallelize. 

```python
import socket
from distributed import Client, LocalCluster
import dask
from collections import Counter

def test():
   return socket.gethostname()

def main():
   cluster = LocalCluster(n_workers=2)
   client = Client(cluster)

   result = []
   for i in range (0,20):
      result.append(client.submit(test).result())
      
   print (Counter(result))

if __name__ == '__main__':
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
from dask_mpi import initialize
from dask.distributed import Client, wait
import socket
import time
from collections import Counter

def test():
   return socket.gethostname()
   
def main():
   initialize(interface='ib0')
   client = Client()
   time.sleep(15)

   result = []

   for i in range (0,100):
      result.append(client.submit(test).result())
      time.sleep(1)
      
   out = str(Counter(result))
   print (f'nodes: {out}')

main()
```
**sbatch script**: This runs the above python script using MPI.
```shell
#!/bin/bash 
#SBATCH --nodes=2
#SBATCH --time=01:00:00
#SBATCH --account=<hpc account>
#SBATCH --partition=<Eagle partition>

module purge
ml intel-mpi/2018.0.3 
mpiexec -np 4 \
    python mpi_dask.py  \
    --scheduler-file scheduler.json \
    --interface ib0 \
    --no-nanny \
    --nthreads 5
```

## Dask jobqueue
Dask can also run using the Slurm scheduler already installed on Eagle. The Jobqueue library can handle submission of a computation to the cluster. This is particularly useful when running an interactive notebook or similar and you need to scale workers. 

```python
from dask_jobqueue import SLURMCluster
import socket
from distributed import Client
from collections import Counter

cluster = SLURMCluster(
   cores=18,
   memory='24GB',
   queue='short',
   project='<hpc account>',
   walltime='00:30:00',
   interface='ib0',
   processes=17,
)

client = Client(cluster)

def test():
   return socket.gethostname()

result = []
cluster.scale(jobs=2)

for i in range (0,2000):
   result.append(client.submit(test).result())
   
print (Counter(result))
print (cluster.job_script())

```

## References
[Dask documentation](https://docs.dask.org/en/latest/)

[Dask Jobqueue](https://jobqueue.dask.org/en/latest/)

[Dask MPI](http://mpi.dask.org/en/latest/)