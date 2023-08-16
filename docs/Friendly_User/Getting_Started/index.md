# Getting Started

## Logging In

To access Kestrel and connect via ssh:
```
ssh kestrel.hpc.nrel.gov
```

DAV nodes can be accessed by using a web browser and connecting to ```https://kd1.hpc.nrel.gov``` or by using the FastX desktop client. 

## Additional Resources

* [Kestrel System Configuration](https://www.nrel.gov/hpc/kestrel-system-configuration.html)
* A collection of sample makefiles, source codes, and scripts for Kestrel can be found in the [Kestrel repo](https://github.com/NREL/HPC/tree/master/kestrel). 


## Running Jobs

To start an interactive session:

1. Allocate the node(s):<br>
    ```salloc --nodes=N --ntasks-per-node=npn --time=1:00:00 ```
1. 
```srun -n np --mpi=pmi2 ./executable``` <br>
where "np" is N*npn. 

!!! warning
    If the argument --mpi=pmi2 is not used, the executable will be launched np times instead of being launched once using np cores. 

There are example job submission scripts in the [Environments Tutorial](../Environments/tutorial.md) page. 

## Compiling


## Contributions
The [Kestrel repo](https://github.com/NREL/HPC/tree/master/kestrel) is open for contributions of examples, scripts, and other resources that would benefit the user community. To contribute, please open a Pull Request or contact [haley.yandt@nrel.gov](mailto:haley.yandt@nrel.gov) and [olivia.hull@nrel.gov](mailto:olivia.hull@nrel.gov). 

