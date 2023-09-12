# Getting Started

## Logging In

To access Kestrel and connect via ssh:
```
ssh <username>@kestrel.hpc.nrel.gov
```

DAV nodes can be accessed by using a web browser and connecting to ```https://kestrel-dav.hpc.nrel.gov``` or by using the FastX desktop client. 

## Additional Resources

* [Kestrel System Configuration](https://www.nrel.gov/hpc/kestrel-system-configuration.html)
* A collection of sample makefiles, source codes, and scripts for Kestrel can be found in the [Kestrel repo](https://github.com/NREL/HPC/tree/master/kestrel). 


## Running Jobs

To start an interactive session:

1. Allocate the node(s):<br>
    ```salloc --nodes=N --ntasks-per-node=npn --time=1:00:00 ```
1. ```srun -n np ./executable``` <br>
where "np" is N*npn, and npn=104 if requesting a whole node. 


There are example job submission scripts in the [Environments Tutorial](../../../Friendly_User/Environments/tutorial.md) page. 

## Contributions
The [Kestrel repo](https://github.com/NREL/HPC/tree/master/kestrel) is open for contributions of examples, scripts, and other resources that would benefit the user community. To contribute, please open a Pull Request or contact [hpc-help@nrel.gov](mailto:hpc-help@nrel.gov). 

## Getting Help
Please contact [hpc-help@nrel.gov](mailto:hpc-help@nrel.gov) with any questions. 
