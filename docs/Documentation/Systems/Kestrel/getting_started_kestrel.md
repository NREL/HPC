# Getting Started

## Logging In

NREL employees can access Kestrel via ssh:
```
ssh <username>@kestrel.hpc.nrel.gov
```

There are no external-facing login nodes for Kestrel. External collaborators can connect to the [HPC VPN](https://www.nrel.gov/hpc/vpn-connection.html) and ssh to Kestrel as directed above. 

DAV nodes can be accessed by using a web browser and connecting to ```https://kestrel-dav.hpc.nrel.gov``` or by using the FastX desktop client. 


We recommend using [Globus](../../Managing_Data/Transferring_Files/globus.md) to transfer files between Eagle and Kestrel. Please see our [Globus documentation](../../Managing_Data/Transferring_Files/globus.md) for information about the Kestrel Globus endpoints. 

To transfer small batches of data, `rsync` or `scp` are also available. 

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
