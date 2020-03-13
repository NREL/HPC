## Login into NREL HPC, Peregrine

A supported set of instructions for logging into the NREL HPC is provided [here](https://hpc.nrel.gov/users/connect/ssh)

Initiate at git-bash terminal session (on Windows) or a terminal (on Mac).

If you are on a standard network with internet connection, like the NREL guest wireless, you will need to login to the gateway system.
```
ssh <username>@hpcsh.nrel.gov
```
Where <username> is is replaced by your HPC username.  Once you’ve connected to hpcsh.nrel.gov or if you are already on a network with direct access to peregrine, run the following command:
```
ssh <username>@peregrine.hpc.nrel.gov 
```
When logging in for the first time you will need to verify the security authenticity and change your password on the HPC.

Once you are on the system you will land in your home directory.

```bash
ssh peregrine.hpc.nrel.gov
*****************************************************************************

                         NOTICE TO USER
....
*****************************************************************************

Send email to hpc-help@nrel.gov for HPC support requests and trouble reports.

[wjones@login4 ~]$ pwd
/home/wjones
```

