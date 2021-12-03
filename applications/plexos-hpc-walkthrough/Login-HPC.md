# Login into NREL HPC, Eagle

A supported set of instructions for logging into the NREL HPC is provided [here](https://hpc.nrel.gov/users/connect/ssh)

Initiate at git-bash terminal session (on Windows) or a terminal (on Mac).

## External Access

If you are on a standard network with internet connection, e.g. the NREL guest wireless, or are using a non-NREL computer you will need to login to the gateway system.

```bash
ssh <username>@hpcsh.nrel.gov
```

Where ``<username>`` is is replaced by your HPC username.  Once you’ve connected to `hpcsh.nrel.gov` or if you are already on a network with direct access to Eagle, run the following command:
```bash
ssh <username>@eagle.hpc.nrel.gov
```
When logging in for the first time you will need to verify the security authenticity and change your password on the HPC.

## Internal Access

If you are using an NREL-issued computer and are connected to the NREL VPN/wired network, you can run the following directly.

```bash
ssh <username>@eagle.hpc.nrel.gov
```

Once you are on the system you will land in your home directory.

```bash
ssh <username>@eagle.hpc.nrel.gov
*****************************************************************************

                         NOTICE TO USERS
....
*****************************************************************************

Send email to hpc-help@nrel.gov for HPC support requests and trouble reports.

[wjones@el3 ~]$ pwd
/home/wjones
```
