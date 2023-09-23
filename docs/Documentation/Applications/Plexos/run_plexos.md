---
title: Running Plexos
parent: Plexos
grand_parent: Applications
---

Please follow the [setup instructions](setup_plexos.md) before running the examples. Example scripts for new users are available within the master branch.

!!! note
    Sometimes newer modules may be availabe in a `test` directory which is hidden by default from the general user base. This obscured release is done to iron out any bugs that may arise during the installation and use of the module while avoiding breaking users existing jobs and workflows. You can use these test modules by running

    ```bash
    module use /nopt/nrel/apps/modules/test/modulefiles
    module avail
    ```

    This should display all of the test modules available in addition to the defaults. We encourage you to reach out to us at HPC-Help@nrel.gov for access if you would like access to these modules.

# PLEXOS Versions 9 and Up

PLEXOS 9.XRY now comes bundled with its own `mono` software. Therefore, so we no longer need to load and call it for  running PLEXOS. We load the following modules

```bash
module load centos gurobi/9.5.1
module load plexos/9.000R09
```

Recall that we can only use the Gurobi solver while running the PLEXOS on the NREL cluster.Now that we have the modules loaded, PLEXOS can be called as follows

```bash
$PLEXOS/PLEXOS64 -n 5_bus_system_v2.xml -m 2024_yr_15percPV_MT_Gurobi
```

The command above assumes that we are running the model `2024_yr_15percPV_MT_Gurobi` from file `5_bus_system_v2.xml`. PLEXOS 9.0RX requires validating user-credentials for a local 
PLEXOS account for each run. Therefore, if we ran the above command in an interactive session, we would need to enter the following username and password

```txt
username : nrelplexos
password : Nr3lplex0s
```

Fortunately, we can bypass the prompt for a local PLEXOS account username and password (useful for slurm batch jobs) by passing them as command line arguments as follows.

```bash
$PLEXOS/PLEXOS64 -n 5_bus_system_v2.xml -m 2024_yr_15percPV_MT_Gurobi -cu nrelplexos -cp Nr3lplex0s
```

!!! caution
    Not providing the username and password in batch jobs WILL cause your jobs to fail.

# Example scripts

## Example 1: Basic Functionality Test

## Example 2: Simple batch script submission

## Example 3: Enhanced batch script submission

## Example 4: Submitting multiple PLEXOS jobs

## Example 5: Running PLEXOS with SLURM array jobs

