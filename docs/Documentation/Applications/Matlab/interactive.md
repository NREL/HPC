# Running MATLAB Software Interactively

*Learn how to run MATLAB software interactively on NREL HPC systems.*

<!-- TODO: Update link for System Connection. -->

To run MATLAB interactively there are two ways to proceed: you can choose to start an
interactive job and use a basic MATLAB terminal (no GUI), or you can use the GUI
with a [FastX](../../../Development/Vizualization/FastX/fastx) session on a DAV
node.  For information on how to connect to NREL HPC systems, see [System
Connection](https://www.nrel.gov/hpc/system-connection.html).

## Running MATLAB via an Interactive Job

After connecting to the login node, the next step is to start an interactive job. For
example, the following command gets # nodes for interactive use for an hour:

```
$ srun -N # --time 01:00:00 --account=<handle> --pty $SHELL
```

When your job starts, you will have a shell on a compute node.

<!-- TODO: Update the User Accounts and Interactive Jobs links below. -->

!!! note

    1. To submit an interactive job you must include the `--account=<handle>` flag
       and include a valid project allocation handle. For more information, see
       [User Accounts](https://www.nrel.gov/hpc/user-accounts.html).
    2. For more information on interactive jobs, see [Running Interactive
       Jobs](https://www.nrel.gov/hpc/eagle-interactive-jobs.html).

From the shell on the compute node, the next steps are to load the MATLAB module
to set up your user environment, which includes setting the location of the
license server,

```
$ module load matlab
```

and starting a simple MATLAB terminal (no GUI),

```
$ matlab -nodisplay
```

## Running MATLAB via a FastX Session on a DAV Node

For instructions on starting a FastX session on a DAV node, see the [FastX
page](../../../.../../Development/Vizualization/FastX/fastx). Once you have
started a FastX session and have access to a terminal, load the MATLAB module to
set up your user environment, which includes setting the location of the license
server,

```
$ module load matlab
```

and start the MATLAB GUI,

```
$ matlab &
```

With FastX, this will enable you to use the GUI as if MATLAB was running
directly on your laptop. The ampersand "&" lets MATLAB run as a background job
so the terminal is freed up for other uses.
