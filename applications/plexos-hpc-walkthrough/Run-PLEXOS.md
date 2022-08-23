Please follow the instructions in [`Setup-PLEXOS.md`](Setup-PLEXOS.md) before
running the example in this directory. Example scripts for new users are available
in Section 2.

**NOTE:**
Sometimes newer modules may be availabe in a `test` directory which is accessible only to
a limited number of users. This limited release is done to iron out any bugs that
may arise during the installation and use of the module and avoid breaking
users existing jobs. You can check if those test modules are available to you by running

```bash
module use /nopt/nrel/apps/modules/test/modulefiles
module avail
```

This should display all of the test modules available in addition to the defaults. We encourage
you to reach out to us at HPC-Help@nrel.gov for access if you would like access to these
modules.

# PLEXOS Versions 9.0RX

Currently there are two versions PLEXOS 9.0R07 and 9.0R09 available on Eagle. PLEXOS 9.0RX
comes bundled with its own `mono` software so we no longer need to load and call it for 
running PLEXOS. We load the following modules

```bash
module load centos gurobi/9.5.1
module load plexos/9.000R09
```

Now that we have the modules loaded, PLEXOS can be called as follows

```bash
$PLEXOS/PLEXOS64 -n 5_bus_system_v2.xml -m 2024_yr_15percPV_MT_Gurobi
```

The command above assumes that we are running the model `2024_yr_15percPV_MT_Gurobi` from file
`5_bus_system_v2.xml`. PLEXOS 9.0RX requires validating user-credentials for a local 
PLEXOS account for each run. Therefore, if we ran the above command in an interactive session. 
We would need to enter the following username and password

```txt
username : nrelplexos
password : Nr3lplex0s
```

Fortunately, we can bypass the prompt for a local PLEXOS account username and password (useful for slurm
batch jobs) by passing them as command line arguments as follows.

```bash
$PLEXOS/PLEXOS64 -n 5_bus_system_v2.xml -m 2024_yr_15percPV_MT_Gurobi -cu nrelplexos -cp Nr3lplex0s
```

**Not providing the username and password in batch jobs WILL cause your jobs to fail.**

# PLEXOS Versions 8.3RX

Models using PLEXOS 8.3 only work with Gurobi. In order to use PLEXOS 8.3 on Eagle, 
we first need to load the following modules

```bash
module load centos mono/6.8.0.105 plexos/8.300R09
```

Eagle has the same versions of XpressMP and Gurobi as the NREL PLEXOS Servers.
Therefore the user simply needs to load the corresponding module on Eagle before
running their models. For example, if the user setup the model to use Xpress 8.11,
load the module for the same by calling the command

```bash
module load xpressmp/8.11.0
```

Similarly, the model was set up to use Gurobi 9.1.2, load its module by running
the command

```bash
module load gurobi/9.1.2
```




# PLEXOS Versions 8.2 and Lower

We will be mostly working out of our `scratch/$USER` directory on Eagle.

## Example 1: PLEXOS Smoke Test

The first example we will run is a "smoke" test which tests whether the versions
of plexos are fully functional. The repository is available
[here](https://github.nrel.gov/hsorense/plexos-smoke-test) on the internal
GitHub website. We should be able to use the following commands to run the test

```bash
cd /scratch/$USER
mkdir smoke
cd smoke
git clone https://github.nrel.gov/hsorense/plexos-smoke-test
cd plexos-smoke-test/

sbatch --account=MYACCOUNT submit_8.0.sh
```

Remeber to replace `MYACCOUNT` with your own HPC account name. once the job has
been submitted, it is possible to ssh into the compute node and view what is happening
on the node. Something like the following can be used

```bash
squeue -u $USER
# Get the node on which your job running
ssh $NODENAME
ps -U $USER -L -o pid,lwp,psr,comm,pcpu | grep -v COMMAND | sort -k3n
```

## Example 2: MSPCM Workshop-OneWeek

This example runs a base case that simulates one week of the energy market

1. Clone the MSPCM Workshop and move into the model directory
  ```bash
  cd /scratch/$USER/
  git clone https://github.com/GridMod/MSPCM-Workshop.git
  cd MSPCM-Workshop/Workshop-Explorations/OneWeek
  ```

2. Within `OneWeek` we will create a new environment file [`newenv`](RunFiles/newenv) using the following commands
  ```bash
  echo 'export PLEXOS_VERSION="8.0"
  export XPRESSMP_VERSION="8.5.6"
  module purge
  module load centos mono/4.6.2.7 xpressmp/$XPRESSMP_VERSION plexos/$PLEXOS_VERSION
  module load conda

  export PLEXOS_TEMP=/scratch/$USER/tmp/$SLURM_JOBID
  export TEMP=$PLEXOS_TEMP
  mkdir -p $PLEXOS_TEMP
  ' > newenv
  ```
  This will ensure that the appropriate modules and conda versions are loaded and
  an output directory.

3. Request an interactive session on Eagle

  ```bash
  srun --account=MYACCOUNT --time=01:00:00 --partition=debug --ntasks=1 --nodes=1 --pty bash
  ```

4. Setup the new environment. Make sure that you have `numpy` and `pandas` in your
`plex1` environment.

  ```bash
  # Ensure that we are in the correct working directory
  cd /scratch/$USER/MSPCM-Workshop/Workshop-Explorations/OneWeek
  source newenv
  conda activate plex1
  ```

5. Run the [`get_week.py`](RunFiles/get_week.py) script. This script essentially extracts a weeks worth of
data for the simulation

  ```bash
  python get_week.py
  ```

6. Finally we run our day ahead example using the following command

  ```bash
  mono $PLEXOS/PLEXOS64.exe -n "one_week_model.xml" -m DAY_AHEAD
  ```

## Example 3: Simple Batch Job Submission

Up till now we have been running PLEXOS examples in an interactive settings. Now
we will try to submit a batch job into the Eagle queue. This job uses the same
`one_week_model.xml` file from Example 2, but for the sake of completeness, we
will download a tarball of the same from a different repo in our batch script.

1. Copy the batch script [`simple`](RunFiles/simple) into your current working directory on Eagle.
The path to [`simple`](RunFiles/simple) within this repository is

  ```bash
  HPC/applications/plexos-hpc-walkthrough/RunFiles/simple
  ```

2. Change the account name on line 8 of `simple` to be your HPC account name

3. Now that we have the `simple` batch script, we can simply run the example by
using the following command

  ```bash
  sbatch simple
  ```

## Example 4: Enhanced Batch Job Submission

The `enhanced` batch file builds upon the `simple` file from Example 3 and has
additional features such as custom output paths for directing the standard
output and standard error streams. This script will also make a record of the
environment being used for the run using the `printenv` command. Finally, the
enhanced script also hedges against failed runs due to licensing issues by
attempting to run multiple times if there is a license failure. In order to run
this script:

1. Copy the batch script [`enhanced`](RunFiles/enhanced) into your current working directory on Eagle.
The path to [`enhanced`](RunFiles/enhanced) within this repository is

  ```bash
  HPC/applications/plexos-hpc-walkthrough/RunFiles/enhanced
  ```

2. Make sure that you specify custom paths on lines 10 & 11 which write the
standard output and standard error to file. Change the account name on line 8.

3. Create the following directory

  ```bash
  mkdir -p /scratch/$USER/slurmout
  ```

3. The script can be run using the command

  ```bash
  sbatch enhanced
  ```

## Example 5: Submitting multiple PLEXOS jobs

We will rely on the following files to run this example

  - [submitPLEXOS.sh](RunFiles/submitPLEXOS.sh)
  - [runPLEXOS.sh](RunFiles/runPLEXOS.sh)
  - [models.txt](RunFiles/models.txt)
  - r3_CO2_150n35_NoCCS2035_HPC.xml

`r3_CO2_150n35_NoCCS2035_HPC.xml` was generated as part of someone's research
and, therefore, is not present within the GitHub repository. If you would like
to run this particular example, please reach out to [Dr. Timothy Kaiser](mailto:Timothy.Kaiser@nrel.gov)
or [Dr. Kinshuk Panda](mailto:Kinshuk.Panda@nrel.gov).

Once you have the XML file, you can run this example by running the command

```bash
./submitPLEXOS.sh r3_CO2_150n35_NoCCS2035_HPC.xml models.txt
```

If you look at [`submitPLEXOS.sh`](RunFiles/submitPLEXOS.sh), you find that
`submitPLEXOS.sh` calls `runPLEXOS.sh` which actually calls the PLEXOS executable

## Example 6: Running Plexos using SLURM array jobs

We will use [`array.sh`](RunFiles/array.sh) as a template for submitting array
jobs on Eagle. This example will use some of run files we used in Example 5
above.

1. Create a new working directory `plexos_array` and copy [`array.sh`](RunFiles/array.sh),
`r3_CO2_150n35_NoCCS2035_HPC.xml`, and `models.txt`
  ```bash
  $ mkdir /scratch/$USER/plexos_array && cd /scratch/$USER/plexos_array
  # Copy necessary files
  $ ls
  array.sh  data  models.txt  r3_CO2_150n35_NoCCS2035_HPC.xml
  ```

  **`data`** above is a directory that contains ReEDS_Data. Please reach out to
  [Dr. Timothy Kaiser](mailto:Timothy.Kaiser@nrel.gov)
  or [Dr. Kinshuk Panda](mailto:Kinshuk.Panda@nrel.gov) to get a copy of this
  directory.

2. We need to set a couple of environment variables
  ```bash
  export filename=r3_CO2_150n35_NoCCS2035_HPC
  export LIST=models.txt
  ```

3. We can now run the example as

  ```bash
  cd /scratch/$USER/plexos_array
  sbatch -A account_name -p short -t 4:00:00 --array=1-3 ./array.sh
  ```
