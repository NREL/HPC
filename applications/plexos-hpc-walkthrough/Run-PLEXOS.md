# Running RTS-GMLC Example Workflow

**NOTE:**
Please follow the instructions in [`Setup-PLEXOS.md`](Setup-PLEXOS.md) before
running the example in this directory.

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
`one_week_model.xml` file from Example 2, but for the sake of completness, we
will download a tarball of the same from a different repo in our batch script.

1. Get batch script by running the command

  ```bash
  wget https://github.nrel.gov/raw/tkaiser2/plexos/master/scripts/simple
  ```

  The batch file will look something like the one [here](RunFiles/simple)


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

1. Get the batch script using

  ```bash
  wget https://github.nrel.gov/raw/tkaiser2/plexos/master/scripts/enhanced
  ```

  The batch file will look something like the one [here](RunFiles/enhanced)

2. Make sure that you specify custom paths for on lines 10 & 11 which write the
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
jobs on Eagle.


# Old instructions for one week that are available on the main fork already

2. Create a symbolic link to the timeseries datafiles, environment and python script
    ```bash
    ln -s ../../RTS-GMLC-Dataset/timeseries_data_files timeseries_data_files
    ln -s ../../plexos-hpc-walkthrough/env-7.4.2.sh .
    ln -s ../../plexos-hpc-walkthrough/get_week.py .
    ```

3. Get yourself an interactive node
 ```bash
 salloc -N 1 -t 60 -A hpcapps -p debug
 ```

4. Setup your environment
  ```bash
  $ source env-7.4.2.sh
  $ cat env-7.4.2.sh
  module purge
  module load centos mono/4.6.2.7 xpressmp/8.0.4 plexos/7.400.2 conda
  export PYTHONPATH=/nopt/nrel/apps/plexos/plexos-coad
  export PLEXOS_TEMP=/scratch/$USER/tmp/$PBS_JOBID
  export TEMP=$PLEXOS_TEMP
  mkdir -p $PLEXOS_TEMP
  $ module list

  Currently Loaded Modules:
    1) centos/7.7     3) xpressmp/8.0.4   5) conda/mini_py37_4.8.3
    2) mono/4.6.2.7   4) plexos/7.400.2

  ```

5. Cut out one week to run DAY_AHEAD model on

  ```bash
  python get_week.py
  cat get_week.py
  ##! /usr/bin/env python
  #import datetime
  #
  #from coad.COAD import COAD
  #from coad.ModelUtil import datetime_to_plex
  #
  #coad = COAD('RTS-GMLC.xml')
  #
  ##7/14/2024
  #date_start = str(datetime_to_plex(datetime.datetime(2024, 7, 14, 0, 0)))
  #new_horizon = coad['Horizon']['Base'].copy("Interesting Week")
  #new_horizon["Step Count"] = "8"
  #new_horizon["Date From"] = date_start
  #new_horizon["Chrono Date From"] = date_start
  #new_horizon["Chrono Step Count"] = "8"
  #coad['Model']['DAY_AHEAD'].set_children(new_horizon)
  #
  #coad.save("one_week_model.xml")
  ```

6. Run PLEXOS

  ```bash
  mono $PLEXOS/PLEXOS64.exe -n "one_week_model.xml" -m DAY_AHEAD
  ```
