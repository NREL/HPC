# Plexos
The two sbatch files shown below will download a data set and
run a Plexos example.  These work on Eagle as of 07/24/2020.

They will create a new directory for each run.  

The scripts will pull a copy of the data set from this repository,
the week.tgz file. Unpack it and then run using the files.  

The data in week.tgz is generated following the instructions
for the second example on the page:

https://tickets.hpc.nrel.gov/collab/display/CSCT/Eagle+Plexos+Examples

This page gives additional instructions for setting up more
complete environment for running Plexos, but is overkill for our
purposed here.  

Both scripts will need at lease minor modifications to run for
an arbitrary user.  The line

#SBATCH --account=hpcapps

will need to be changed to point to your account on Eagle.

The "enhanced" script has many bells and whistles.  First
it puts all normal slurm output in a particular directory.

#SBATCH -o /scratch/USER/slurmout

This is done so that all stdout and stderr from all runs
will end up in the same place.  However, you will need to
change the path to a directory you own.  The same path 
needs to be set on the last line of the file.

Here is an easy way to create a new directory and modify
the script

mkdir -p /scratch/$USER/slurmout

sed -i.org s/USER/$USER/ enhanced

Optionally, you can have output go to a file logfile.xxxxx
where xxxxx is a date time stamp.  This can be enabled
by changing false to true on the indicated line.

We keep a record of the environment using the lines

printenv > env

ls -lt 

There have been instances of runs failing because of 
license issues.  This "enhanced" script will try multiple times
to run if there is a license failure.  It will also 
report the ability to see various machines using the
ping command.  This will only occur on failure.  You
can also be notified of failures by uncommenting the
"mail" line.


You can download these scripts from the page:

https://github.nrel.gov/tkaiser2/plexos

or 

wget https://github.nrel.gov/raw/tkaiser2/plexos/master/scripts/enhanced

and 

wget https://github.nrel.gov/raw/tkaiser2/plexos/master/scripts/simple

## To run:

Edit the files simple and enhanced as shown above.  Then:

sbatch simple

sbatch enhanced


## Notes
- The files in the week.tgz tar ball will be deleted since we have copies of them.
- The program tymer is a simple timing routine.  "tymer -h" will show usage.  


# simple
```
#!/bin/bash 
#SBATCH --job-name="4plexos"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:15:00
#SBATCH --partition=debug
### You need to change the next line
#SBATCH --account=hpcapps


mkdir $SLURM_JOB_ID
cat $0 > $SLURM_JOB_ID/script
cd $SLURM_JOB_ID


export MAX_TEMP_FILE_AGE=50
export PLEXOS_TEMP=`pwd`/plexos_temp
export TEMP=$PLEXOS_TEMP

mkdir -p $PLEXOS_TEMP

## Select our version...

export PLEXOS_VERSION="8.0"
export XPRESSMP_VERSION="8.5.6"
module purge
module load centos mono/4.6.2.7 xpressmp/$XPRESSMP_VERSION plexos/$PLEXOS_VERSION

# Get our data
wget https://github.nrel.gov/tkaiser2/plexos/raw/master/week.tgz
tar -xzf week.tgz
ls -lt 

mono $PLEXOS/PLEXOS64.exe -n "one_week_model.xml" -m DAY_AHEAD &> mono_log.$SLURM_JOB_ID || echo "mono fail"

# Remove the files in the *tgz file.  We don't need them anymore.
tar -tzf week.tgz | grep -v tymer |sed "s,/.*,," | sort -u  | while IFS= read -r line ; do rm -rf "$line" ; done

# Copy slurm stderr & stdout to this directory.  
cp ../std*.$SLURM_JOB_ID . || echo "No std*.$SLURM_JOB_ID"

```

# enhanced

```
#!/bin/bash 
#SBATCH --job-name="4plexos"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:15:00
#SBATCH --partition=debug
### You need to change the next line
#SBATCH --account=hpcapps
### You will need change the two lines to point to a directory you own
#SBATCH -o /scratch/USER/slurmout/stdout.%j
#SBATCH -e /scratch/USER/slurmout/stderr.%j


mkdir $SLURM_JOB_ID
cat $0 > $SLURM_JOB_ID/script
cd $SLURM_JOB_ID

# Next  lines are not required but will force all output to logfile.* including
# slurm stdout and stderr.  To activate/deactivate set to true/false.
if false ; then
exec 3>>logfile.`date +"%y%m%d%H%M%S"` 
exec 4>&3 
exec 5>&1 6>&2              # save "pointers" to stdin and stdout
exec 1>&3 2>&4              # redirect stdin and stdout to file
fi



export MAX_TEMP_FILE_AGE=50
export PLEXOS_TEMP=`pwd`/plexos_temp
export TEMP=$PLEXOS_TEMP

mkdir -p $PLEXOS_TEMP

## Select our version...

export PLEXOS_VERSION="8.0"
export XPRESSMP_VERSION="8.5.6"
module purge
module load centos mono/4.6.2.7 xpressmp/$XPRESSMP_VERSION plexos/$PLEXOS_VERSION

# Get our data
wget https://github.nrel.gov/tkaiser2/plexos/raw/master/week.tgz
tar -xzf week.tgz

# What we have
printenv > env
ls -lt 
# Make multiple attempts in case we can't get a license
# To simplfy you can take out everything except the "mono"
# line in the for / done block
for attempt in a b c d e ; do
	export WAIT=120
# Time our run
    ./tymer times begining
    mono $PLEXOS/PLEXOS64.exe -n "one_week_model.xml" -m DAY_AHEAD &> mono_log.$SLURM_JOB_ID || echo "mono fail"
    ./tymer times finished
	grep "Unable to acquire license" mono_log.$SLURM_JOB_ID >& /dev/null
	if [ $? -eq 0 ] ; then
		date                              >> ping.out
		ping -c 3 $HOSTNAME               >> ping.out
                ping -c 3 eagle-dav.hpc.nrel.gov  >> ping.out
                ping -c 3 google.com              >> ping.out
                ping -c 3 10.60.3.188             >> ping.out
		cp mono_log.$SLURM_JOB_ID mono_log.$SLURM_JOB_ID.$attempt	
		#mail <  mono_log.$SLURM_JOB_ID -s $SLURM_JOB_ID  USER@nrel.gov	
		echo will try again in $WAIT seconds
		sleep $WAIT
	else
		echo "found it"
		break
	fi
done

# Remove the files in the *tgz file.  We don't need them anymore.
tar -tzf week.tgz | grep -v tymer |sed "s,/.*,," | sort -u  | while IFS= read -r line ; do rm -rf "$line" ; done

# Copy slurm stderr & stdout to this directory.  
# You will need to change the next line to point to the directory specified in the header.
cp /scratch/USER/slurmout/std*.$SLURM_JOB_ID . || echo "No std*.$SLURM_JOB_ID"

```

