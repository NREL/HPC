#!/bin/bash
# This script runs PLEXOS on Eagle

# export filename=FILENAME
# export model=MODEL
# export time=TIME
# sbatch -A account -p partition -t time ./runPLEXOS.sh
# Or use a separate script that sets variables.
# and runs sbatch.

# This function tests if the PLEXOS license server can be seen
# It will check 5 times and error out if it is not seen. It
# does not check if your license file is valid.
license () {
for attempt in `seq 5` ; do
  ping -c 2 10.60.3.188 > /dev/null
  if [ $? -eq 1 ] ; then
    date
    echo -e "Can not see license server. \nWill try again in 60 seconds.\n"
  else
    return 0
  fi
  if [ $attempt -lt 5 ] ; then sleep 60 ; fi
  done
  date
  echo "license lookup failed - exiting"
  date >> license.fail
  hostname >> license.fail
  ping -c 2 10.60.3.188 >> license.fail
  ping -c 2 eagle-dav.hpc.nrel.gov >> license.fail
  exit
}

# Call the license check function
license
echo "found license server"

module use -a /nopt/nrel/apps/modules/default/modulefiles
module purge

# set the version to use - defaults to 8.2 or you
# can set it in the calling environment like you
# do for filename and model
case $version in
7.4) # loads PLEXOS 7.4
module load mono/4.6.2.7 xpressmp/8.0.4 centos plexos/7.400.2
;;
8.1) # loads PLEXOS 8.1
module load centos mono xpressmp/8.5.6 plexos/8.100R02
;;
*) # default to PLEXOS 8.2
module load centos mono/6.8.0.105 xpressmp/8.5.6 plexos/8.200R01
version=8.2
;;
esac

echo 'I am in ' $PWD ' submitting for PLEXOS ' $version

# MAX_TEMP_FILE_AGE is an environmental variable that Plexos uses to determine
# how old a temporary directory can be (in days) before it should be purged
# see http://wiki.energyexemplar.com/index.php?n=Article.AdvancedSettings
export MAX_TEMP_FILE_AGE=50

# PLEXOS_TEMP is an environmental variable that Plexos uses to store temporary files.
# Plexos creates subdirectories in this directory for each run.
# If the subdirectory gets deleted during the run then the run will fail when it
# tries to write the solution file.
export PLEXOS_TEMP=/scratch/$USER/tmp/$SLURM_JOBID
export TEMP=$PLEXOS_TEMP

#make sure the PLEXOS_TEMP and TEMP directories exist
mkdir -p $PLEXOS_TEMP $TEMP


## Run PLEXOS model
plexos_command="mono $PLEXOS/PLEXOS64.exe -n "${filename}.xml" -m "${model}""
echo $plexos_command
$plexos_command

## move zip files (keep in folder by xml so as not to overwrite solutions)
cd ../..
mkdir -p ${filename}_solutions
echo 'Moving zip files'
mv "${filename}/${model}/Model ${model} Solution/Model ${model} Solution.zip" "${filename}_solutions/Model ${model} Solution.zip"

## create h5 files using julia script
export doh5=0
if [ "$doh5" == 1 ] && [ -a runH5.jl ] ;then
echo "Running h5plexos"
module use /home/gstephen/apps/modules
module load julia
julia runH5.jl ${filename}_solutions ${model}
else
echo "Skipping h5plexos"
fi
