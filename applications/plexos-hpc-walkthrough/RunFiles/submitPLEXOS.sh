#!/bin/bash

# Example input to run the models listed in .txt in the .xml input file
# ./submitPLEXOS.sh r3_CO2_150n35_NoCCS2035_HPC.xml models.txt

############################ USER MODIFIED SECTION ########################
export runtime="08:00:00"
export alloc="hpcapps" #allcoation to use
export runscript="runPLEXOS.sh"
##########################################################################

name="${1%.*}"
models=$2

echo $name
mkdir -p ${name}

rootdir=$(pwd)
cd $rootdir

### set topdir if you want a directory for
### all jobs submitted at then same time
topdir=`date +"%y%m%d%H%M%S"`

while read line; do
if [ -n "$topdir" ] ; then
export mydir=${topdir}/${name}/${line}
echo running in $topdir
else
export mydir=${name}/${line}
fi

# clean out the directory if it exists
rm -rf ${mydir}
mkdir -p ${mydir}

cp "${runscript}" "$mydir/."

cp "${name}.xml" "$mydir/."

ln -fs "${rootdir}/data/" "$mydir/."

cd "$mydir/"

echo "++++++++"
pwd
ls -lt
echo "++++++++"

export SLURM_SUBMIT_DIR=$(pwd)
submitcommand="sbatch -A ${alloc} -t ${runtime} --export=filename="${name}",model="${line}" ${runscript} --qos=medium"
echo $submitcommand
$submitcommand
cd $rootdir

done < $models
