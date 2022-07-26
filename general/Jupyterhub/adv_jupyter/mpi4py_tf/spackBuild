#!/bin/bash
#SBATCH --job-name="jupyter"
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --partition=short

# Make a python/jupyter/mpi4py/pandas/tensorflow/cupy environment using spack.

# We also install a bare version of R with Rmpi. The R and Python versions of 
# MPI should work together.  See: https://github.com/timkphd/examples/tree/master/mpi/mixedlang
# for examples

# **********  Install directory **********
# Suggest you do this in /projects.
# It will not fit in home and scratch
# will go away.
export IDIR=/projects/hpcapps/tkaiser
cd $IDIR

# The actual install directory is a combination of the
# directory "IDIR" which you must set and a date/time
# stamp.  Thus you can run this script multiple times
# to create multiple copies.

# This script worked properly on Eagle on 05/27/2021.
# Spack can be a bit finicky with versions changing 
# over time.  Read: "This worked on 05/27/2021 but
# it might not work in the future."


# R needs jdk-11.0.10 to build.  The spack install of
# jdk-11.0.10 sometimes fails.  So we point to a version
# installed on Eagle by creating the file:
# spack/etc/spack/packages.yaml  See below.


#If you don't have tymer use this poor man's version
command -v tymer >/dev/null 2>&1 || alias tymer='python -c "import sys ;import time ;print(time.time(),time.asctime(),sys.argv[1:])" '
#You can get the full version from
#https://raw.githubusercontent.com/timkphd/examples/master/tims_tools/tymer

# This is where tymer will put its data so we clean it out
rm ~/sbuild 


tymer ~/sbuild start

export TMP=/scratch/$USER/tmp
export TMPDIR=/scratch/$USER/tmp

mkdir -p $TMP

module purge

##### Install R ?
#false
true
DOR=$?
if [ $DOR -eq 0 ] ; then   echo Will install R ;fi

# conda is needed for git, it also has cmake
ml conda

# If we use Intel MPI load it here but we want
# to unload Intel compilers
#ml intel-mpi/2020.1.217
#module unload comp-intel/2020.1.217

#load mpt version of MPI if not using IntelMPI (loaded above)
ml mpt/2.23

#Our base compilers
ml gcc/10.1.0 cuda/11.2 cudnn/8.1.1/cuda-11.2 

# Our install directory is a date/time stamp

MYBASE=`date +"%m%d%H%M%S"`
mkdir $MYBASE
cd $MYBASE
export BLDIR=`pwd`
echo "Building in:" $BLDIR

# Make a copy of our script
cat $0 > build_script

tymer ~/sbuild start spack
git clone https://github.com/spack/spack.git
cd spack

# Install of openjdk is a bit iffy.  Use a local copy
cat > pack << HERE
packages: 
  all: 
    providers: 
      openjdk: 
      - openjdk 
  openjdk: 
    buildable: false 
    externals: 
    - spec: openjdk 
      prefix: /nopt/nrel/apps/openmpi/4.1.0-gcc-8.4.0/jdk-11.0.10
HERE

export PATH=/nopt/nrel/apps/openmpi/4.1.0-gcc-8.4.0/jdk-11.0.10/bin:$PATH
export LD_LIBRARY_PATH=/nopt/nrel/apps/openmpi/4.1.0-gcc-8.4.0/jdk-11.0.10/lib:$LD_LIBRARY_PATH

mv pack etc/spack/packages.yaml

. share/spack/setup-env.sh 

tymer ~/sbuild done setup

#spack install cmake

#tymer ~/sbuild done cmake

cd $BLDIR

spack install python@3.9.5

tymer ~/sbuild done python

if [ $DOR -eq 0 ] ; then
  spack install r@4.1.0
  tymer ~/sbuild done r
fi


#spack install git

#tymer ~/sbuild done git

# conda is needed for git but now we have our own
#module unload conda

#cd spack/share/spack/modules/linux-centos7*
cd spack/share/spack/lmod/linux-centos7*
export NEWMOD=`pwd`
module use $NEWMOD
pwd
ls

ml `ls | grep python-3.9`
if [ $DOR -eq 0 ] ; then
  ml `ls | grep r-4`
fi
#ml `ls | grep git`
#ml `ls | grep cmake`

cd $BLDIR

tymer ~/sbuild done load

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py 

tymer ~/sbuild done pip

pip install jupyter matplotlib scipy pandas xlwt dask
pip install jupyterlab

tymer ~/sbuild done jupyter


# slurm magic commands
pip install git+https://github.com/NERSC/slurm-magic.git

pip --no-cache-dir install mpi4py



tymer ~/sbuild done mpi4py

if [ $DOR -eq 0 ] ; then
curl --insecure https://cran.r-project.org/src/contrib/Rmpi_0.6-9.1.tar.gz -o Rmpi.tar.gz

MY_MPI_PATH=`which mpicc| sed s,/bin/mpicc,,`

echo MY_MPI_PATH= $MY_MPI_PATH

#Yes, this is the correct type; that is not MPICH2.
export TYPE=OPENMPI
#Note the library we need for IntelMPI is in ..../release
ml
#Not sure why srun is needed here but for IntelMPI it crashes without it.
srun -n 1 R CMD INSTALL --configure-args="\
--with-Rmpi-include='$MY_MPI_PATH/include' \
--with-Rmpi-libpath='$MY_MPI_PATH/lib' \
--with-mpi='$MY_MPI_PATH/bin/mpicc' \
--with-Rmpi-type='$TYPE'"  \
Rmpi.tar.gz    

tymer ~/sbuild done Rmpi

fi

pip --no-cache-dir install tensorflow==2.5.0
pip --no-cache-dir install tensorflow-gpu==2.5.0
pip --no-cache-dir install horovod[tensorflow]==0.22.0

tymer ~/sbuild done tensorflow

# This is a hack.  cupy is looking for libcusolver.so.10
# but it works by linking to version 11.
if [ ! -e ~/lib/libcusolver.so.10  ] ; then
    mkdir -p ~/lib
    ln -s /nopt/nrel/apps/cuda/11.2/targets/x86_64-linux/lib/libcusolver.so libcusolver.so.10
fi

export CFLAGS=-L/nopt/nrel/apps/cuda/11.2/targets/x86_64-linux/lib
export LDFLAGS="-L/nopt/nrel/apps/cuda/11.2/targets/x86_64-linux/lib -L/nopt/nrel/apps/cuda/11.2/compat"
export LIBRARY_PATH=/nopt/nrel/apps/cuda/11.2/targets/x86_64-linux/lib:
export LD_LIBRARY_PATH=/nopt/nrel/apps/cuda/11.2/targets/x86_64-linux/lib:/nopt/nrel/apps/cudnn/8.1.1-cuda-11.2/lib64:/nopt/nrel/apps/cuda/11.2/lib64:/nopt/mpi/mpt-2.23/lib:/nopt/slurm/current/lib:$HOME/lib:$LD_LIBRARY_PATH


pip --no-cache-dir install cupy==9.0.0

tymer ~/sbuild done cupy


#Add Tim's thread mapping module
wget  https://raw.githubusercontent.com/NREL/HPC/master/slurm/source/setup.py
wget  https://raw.githubusercontent.com/NREL/HPC/master/slurm/source/spam.c
python3 setup.py install

tymer ~/sbuild done spam


echo "TO USE:"
echo "export LD_LIBRARY_PATH="$LD_LIBRARY_PATH
echo ""
echo "source $IDIR/$MYBASE/spack/share/spack/setup-env.sh"
echo ""
echo "module use " $NEWMOD
echo ml `ml 2>&1 | grep 1 | sed "s/.)//g"`
echo
echo "YOU SHOULD BE ABLE TO SHORTEN LD_LIBRARY_PATH"
echo
echo "Suggestion, Try this:"
#echo "  Run the \"source line\""
echo "module use " $NEWMOD
echo "  Run the \"ml line\""
echo " Then:"
echo export LD_LIBRARY_PATH=$HOME/lib:\$LD_LIBRARY_PATH

# Copy our output to our build directory
cp $SLURM_SUBMIT_DIR/slurm-$SLURM_JOBID.out $BLDIR


