:<<++++

Author: Tim Kaiser

Build a new version of python mpi4py.
Works with IntelMPI, MPT, and OpenMPI, just change the module load commands.


USAGE:
    source jupyter.sh

 To use the new version after the initial Install
   module load conda
   source activate
   source activate $MYVERSION
   module load gcc/8.4.0 
   #module load intel-mpi/2020.1.217
   module load mpt

++++

### Build a new version of python with and Intel MPI version of mpi4py
CWD=`pwd`
export MYVERSION=dompt
cd ~
module load conda 2> /dev/null || echo "module load conda failed"
conda create --name $MYVERSION python=3.8 jupyter matplotlib scipy pandas xlwt dask -y

### Don't do conda init
### Just do source activate
source activate 
source activate $MYVERSION

which pip
which python

### Install mpi4py
module load gcc/8.4.0  2> /dev/null || echo "module load gcc failed"
#module load intel-mpi/2020.1.217  2> /dev/null || echo "module load mpi failed"
module load mpt  2> /dev/null || echo "module load mpi failed"
pip --no-cache-dir install mpi4py


### Install slurm magic commands 
pip install git+git://github.com/NERSC/slurm-magic.git

cat <<++++
In a jupyter activate slurm_magic with 
%load_ext slurm_magic
++++

cd $CWD

### Install core mapping module
python setup.py install


