#!/usr/bin/env python
# numpy is required
import numpy
import sys
from numpy import *
# mpi4py module
from mpi4py import MPI
from time import sleep
from os import getenv
# Initialize MPI
comm=MPI.COMM_WORLD

# Get information about this run
numprocs=comm.Get_size()
myid=comm.Get_rank()
name = MPI.Get_processor_name()
# get the MPI version and try to 
# clean up the description a bit.
lib=MPI.Get_library_version()
version=MPI.Get_version()
version=str(version)
version=version.replace(",",".")
version=version.replace(" ","")
tag=" "
if myid == 0:
	print("Tasks: ",numprocs," MPI Version ",version)
	print("Running MPI libary ",lib)
# tag is just an optional command line argument that
# will be printed by each process. If it is an integer
# sleep for that number of seconds.
	if len(sys.argv) > 1:
		tag=sys.argv[1]

tag=comm.bcast(tag, root=0)
try :
# To force a task to a specific core...
# See [spam.c setup.py] for  the source for 
# the spam module.  It contains  code to get 
# the calling process's core and to  force it 
# to a particular core.
# In particular to build the module:
# Load your conda environment then:
# python setup.py install
# cp build/lib.*/*so .
# or set your PYTHONPATH variable to point
# to the directory containing the library.

	cores=36
	from spam import *
	offset=getenv('OFFSET','0')
	offset=int(offset)
	forcecore((myid+offset) % cores)
	core=findcore()
except:
	core=-1
print('xxxxxx Hello from {0:2d} on {1:10s},{2:4d}  {3:s}'.format(myid,name,core,tag))

try:
	tag=int(tag)
except:
	tag=0

sleep(tag)

MPI.Finalize()


