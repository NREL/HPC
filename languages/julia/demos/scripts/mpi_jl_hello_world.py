# This is an example of how to integrate python's MPI wrapper, mpi4py, with
# julia's MPI wrapper, MPI.jl.  This is a simple 'Hello World' example.
#
# We make use of pyjulia which gives access to Julia from Python. For some
# basic usage of pyjulia see the jupyter notebook PyJulia_Demo.ipynb in
# Julia_HPC_Guides/demos/notebooks. For more details (including installation
# instructions) see the pyjulia documentation:
#     https://pyjulia.readthedocs.io/en/latest/
# For more details on mpi4py and MPI.jl, see their documentation:
#     https://mpi4py.readthedocs.io/en/stable/
#     https://juliaparallel.github.io/MPI.jl/stable/
#
# WARNING: Before running this script, you must setup mpi4py and MPI.jl to use
# the SAME MPI library. This is described in the README.md found in the same
# directory as this script.

# Something in the below julia interface setup forks a process--setup julia before
# importing mpi4py to keep MPI from complaining about the fork
from julia.api import Julia
jl = Julia(compiled_modules=False)

import mpi4py
from mpi4py import MPI as pympi
comm = pympi.COMM_WORLD

# Activate the desired julia environment
jl.using("Pkg")
from julia import Pkg
Pkg.activate(".")

# Make julia global space available
from julia import Main

jl.using("MPI")
from julia import MPI as jlmpi

# Initialize Julia MPI without initializing the libmpi--this is part of the
# initialization done by MPI.Init() in MPI.jl
jl.eval(
    'function init_mpi(); for f in MPI.mpi_init_hooks; f(); end; return; end;'
)
Main.init_mpi()

# Convert pympi comm to jlmpi comm
Main.handle = pympi._handleof(comm) # make handle accessible to julia
jl.eval('comm = MPI.Comm(MPI.MPI_Comm(handle))') # create julia comm

# WARNING: You might think that we could use a statement like
#     Main.comm = jlmpi.Comm(jlmpi.MPI_Comm(pympi._handleof(comm)))
# to turn the python MPI comm into a julia MPI comm instead of the above `eval`.
# However, this will fail when using MPICH (it works with OpenMPI). The reason
# is that MPICH uses integers to differentiate MPI comms (OpenMPI uses raw
# pointers) . So for MPICH, `jlmpi.MPI_Comm(pympi._handleof(comm))` returns a
# `Cint` (which is a specialized julia Int32 for interfacing with C/Fortran
# libraries). When it comes back to python, it is converted to a python `int`
# which is then converted to a Julia Int64 when given to `jlmpi.Comm` as an
# argument. The result is a type error. We can avoid this MPICH incompatibility
# by using the above `eval` statement.

rank = comm.Get_rank()
size = comm.Get_size()
print("Hello world, I am python rank {:d} of {:d}.".format(rank, size))
comm.Barrier()

Main.rank = jlmpi.Comm_rank(Main.comm)
Main.size = jlmpi.Comm_size(Main.comm)
jl.eval('println("Hello world, I am julia rank $rank of $size.")')
jlmpi.Barrier(Main.comm)
