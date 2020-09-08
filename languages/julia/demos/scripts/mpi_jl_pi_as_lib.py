# This is an example of how to integrate python's MPI wrapper, mpi4py, with
# julia's MPI wrapper, MPI.jl. This example approximates pi using Monte Carlo
# integration as a 'library' call. It uses julia functions defined in
# 'pi_func.jl'.
#
# We make use of pyjulia which gives access to Julia from Python.  For some
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

import numpy as np

comm = pympi.COMM_WORLD

# Activate the desired julia environment
jl.using('Pkg')
from julia import Pkg
Pkg.activate(".")

jl.using('MPI')

from julia import Main
from julia import MPI as jlmpi

jl.eval('include("pi_func.jl")')
# Initialize jlmpi stuff without initializing libmpi again --
# function definition is in pi_func.jl
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


# We can use the below if we're using OpenMPI but not MPICH
# Main.comm = jlmpi.Comm(jlmpi.MPI_Comm(pympi._handleof(comm)))

rank = jlmpi.Comm_rank(Main.comm)

pi_est = Main.estimate_pi(Main.comm)

if rank == 0:
    print("pi estimate = {}".format(pi_est))

exit()
