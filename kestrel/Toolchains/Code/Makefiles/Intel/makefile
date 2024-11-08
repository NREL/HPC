SHELL:=/usr/bin/bash

recurse:
	. /nopt/nrel/apps/env.sh            ; \
	module purge                        ; \
	module load  intel-oneapi-compilers ; \
	module load intel-oneapi-mpi        ; \
	module load gcc-stdalone            ; \
	$(MAKE) -f $(firstword $(MAKEFILE_LIST)) intel gnu rmo

intel: ex_f ex_c 
gnu: gex_f gex_c 

#Intel mpi  with intel compilers
PF90=mpiifort
PCC=mpiicc
ICC=icx

#Uncomment the next two lines to build with icc instead of icx
#ICC=icc
#XOPT=-diag-disable=10441

#Intel mpi with gcc and gfortran
PG90=mpif90
PGCC=mpicc

OPS=-O3 -g -fopenmp


triadi.o: triad.c
	$(ICC) $(OPS) $(XOPT) -c triad.c -o triadi.o

triadg.o: triad.c
	gcc $(OPS) -c triad.c -o triadg.o
	

ex_f: ex1.f90 triadi.o
	$(PF90) $(OPS)       ex1.f90  triadi.o -o ex_f

ex_c: export I_MPI_CC=$(ICC)
ex_c: ex1.c 
	$(PCC) $(OPS) $(XOPT)  ex1.c  triadi.o -o ex_c

gex_f: ex1.f90 triadg.o
	$(PG90) $(OPS)  ex1.f90 triadg.o  -o gex_f

gex_c: ex1.c triadg.o
	$(PGCC) $(OPS)       ex1.c  triadg.o -o gex_c


clean:rmo
	rm -rf  ex_f ex_c gex_f gex_c intel.tar exc exf

#clean up after the compile
rmo:
	rm -rf *o *mod*

tar:
	tar -cf intel.tar ex1.c ex1.f90 makefile script triad.c debug

