#Set MPI version and commands.  These can be overloaded by command line options
#for mpt:    make MOD="mpt  mkl cuda gcc/8.4.0"       MPICC=mpicc  MPIFC=mpif90
#for intel:  make MOD="intel-mpi/2020.1.217 mkl cuda" MPICC=mpiicc MPIFC=mpiifort
MOD=mpt  mkl cuda gcc/8.4.0
MPICC=mpicc
MPIFC=mpif90

FC=ifort

#MOD=intel-mpi/2020.1.217 mkl cuda
#MPICC=mpiicc
#MPIFC=mpiifort


#set the module and call make again
recurse: 
	source $$MODULESHOME/init/bash ; ml $(MOD)  ; $(MAKE) -f $(firstword $(MAKEFILE_LIST)) programs

#here we list the real targets
programs: phostone stf_01 f_ex02 c_ex02 mpigpu slowinvert purempi afile invertc in_list

phostone: phostone.c 
	$(MPICC) -fopenmp phostone.c -o phostone

invertc: invertc.c 
	gcc -fopenmp invertc.c -o invertc

purempi: phostone.c 
	$(MPICC) -DSTUBS phostone.c -o purempi

stf_01: stf_01.f90 
	$(MPIFC)  stf_01.f90 -o stf_01

f_ex02: f_ex02.f90
	$(MPIFC)  f_ex02.f90 -o f_ex02

slowinvert: slowinvert.f90
	$(FC)  -mkl -O3 -fopenmp slowinvert.f90 -o slowinvert

c_ex02: c_ex02.c
	$(MPICC)  c_ex02.c -o c_ex02

in_list:
	./doarray.py

mpigpu: hysub.cu hymain.c
	nvcc -c hysub.cu
	$(MPICC) hymain.c hysub.o -lcudart -lstdc++ -o mpigpu

afile:
	echo line1 > afile
	echo line2 >> afile

clean:
	rm -rf in_list afile phostone stf_01 f_ex02 c_ex02 mpigpu *mod slowinvert hysub.o purempi build invertc

install: recurse
	cp     in_list afile phostone stf_01 f_ex02 c_ex02 mpigpu doarray.py report.py slowinvert tymer purempi invertc ..

dist-clean: clean
	rm -rf  ../in_list ../afile ../phostone ../stf_01 ../f_ex02 ../c_ex02 ../mpigpu ../doarray.py ../report.py ../tymer ../slowinvert ../purempi ../invertc

#.PHONY: afile

