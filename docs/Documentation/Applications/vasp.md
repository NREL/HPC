## About VASP

VASP computes an approximate solution to the many-body Schrödinger equation, either within density functional theory (DFT), solving the Kohn-Sham equations, or within the Hartree-Fock (HF) approximation, solving the Roothaan equations. Hybrid functionals that mix the Hartree-Fock approach with density functional theory are implemented as well. Furthermore, Green's functions methods (GW quasiparticles, and ACFDT-RPA) and many-body perturbation theory (2nd-order Møller-Plesset) are available in VASP.

In VASP, central quantities, like the one-electron orbitals, the electronic charge density, and the local potential are expressed in plane wave basis sets. The interactions between the electrons and ions are described using norm-conserving or ultrasoft pseudopotentials, or the projector-augmented-wave method.

To determine the electronic ground state, VASP makes use of efficient iterative matrix diagonalization techniques, like the residual minimization method with direct inversion of the iterative subspace (RMM-DIIS) or blocked Davidson algorithms. These are coupled to highly efficient Broyden and Pulay density mixing schemes to speed up the self-consistency cycle.

For further details, documentation, forums, and FAQs, see the [VASP website](https://www.vasp.at/).

## Accessing VASP on NREL's HPC clusters

The VASP license requires users to be a member of a "workgroup" defined by the University of Vienna or Materials Design. If you are receiving "Permission denied" errors when trying to use VASP, you must be made part of the "vasp" Linux group first. To join, please contact us with the following information:

- Your name
- The workgroup PI
- Whether you are licensed through Vienna (academic) or Materials Design, Inc. (commercial)
- If licensed through Vienna:
  - the e-mail address under which you are registered with Vienna as a workgroup member (this may not be the e-mail address you used to get an HPC account)
  - Your VASP license ID
- If licensed through Materials Design,
  - proof of current licensed status

Once status can be confirmed, we can provide access to our VASP builds on Eagle, Swift or Vermilion. 

## Supported Versions

NREL offers support for VASP 5 and VASP 6 on CPUs, and a GPU build on Eagle and Vermilion as well.
- For CPUs, VASP can be built with either Intel compilers/Intel MPI or GNU compilers/Open MPI. In general the Intel MPI builds are faster and are reoccommended over the Open MPI builds.
- For GPUs, VASP can be built with Cuda or with OpenACC. The OpenACC GPU-port of VASP was released with VASP 6.2.0, and the Cuda GPU-port of VASP was dropped in VASP.6.3.0. The OpenACC build shows significant performance improvement compared to the Cuda build, but is more susceptible to running out of memory. 

|                    |     Eagle     |     Swift     |   Vermilion   |
| ------------------ | ------------- | ------------- | ------------- |
| VASP 5 (Intel MPI) |       x       |               |       x       |
| VASP 6 (Intel MPI) |       x       |       x       |       x       |
| VASP 6 (Open MPI)  |       x       |       x       |       x       |
| GPU VASP (Cuda)    |       x       |               |       x       |
| GPU VASP (OpenAcc) |       x       |               |       x       | 

Three distinct executables have been made available with each build of VASP:

1. vasp_gam is for Gamma-point-only runs typical for large unit cells;

2. vasp_std is for general k-point meshes with collinear spins; and,

3. vasp_ncl is for general k-point meshes with non-collinear spins.

```vasp_gam``` and ```vasp_std``` builds include the alternative optimization and [transition state theory tools from University of Texas-Austin](http://theory.cm.utexas.edu/vtsttools/) developed by Graeme Henkelman's group, and [implicit solvation models from the University of Florida](http://vaspsol.mse.ufl.edu/) developed by Mathew and Hennig.

## Getting Started

VASP is available through modules on Eagle, Swift and Vermilion. Use the command ```module avail vasp``` to view the versions of VASP available on each cluster. The module marked with "(D)" is the default module - slurm will chose to load this module if you run the command ```module load vasp```. To load other vasp modules, include the entire module name in the module load command, such as ```module load vasp/5.4.4_centos77```.

On Eagle, you should see:
```
[user@el1 ~]$ module avail vasp

------------------------------------------------------------------- /nopt/nrel/apps/modules/test/modulefiles --------------------------------------------------------------------
   vasp/5.4.4_raptor    vasp/6.1.0    vasp/6.1.1-openmpi    vasp/6.3.1-nvhpc_acc (D)

------------------------------------------------------------------ /nopt/nrel/apps/modules/default/modulefiles ------------------------------------------------------------------
   vasp/5.4.4_centos77    vasp/6.1.1    vasp/6.1.2

```

On Swift, you should see:
```
[clarson@swift-login-1 ~]$ module avail vasp

------------------------------------------ /nopt/nrel/apps/210928a/level02/modules/lmod/linux-rocky8-x86_64/openmpi/4.1.1-mkxx6h3/Core ------------------------------------------
   vasp/6.1.1-qb5uihl

----------------------------------------------------- /nopt/nrel/apps/210928a/level01/modules/lmod/linux-rocky8-x86_64/Core -----------------------------------------------------
   vaspintel/1.0-mkr32z6
```

On Vermilion, you should see:
```
[clarson@vs-login-1 ~]$ module avail vasp

------------------------------------------------------------------- /nopt/nrel/apps/modules/test/modulefiles --------------------------------------------------------------------
   vasp/5.4.4    vasp/6.1.1-openmpi    vasp/6.3.1-nvhpc_acc    vasp/6.3.1 (D)
```

## Example Job Scripts



## Troubleshooting

VASP known issues
- VASP on Vermilion can crash on multi-node jobs (see workarounds in Vermilion section) - OpenMPI more stable for multi-node jobs
- OpenACC build eats up more memory but is faster (Cuda build doesn't eat up as much memory)
- VASP 6.1.1 (or maybe it's 6.1.2?) crashes hse calculations (due to specific version of compiler and vasp) - use a newer version instead

