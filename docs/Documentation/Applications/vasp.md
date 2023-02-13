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

