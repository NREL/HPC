# Additional information

## Known Issues

- `openmpi5` works only with `mpirun` because Slurm is configured without PMIx.
- `mpiifx` previously failed because of missing path definitions. This should be fixed; please send a report to [hpc-help@nlr.gov](mailto:hpc-help@nlr.gov) if it still fails.

## Frequently Asked Questions (FAQ)

### I cannot find the module I need.

Use `module spider ModuleName` first to verify availability. If it still cannot be found, email [hpc-help@nlr.gov](mailto:hpc-help@nlr.gov).

### I need to mix and match compilers and libraries/MPI. How can I do that?

Modules do not support arbitrary mix-and-match combinations. For example, if `oneapi` is loaded, only software compiled with oneAPI is shown. For custom stacks, use Spack and contact [hpc-help@nlr.gov](mailto:hpc-help@nlr.gov) to be matched with a Spack expert.

### Can I use Miniforge with other modules?

It is technically possible, but Miniforge is intended to provide an isolated environment. Module load order can affect `PATH` and `LD_LIBRARY_PATH`.

### What if I want a different CUDA version?

Other CUDA versions are available under Core modules. Contact [hpc-help@nlr.gov](mailto:hpc-help@nlr.gov) for additional versions. Note that CUDA modules under Core do not automatically expose CUDA-enabled software; CUDA modules under Base do.

### Should my modulefile be Lua or Tcl?

Lua modulefiles are recommended, but Tcl modulefiles should also work.

### Why doesn't `ml conda` or `ml mamba` work?

Use `ml miniforge3` or `ml anaconda3` to enable conda and mamba.
