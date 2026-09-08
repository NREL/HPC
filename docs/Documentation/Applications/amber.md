# Amber

**Documentation:** [Amber Reference Manual](https://ambermd.org/doc12/Amber26.pdf)

*Amber is a suite of biomolecular simulation programs. On Kestrel, the `pmemd` molecular dynamics engine is installed with GPU (CUDA) and CPU/MPI support.*
The scope of this documentation is limited to the [RHEL9](https://natlabrockies.github.io/HPC/RHEL9_upgrade/) nodes on Kestrel, where both CPU and GPU versions of Amber are installed on GPU stack, currently accessible through the `kl5` login node. An older CPU-only version of Amber exists on the RHEL8 nodes. 

!!! tip "Licensing"
	Amber is licensed software. The Kestrel install accepted the
	[non-commercial license](https://ambermd.org/GetAmber.php), which covers
	non-commercial use. Commercial use requires a separate license.

## Getting Started

The `amber` module provides `pmemd` (CPU, serial), `pmemd.MPI` (CPU, MPI),
and the CUDA-accelerated `pmemd.cuda`/`pmemd.cuda.MPI` (single- and
multi-GPU MD). netCDF-Fortran tools (`ncdump`, `ncgen`, `nf-config`) are
also included.

!!! note
	This build is `pmemd`-only and does **not** include AmberTools
	(`sander`, `tleap`, `cpptraj`, `MMPBSA.py`, etc.). For AmberTools, load
	`miniforge3` and follow the
	["Binary distribution via conda"](https://ambermd.org/GetAmber.php)
	instructions, or contact HPC-Help.

```
module use /nopt/nlr/apps/kestrel-gpu/apps_modules/26.05
module load amber/26-craype-gnu-cuda
```

This auto-loads the required Cray PE toolchain (`PrgEnv-gnu`, `cray-mpich`,
etc.) and sets `AMBERHOME`, `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH`. It
works from login, CPU, or GPU nodes — a GPU is only needed at run time for
the `pmemd.cuda*` executables.

### Example Job Scripts

??? example "Kestrel GPU — single GPU"

	```slurm
	#!/bin/bash
	#SBATCH --account=<alloc>
	#SBATCH --partition=gpu-h100
	#SBATCH --nodes=1
	#SBATCH --ntasks=1
	#SBATCH --cpus-per-task=32
	#SBATCH --gres=gpu:1
	#SBATCH --mem=80G
	#SBATCH --time=24:00:00

	module use /nopt/nlr/apps/kestrel-gpu/apps_modules/26.05
	module load amber/26-craype-gnu-cuda

	srun pmemd.cuda -O -i mdin -o mdout -p prmtop -c inpcrd \
	                -r restrt -x mdcrd -inf mdinfo
	```

??? example "Kestrel CPU — MPI"

	```slurm
	#!/bin/bash
	#SBATCH --account=<alloc>
	#SBATCH --partition=standard
	#SBATCH --nodes=1
	#SBATCH --ntasks=64
	#SBATCH --cpus-per-task=1
	#SBATCH --mem=128G
	#SBATCH --time=24:00:00

	module use /nopt/nlr/apps/kestrel-gpu/apps_modules/26.05
	module load amber/26-craype-gnu-cuda

	srun pmemd.MPI -O -i mdin -o mdout -p prmtop -c inpcrd \
	                -r restrt -x mdcrd -inf mdinfo
	```

!!! note
	If you launch `pmemd` or `pmemd.cuda` (the non-MPI serial binaries) with
	`srun` inside an existing allocation, add `--mpi=none` or you may see a
	`PMI2_Job_GetId` error. This does not apply to `pmemd.MPI` /
	`pmemd.cuda.MPI`, which are true MPI programs.

## Choosing GPU precision

`pmemd.cuda` defaults to **SPFP** (hybrid single precision + fixed-point),
recommended for production MD. Full double precision is available as
`pmemd.cuda_DPFP` / `pmemd.cuda_DPFP.MPI` for the rare case that requires it.

## Performance guidance

Per Amber's own guidance and confirmed by benchmarking on Kestrel H100s:

- **A single H100 GPU is almost always the fastest option** — for a
  23,558-atom DHFR-in-water (JAC) benchmark, 1 GPU (432 ns/day) beat a full
  128-core, 2-node CPU job (53 ns/day) by ~8×.
- **Prefer running N independent single-GPU jobs over one N-GPU job.**
  Multi-GPU (intra- or inter-node) only pays off for large systems
  (~90k+ atoms); for smaller systems it is slower than 1 GPU.
- On CPU, parallel efficiency for `pmemd.MPI` falls off past ~32-64 ranks
  per node (JAC case: 57% efficiency at 32 ranks vs 27% at 128 ranks across
  2 nodes) — run multiple independent jobs rather than one oversized job.

## Advanced

Full build/toolchain details, verified scaling tables (1/2/4-GPU and CPU
scaling), reference-output verification, and troubleshooting are in the
[maintainer notes](https://ambermd.org/GPUHowTo.php) and the install's own
guide at `/nopt/nlr/apps/kestrel-gpu/software/amber/amber-26-guide.md`.

## Troubleshooting

**`STOP PMEMD Terminated Abnormally!` with no other output**
Missing/incorrect `-i mdin -c inpcrd -p prmtop` arguments, or one of those
files is unreadable.

**`cudaMemcpyToSymbol: ... all CUDA-capable devices are busy`**
Check that your job actually requested GPUs via `--gres=gpu:N`.

**Module load fails to find `PrgEnv-gnu`**
Use a login shell (`bash -l`/`bash -lc`) so Kestrel's `/etc/profile.d/*.sh`
and the Lmod module tree are sourced.

## Support

- Amber user forum: <http://archive.ambermd.org/>
- Amber reference manual: <https://ambermd.org/doc12/Amber26.pdf>
- Amber GPU HowTo: <https://ambermd.org/GPUHowTo.php>
