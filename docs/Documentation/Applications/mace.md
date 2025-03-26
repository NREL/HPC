The Multi Atomic Cluster Expansion (MACE) is a neural network-based approach for atomic scale materials modeling. Using higher order equivariant message passing, MACE represents the state-of-the-art in machine-learned interatomic potentials (MLIPs), needing orders of magnitude less training data than non-equivariant approaches and enabling approximately a million times more energy and force evaluations than is possible with DFT with the same computational resources. Researchers can use this enhanced throughput to do material screenings, calculate time-averaged quantities over time scales relevant to experiment, or treat heterogeneous, multicomponent systems. The approach is implemented in two steps: 

1. The training step is done in a standalone software package from the MACE developers.

2. The evaluation step can be done with the standalone MACE package, but is more typically done within LAMMPS or ASE.

## Installing MACE standalone package

The Kestrel-optimized MACE conda environment can be found at: `/nopt/nrel/apps/gpu_stack/software/mace/environments`

Copy the `mace_kestrel.yml` to one of your own directories, and build the environment as from that directory as follows:

`$ module load conda`

`$ conda env create -n <your_mace_env_name> -f mace_kestrel.yml`

Then check that MACE is installed within the conda environment: 

`$ conda list mace`

Should show:
Name                    Version                   Build  Channel
mace-torch                0.3.5                    pypi_0    pypi

And check that mace-torch working by issuing the following commands: 

`$ python`
`>>> import torch`
`>>> x = torch.rand(5, 3)`
`>>> print(x)`
`>>> exit`

This should show the following after the print statement:
tensor([[0.1828, 0.4496, 0.8743],
        [0.9411, 0.3940, 0.1104],
        [0.9148, 0.7720, 0.7389],
        [0.6979, 0.0421, 0.1763],
        [0.6563, 0.7603, 0.8917]])

This environment is required for all commands in the MACE standalone package, including  those for training (`mace_run_train`), evaluation (`mace_eval_configs`) and creating lammps models (`mace_create_lammps_model`). 

## Generating training data
Training data for your MACE MLIP can be generated with any quantum mechanical software package (e.g. VASP, JDFTx, Quantum ESPRESSO, Q-Chem, etc.) This should be in an [extended XYZ format](https://www.ovito.org/manual/reference/file_formats/input/xyz.html) that contains the forces and energies in addition to the species and position. Training xyz files may be concatenated together, e.g.: 

```
137 
Lattice="8.298 0.000 0.000 4.149 7.186 0.000 0.000 0.000 30.000" Properties=species:S:1:pos:R:3:forces:R:3 energy=-112196.4817837208 
Pt   1.382959   0.798452   7.258370  -0.009086   0.070059  -0.293957 
Pt   2.765916   3.193808   7.258370   0.003288  -0.015350  -0.455340 
Pt   4.148873   5.589164   7.258370  -0.061542  -0.013544  -0.227623
[...] 
137 
Lattice="8.298 0.000 0.000 4.149 7.186 0.000 0.000 0.000 30.000" Properties=species:S:1:pos:R:3:forces:R:3 energy=-112225.6448941744 
Pt   1.382959   0.798452   7.258370  -0.008809   0.094532  -0.233627 
Pt   2.765916   3.193808   7.258370   0.011519  -0.015212  -0.446426 
Pt   4.148873   5.589164   7.258370  -0.130966  -0.038721  -0.129331 
[...]
[etc.]
```

## Training with MACE
After installing the MACE package and generating your training data, you are ready to train a MLIP using the script below. A complete description of the training parameters can be found on the [MACE github package website](https://github.com/ACEsuit/mace) or the [MACE documentation website](https://mace-docs.readthedocs.io/en/latest/guide/training.html).

```
#!/bin/bash
#SBATCH --account=<your-account-name> 
#SBATCH --nodes=1
#SBATCH --gpus=1 #Ngpus used per node
#SBATCH --ntasks-per-node=1 # =Ngpus used per node
#SBATCH --cpus-per-task=1 # =Ngpus
#SBATCH --time=04:00:00
#SBATCH --job-name=<your-job-name>
#SBATCH --mem=85G # =Ngpus*85G
  
conda activate your_mace_env_name

mace_run_train \
    --name="MACE_model" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --E0s='{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1484.1187695035828, 8:-2042.0330099956639}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=1500 \
    --swa \
    --start_swa=1200 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
```

Upon completion of this training, a MACE MLIP model file will be generated with a `.model` extension. This can be used for evaluation of the model within the standalone package or within ASE. If you plan on using your MACE MLIP to run molecular dynamics calculations using LAMMPS, you need to run `mace_create_lammps_model` to create a LAMMPS MACE MLIP model file with an added `-lammps.pt` extension: 

`$ mace_create_lammps_model my_mace.model`

The output will be: `my_mace.model-lammps.pt`

## Evaluation with MACE
Once your MACE MLIP is generated, you can evaluate atomic configurations specified in an xyz format within our provided MACE environment as follows:

```
mace_eval_configs \
    --configs="your_configs.xyz" \
    --model="your_model.model" \
    --output="./your_output.xyz"
```

Alternatively, you can use the `.model` file within [ASE detailed here](https://mace-docs.readthedocs.io/en/latest/guide/ase.html#running-md-simulations), or the `.pt` file within [LAMMPS detailed here](https://mace-docs.readthedocs.io/en/latest/guide/lammps.html#id2).