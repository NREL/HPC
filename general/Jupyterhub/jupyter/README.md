# jupyter lab

Tools for working with jupyter lab on eagle

There are a couple of bash scripts that can be used for launching a jupyter lab instance on an eagle compute node and then connecting via your local web browser.

One example workflow is to use the `sbatch_jupyter.sh`, `auto_launch_jupyter.sh` scripts.

Create your conda environment:
```bash
conda create -n jupyterenv python=3.8 jupyterlab
```

Update the `sbatch_jupyter.sh` script with your account information and desired run time:

```bash
#!/bin/bash  --login

#SBATCH --account=<project handle>
#SBATCH --time=01:00:00

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36
```

Then, you can submit and monitor the job automatically with:

```bash
bash auto_launch_jupyter.sh
```

This will submit the `sbatch_jupyter.sh` script to eagle and then monitor the status, eventually providing you will the commands you need to connect. The output should look something like:

```
[<hpc-username>@el2 bash]$ bash auto_launch_jupyter.sh
Checking job status..
job still pending..
job is running!
getting jupyter information, hang tight..
okay, now run the follwing on your local machine:
ssh -L 7878:r1i6n1:7878 el2.hpc.nrel.gov
then, navigate to the following on your local browser:
http://127.0.0.1:7878/?token=ed23bab2d938f52122c8cccf833dbee185d62adbfe56d02b
```

