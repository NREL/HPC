
## Run Magma

# use a login node

* if your node command prompt looks like `$USR@n0000`, you need to exit your interactive job
```bash
[cbarrows@n1009 OneWeek]$ exit
logout

qsub: job 3219389 completed
[cbarrows@login2 OneWeek]$ 
```

## Run MAGMA on a login node

```bash
cd ~
git clone https://github.com/NREL/MAGMA.git
cd /scratch/$USER/MSPCM-Workshop/Workshop-Explorations/OneWeek
module use /nopt/nrel/apps/modules/candidate/modulefiles
module purge
module load epel/6.6 R/3.2.2 pandoc/1.19.2.1
xvfb-run -a Rscript run_html_output_rts_DA.R
```

## Use scp in git-bash to copy the html to your local machine and view

* open a new git-bash prompt (local)

* change `$USER` _(2 spots)_ to your HPC username in the below command

```bash
scp "$USER@hpcsh.nrel.gov:/scratch/$USER/MSPCM-Workshop/Workshop-Explorations/OneWeek/Model\ DAY_AHEAD\ Solution/HTML_output_DA.html" .

start HTML_output_DA.html
```

[Example Magma output](https://gridmod.github.io/MSPCM-Workshop/HTML_output_DA.html) is posted in github pages.
