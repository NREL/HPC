# Running RTS-GMLC Example Workflow

1. Move into the model directory
 ```
 cd /scratch/$USER/MSPCM-Workshop/Workshop-Explorations/OneWeek
 ```

2. Create a symbolic link to the timeseries datafiles, environment and python script
 ```
 ln -s ../../RTS-GMLC-Dataset/timeseries_data_files timeseries_data_files
 ln -s ../../plexos-hpc-walkthrough/env-7.3.3.sh .
 ln -s ../../plexos-hpc-walkthrough/get_week.py .
 ```

3. Get yourself an interactive node
 ```
 salloc -N 1 -t 60 -A hpcapps -p debug
 ```

4. Setup your environment
 ```
$ source env-7.4.2.sh
$ cat env-7.4.2.sh
module purge
module load centos mono/4.6.2.7 xpressmp/8.0.4 plexos/7.400.2 conda # coad
export PLEXOS_TEMP=/scratch/$USER/tmp/$PBS_JOBID
export TEMP=$PLEXOS_TEMP
mkdir -p $PLEXOS_TEMP
$ module list

Currently Loaded Modules:
  1) centos/7.7     3) xpressmp/8.0.4   5) conda/mini_py37_4.8.3
  2) mono/4.6.2.7   4) plexos/7.400.2



 ```

5. Cut out one week to run DAY_AHEAD model on

 ```bash
python get_week.py
cat get_week.py
##! /usr/bin/env python
#import datetime
#
#from coad.COAD import COAD
#from coad.ModelUtil import datetime_to_plex
#
#coad = COAD('RTS-GMLC.xml')
#
##7/14/2024
#date_start = str(datetime_to_plex(datetime.datetime(2024, 7, 14, 0, 0)))
#new_horizon = coad['Horizon']['Base'].copy("Interesting Week")
#new_horizon["Step Count"] = "8"
#new_horizon["Date From"] = date_start
#new_horizon["Chrono Date From"] = date_start
#new_horizon["Chrono Step Count"] = "8"
#coad['Model']['DAY_AHEAD'].set_children(new_horizon)
#
#coad.save("one_week_model.xml")
```

6. Run PLEXOS

  ```bash
mono $PLEXOS/PLEXOS64.exe -n "one_week_model.xml" -m DAY_AHEAD
```
