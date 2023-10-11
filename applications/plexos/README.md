# PLEXOS Examples

Introductory example scripts for new users to run PLEXOS on the HPC:

1. [submit_simple.sh](submit_simple.sh) : Run a simple PLEXOS model

2. [submit_enhanced.sh](submit_enhanced.sh) : Runs a simple PLEXOS model but attempts retries if the PLEXOS license cannot be found

3. [submit_multiple.sh](submit_multiple.sh) : Runs multiple PLEXOS models by submitting multiple batch files

4. [submit_job_array.sh](submit_job_array.sh) : Run multiple PLEXOS models using Job array instead

5. [submit_plexos.sh](submit_plexos.sh) : Actual batch file called by `submit_multiple.sh` to submit multiple jobs

6. [models.txt](models.txt) : Contains a list of models that exist within the `5_bus_system_v2.xml` dataset. This is not an exhaustive list

7. [5_bus_system_v2.xml](5_bus_system_v2.xml) : PLEXOS dataset that we will use for these examples

8. Solar, wind, and load timeseries CSV files.
