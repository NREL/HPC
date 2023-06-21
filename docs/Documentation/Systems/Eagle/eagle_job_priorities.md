# Job Priorities on Eagle 
*Job priority on Eagle is determined by a number of factors including queue wait time (AGE), job size, the need for limited resources (PARTITION), request for priority boost (QOS), and Fair-Share.*

Learn about [job partitions and scheduling policies](./eagle_partitions.md).

## How to View Your Job's Priority 
The ```sprio``` command may be used to look at your job's priority. Priority for a job in the queue is calculated as the sum of these components:

| Component | Contribution |
| ----------| ------------ | 
| AGE       | Jobs accumulate priority points per minute the job spends eligible in the queue.|
| JOBSIZE   | Larger jobs have some priority advantage to allow them to accumulate needed nodes faster.|
| PARTITION | Jobs routed to partitions with special features (memory, disk, GPUs) have priority to use nodes equipped with those features.|
| QOS       | Jobs associated with projects that have exceeded their annual allocation are assigned low priority.<br>Jobs associated with projects that have an allocation remaining are assigned normal priority. These jobs start before jobs with a low priority.<br>A job may request high priority using --qos=high. Jobs with high priority start before jobs with low or normal priority. Jobs with qos=high use allocated hours at 2x the normal rate.|
| FAIR-SHARE| Each projects Fair-Share value will be (Project Allocation) / (Total Eagle Allocation).  Those using less than their fair share in the last 2 weeks will have increased priority.  Those using more than their fair share in the last 2 weeks will have decreased priority. | 

The ```squeue --start <JOBID>```  command can be helpful in estimating when a job will run.

The ```scontrol show job <JOBID>``` command can be useful for troubleshooting why a job is not starting.

## How to Get High Priority for a Job
You can submit your job to run at high priority or you can request a node reservation.

### Running a Job at High Priority 
**Jobs that are run at high priority will be charged against the project's allocation at twice the normal rate.** If your job would have taken 60 hours to complete at normal priority, it will be charged 120 hours against your allocation when run with high priority.

If you've got a deadline coming up and you want to **reduce the queue wait time** for your jobs, you can run your jobs at high priority by submitting them with the ```--qos=high``` option. This will provide a large priority boost, which will move the job to the top of the list of jobs waiting for resources.

### Requesting a Node Reservation
If you are doing work that requires real-time Eagle access in conjunction with other ESIF user facility laboratory resources, you may request that nodes be reserved for specific time periods.

Your project allocation will be charged for the entire time you have the nodes reserved, whether you use them or not.

To request a reservation, contact [HPC Help](mailto://hpc-help@nrel.gov). 
