# Job Priorities on Kestrel
*Job priority on Kestrel is determined by a number of factors including time the job is eligible to run in the queue (age),
the size of the job (jobsize), resources requested and their partition (partition), quality of service and the 
associated priority (qos), and the relative fair-share of the individual allocation.*

Learn about [job partitions and scheduling policies](./index.md).

## Job Priority & Scheduling

The Slurm scheduler has two scheduling loops: (1) the main scheduling loop, which schedules jobs in strict priority order, and (2) the backfill scheduling loop, that allows lower priority jobs to be scheduled (as long as the expected start time of higher priority jobs is not affected).  In both cases, Slurm schedules in strict priority, with higher priority jobs being considered first for scheduling; however, due to the resources requested or other configuration options, there may be
availability for backfill to schedule lower priority jobs (with the same caveat as before, that lower priority jobs can not
affect the expected start time of higher priority jobs).

An individual job's priority is a combination of multiple factors: (1) age, (2) nodes requested or jobsize, (3) partition
factor, (4) quality of service (qos), and (5) the relative fair-share of the individual allocation.  There is a weighting
factor associated with each of these components (shown below) that determines the relative contribution of each factor to
a job's priority. 

| Component | Weighting Factor | Nominal Weight| Note |
| :---| :---: | :---: | :--- | 
| AGE | 30,589,200 |4% | Jobs accumulate AGE priority while in the queue and eligible to run (up to a maximum of 14 days) |
| JOBSIZE | 221,771,700 | 29%| TO BE CHANGED
| PARTITION | 38,236,500 | 5% | Not currently implemented in Kestrel; all jobs receive max partition priority.|
| QOS | 76,473,000 | 10%| A job may request high-priority using --qos=high.  Jobs with this flag selected receive maximum
| FAIR-SHARE| 397,659,600 | 55% |  A project is under-served (and receives a higher fair-share priority) if the projects' usage is low relative to the size of its' allocation.  There is additional complexity discussed below.|

## Fairshare

A project's fairshare is a function of: (1) the project allocation, (2) the sum of the siblings allocations, and (3) recent usage of both the project and the siblings.  The top level of the fairshare tree is the allocation pool (EERE/NREL) with 85% of the machine assigned to EERE and 15% to NREL.  The level fairshare for both EERE and NREL would be calculated using the following equation, where EERE and NREL are siblings:

$$Level Fairshare = \frac{S}{U}$$

where 

$$S = \frac{Sraw_{self}}{Sraw_{self+siblings}}, \quad U = \frac{Uraw_{self}}{Uraw_{self+siblings}}$$

This is repeated at each level of the fairshare tree, where the siblings at each level are used in the calculation (e.g., at the EERE Office level, the siblings for EERE_WETO are the other EERE offices; within EERE_WETO, the siblings are the other projects contained in that office).  Once the level fairshare calculations are complete,  a ranked list is built using a depth first traversal of the fairshare tree and a projects fairshare priority is proportional its' position on this list.


## How to Get High Priority for a Job
You can submit your job to run at high priority or you can request a node reservation.

### Running a Job at High Priority 
**Jobs that are run at high priority will be charged against the project's allocation at twice the normal rate.** If your job would have taken 60 hours to complete at normal priority, it will be charged 120 hours against your allocation when run with high priority.

If you've got a deadline coming up and you want to **reduce the queue wait time** for your jobs, you can run your jobs at high priority by submitting them with the ```--qos=high``` option. This will provide a small priority boost.

### Requesting a Node Reservation
If you are doing work that requires real-time Kestrel access in conjunction with other ESIF user facility laboratory resources, you may request that nodes be reserved for specific time periods.

Your project allocation will be charged for the entire time you have the nodes reserved, whether you use them or not.

To request a reservation, contact [HPC Help](mailto://hpc-help@nrel.gov).

