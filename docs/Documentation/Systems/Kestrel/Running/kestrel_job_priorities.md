# Job Priorities on Kestrel
*Job priority on Kestrel is determined by a number of factors including time the job is eligible to run in the queue (age),
the size of the job (jobsize), resources requested and their partition (partition), quality of service and the 
associated priority (qos), and the relative fair-share of the individual allocation.*

Learn about [job partitions and scheduling policies](./index.md).

## Job Priority & Scheduling

The Slurm scheduler has two scheduling loops: (1) the main scheduling loop, which schedules jobs in strict priority order, and (2) the backfill scheduling loop, that allows lower priority jobs to be scheduled (as long as the expected start time of higher priority jobs is not affected).  In both cases, Slurm schedules in strict priority, with higher priority jobs being considered first for scheduling; however, due to the resources requested or other configuration options, there may be availability for backfill to schedule lower priority jobs (with the same caveat as before, that lower priority jobs can not affect the expected start time of higher priority jobs).

An individual job's priority is a combination of multiple factors: (1) age, (2) nodes requested or jobsize, (3) partition
factor, (4) quality of service (qos), and (5) the relative fair-share of the individual allocation.  There is a weighting
factor associated with each of these components (shown below) that determines the relative contribution of each factor:

| Component | Weighting Factor | Nominal Weight| Note |
| :---| :---: | :---: | :--- | 
| AGE | 30,589,200 |4% | Jobs accumulate AGE priority while in the queue and eligible to run (up to a maximum of 14 days) |
| JOBSIZE | 221,771,700 | 29%| TO BE CHANGED
| PARTITION | 38,236,500 | 5% | Not currently implemented in Kestrel; all jobs receive max partition priority.|
| QOS | 76,473,000 | 10%| A job may request high-priority using --qos=high.  Jobs with this flag selected receive a priority boost.
| FAIR-SHARE| 397,659,600 | 55% |  A project is under-served (and receives a higher fair-share priority) if the projects' usage is low relative to the size of its' allocation.  There is additional complexity discussed below.|

## Fairshare

Fairshare is a scheduling system where a project's allocation represents a fractional percentage of the machine.  The intent of the fairshare priority is to elevate or lower priorities of project allocations such that they roughly mirror the assigned fractional percentage.  A project's fairshare priority would be elevated if the utilization is low relative to the allocation, where utilization is a function of sibling projects (same office) as well as high-level parents.  Similarily, a project's fairshare priority would be lower if the utilization is high relative to the allocation.  

A fairtree with a hypothetical allocation is illustrated below:

<img src="../../../../../assets/images/Slurm/Fairtree.png" width="400">

In this hypothetical scenario, fairshare values would be calculated at each vertice.  The calculations for a vertice's fairshare is a function of: (1) the allocation, (2) the sum of the siblings allocations, and (3) recent usage of both the allocation and the siblings.  For example, the nominal percentage of the machine assigned to Project 2 would be ~28% (33% of the EERE allocation, which is 85%).  The utilization of all projects in Office 2 would be used to calculate the specific level fairshare value: 

$$Level Fairshare = \frac{S}{U}$$

where 

$$S = \frac{Sraw_{self}}{Sraw_{self+siblings}}, \quad U = \frac{Uraw_{self}}{Uraw_{self+siblings}}$$

This is repeated at each level of the fairshare tree, and a ranked list is built using a depth first traversal of the fairshare tree.  A proejcts fairshare priority is proportional to its' position on this list.  The list is descended depth first in part to prioritize the higher level assigned percentages (e.g.,  the EERE and NREL utilization is balanced first, then individual offices within EERE and NREL, and finally projects within offices).  Therefore it is hypothetically possible that an underserved allocation exists, but has a lower fairshare priority due to its siblings and parents uage. 

As additional complexity, the above usage calculations are modified by a half-decay system that emphasizes more recent usage and de-emphasizes historical usage:

$$ U = U_{currentperiod} + ( D * U_{lastperiod}) + (D * D * U_{period-2}) + ...$$

The decay factor, *D*, is a number between 0 and 1 that achieves the half-decay rate specified by the Slurm configruation files (14 day on Kestrel).


## How to Get High Priority for a Job
You can submit your job to run at high priority or you can request a node reservation.

### Running a Job at High Priority 
**Jobs that are run at high priority will be charged against the project's allocation at twice the normal rate.** If your job would have taken 60 hours to complete at normal priority, it will be charged 120 hours against your allocation when run with high priority.

If you've got a deadline coming up and you want to **reduce the queue wait time** for your jobs, you can run your jobs at high priority by submitting them with the ```--qos=high``` option. This will provide a small priority boost.

### Requesting a Node Reservation
If you are doing work that requires real-time Kestrel access in conjunction with other ESIF user facility laboratory resources, you may request that nodes be reserved for specific time periods.

Your project allocation will be charged for the entire time you have the nodes reserved, whether you use them or not.

To request a reservation, contact [HPC Help](mailto://hpc-help@nrel.gov).

