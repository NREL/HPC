---
title: March 2021 NREL HPC Monthly Update
data: 2021-03-03
layout: default
brief: Publishing tracker, FY21 allocation reductions
---

# Elevate Your Work With New Tracking for Advanced Computing in the NREL Publishing Tracker

There is a new question on the User Facilities & Program Areas page when you enter a publication into the Pub Tracker 
– "The High Performance Computing Facility was used to produce results or data used in this publication." Please be 
sure to check Yes on this question for your work that made use of the HPC User Facility or other systems in the ESIF 
HPC Data Center. In addition, there are three new Program Areas to use to tag your publication under the Advanced 
Computing heading: Cloud, HPC and Visualization & Insight Center. Making use of these metadata will enable us to 
elevate your work through communications highlights, feature stories, and reporting to EERE.

More information about the NREL Publishing Tracker can  be found by visiting the 
[Access and Use the NREL Publishing Tracker page](https://thesource.nrel.gov/publishing/) on the Source. 


# Fiscal Year 2021 Quarterly Allocation Reductions

You may have noticed that NREL did not make any reductions to allocations that were under-using their AUs during Q1. 
This is for two reasons. First, we were in the process of putting together a new, more transparent, fairer allocation 
reduction policy. Second, we are aware that many users were inconvenienced by the fact that allocation decisions were 
issued on October 1.
 
We realize allocation reductions for low use are not popular with our users. However, they are physically necessary. AUs 
are an "expiring resource." If an AU is not used in Q1, it cannot be stored and saved for use in Q4. Because of this, 
we have to remove some percentage of the unused AUs every quarter. Otherwise we can hit a situation where we have many 
more AUs available to users than the machine can physically provide as the year progresses. This creates long queue times 
that make Eagle physically unusable.
 
Our new allocation policy is given below.  We had an informal discussion with users across multiple centers before designing 
this policy. Users emphasized the need for a policy to be clear enough so they could see what they might lose at the end of 
the quarter. Users have also increasingly requested allocations that had different usages in different quarters to deal with 
their project needs, and we wanted a policy that treated these fairly.     
 
Shortly after Q1, Q2, and Q3 ends, allocations will be automatically adjusted to account for low utilization against 
planned usage. At the end of each quarter, the total allocation units used for the year to date will be compared to 
the total planned usage for the year to date for each allocation. At the end of Q1, the total used for the year to 
date is compared to the Q1 planned usage; at the end of Q2, the total used for the year to date is compared to the 
Q1+Q2 planned usage, and at the end of Q3, the toal used for the year to date is compared to the Q1+Q2+Q3 planned usage. 
 
Allocation units are then removed based on the table below. Note that allocation reductions are meant to be cumulative 
over the course of a year, and not compounding. If, for instance, a project was reduced by 10,000 AUs for low usage 
in Q1, and the reduction table suggests the project should be reduced by 25,000 AUs at the end of Q2, 15,000 
(25,000-15,000) AUs should be removed at the end of Q2 to make the total removal for the year to date equal to 25,000 AUs.


| Percentage of planned AUs used to date | Percentage of planned to date AUs removed | 
|----------------------------------------|-------------------------------------------|
|More than 70%                           | 0% (No reduction)                         |
|Less than 70% but greater than 55%      | 20% 										 |
|Less than 55% but greater than 40%      | 35%										 | 
|Less than 40% but greater than 20%      | 55% 										 |
|Less than 20%                           | 80%  									 |


To understand how this process would work, we consider the following two 100,000 AU allocations, one with a 
uniform distribution of planned AUs throughout the year, and one with a distribution designed to enable development
in Q1 and production runs in Q2 through Q4. The two allocations are described in the table below.

| Quarter | Allocation "Renewables" | Allocation "Efficiency"|
|---------|-------------------------|------------------------|
| Q1 	  | 25,000 	                | 10,000                 |
| Q2 	  | 25,000 	                | 30,000                 |
| Q3 	  | 25,000 	                | 30,000                 |
| Q4 	  | 25,000 	                | 30,000                 |

If "Renewables" uses 9,000 AUs in Q1, 20,000 AUs in Q2, and 25,000 AUs in Q3, it would be reduced in the following manner: 
 
After Q1, the project will have used 36% (9,000/25,000) if its allocation, leading to 13,750 (55% x 25,000) AUs being removed.    
 
After Q2, the project will have used 58% [(9,000+20,000)/(25,000+25,000)] of its allocation, leading to 10,000 AUs 
(20% x 50,000) being potentially removed. However, because 12,500 AUs were removed in Q1, no removal is performed. Note 
that AUs are not restored in this case. 
 
After Q3, the project will have used 72 % [(9,000+20,000+25,000)/(25,000+25,000+25,000)] of its allocation. No reduction 
would be made. The project would then begin Q4 with 33,500 AUs (100,000-9,000-20,000-25,000-12,500). 
 
If "Efficiency" uses AUs under the same schedule, after Q1, the project will have used 90% (9,000/10,000) of its 
allocation and will not be penalized. 
 
After Q2, the project will have used 72.5% [(9,000+20,000)/(10,000+30,000)] of its allocation and will not be penalized. 
 
After Q3, the project will have used 77% [(9,000+20,000+25,000)/(10,000+30,000+30,000)] of its allocation, and will not be 
penalized. The project would then begin Q4 with 46,000 AUs.  
 
Note that "Renewables" and "Efficiency" have the same total (100,000 AUs) but lose very different amounts of AUs over 
the course of the year. This is because the allocation request "Efficiency" is more closely tuned to the user’s actual 
us of HPC resources. NREL allows users to tune their allocation request through the use of "profiles" in the allocation 
request to avoid this sorts of reductions.

This information can also be found by visiting the 
[Fiscal Year 2021 Quarterly Allocation Reductions](https://www.nrel.gov/hpc/resource-allocation-reduction.html) page on our website.

