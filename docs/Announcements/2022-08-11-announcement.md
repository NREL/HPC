---
title: August 2022 Monthly Update
data: 2022-08-01
layout: default
brief: Annual Reports, NSRDB Updates
---

# Annual Reports due August 31
FY22 HPC annual reports are due by August 31, 2022. If you have received an email prompting you to submit an annual report for any of your FY22 projects please follow the instructions provided. If you have any questions please contact [hpc-reporting@nrel.gov](mailto:hpc-reporting@nrel.gov).

# Notification of Full NSRDB Data Update
Over the past few years, the NSRDB team has been hard at work improving their solar data product. There have been numerous improvements in software, data quality, cloud property algorithms, solar position algorithms, and surface albedo data. The current data and software version is v3.2.2, and you can see the [NSRDB version history](https://nsrdb.nrel.gov/about/version-history) for more details. For the new and improved NSRDB data to proliferate, the NSRDB team will be manipulating the /datasets/NSRDB/ directory on Eagle and will eventually deprecate the data in the /datasets/NSRDB/v3/ directory. Here is an outline of the changes that will be made along with an estimated timeline:

* Before 8/13, the new NSRDB data will be copied to the /datasets/NSRDB/current/ directory (including data for 2021!)
* On 8/20, the old NSRDB data in /datasets/NSRDB/v3/ will be moved to /datasets/NSRDB/deprecated_v3/
* On 9/3, the data in /datasets/NSRDB/conus/ and /datasets/NSRDB/full_disc/ will be replaced with the new data (this high-res data is too big to keep two copies).
* On 10/1, the old NSRDB data in /datasets/NSRDB/deprecated_v3/ will be removed permanently.

You should be aware of one significant change if you have hard-coded your site location index values into any code: the NSRDB team is updating the meta data in the standard 4km 30min NSRDB product. The old meta data had several errors and inconsistencies which will be fixed in the new meta. A mapping of site index values from the “v3” meta to the new meta can be found [here](https://app.box.com/s/gqehjrmo6s17h2i9cqs1gh7ap9tidal4) and is also copied at /datasets/NSRDB/nsrdb_v3_to_current_map.csv on Eagle.

Minor differences in the data should be expected, but please reach out to [Grant Buster](mailto:Grant.Buster@nrel.gov) and [Manajit Sengupta](mailto:Manajit.Sengupta@nrel.gov) with anything you see that looks like a true error.

Thanks for your cooperation and thanks for using the NSRDB!
