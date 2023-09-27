---
title: Plexos
parent: Applications
---

# PLEXOS

*PLEXOS is a simulation software for modeling electric, gas, and water systems for optimizing energy markets.* 

Users can run PLEXOS models on NREL's computing clusters. However, users need to build the PLEXOS models on a Windows system as there is no GUI available on the clusters and on Linux in general

## Available modules

| Kestrel         | Eagle           | Swift           | Vermilion |
|:---------------:|:---------------:|:---------------:|:---------:|
|                 | plexos/8.300R09 |                 ||                        
|                 | plexos/9.000R07 |                 ||
| plexos/9.000R09 | plexos/9.000R09 | plexos/9.000R09 ||
|                 | plexos/9.200R05 |                 ||
| plexos/9.200R06 ||||


!!! info
    A user can only run PLEXOS with Gurobi solvers at this time. Please set up your model accordingly.

## Contents

1. [Setting up PLEXOS](setup_plexos.md)
2. [Running Plexos](run_plexos.md)