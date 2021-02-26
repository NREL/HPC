#!/bin/bash

module load conda
conda create -y -c conda-forge -n jupyterenv python=3.8 pandas scipy numpy matplotlib seaborn jupyterlab  
