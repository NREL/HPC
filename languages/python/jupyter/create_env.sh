#!/bin/bash

module load conda
conda create -y -c conda-forge -n jupyterlab python=3.8 pandas scipy numpy matplotlib seaborn jupyterlab  