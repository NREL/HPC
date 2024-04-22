#!/bin/bash

# restores default environment, which is PrgEnv-cray
# we want to use module restore rather than module purge; module load PrgEnv-cray
# to preserve some environment variables.
module restore
