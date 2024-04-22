#!/bin/bash

# clear the module environment
module restore
module swap PrgEnv-cray PrgEnv-intel
module unload cray-libsci
