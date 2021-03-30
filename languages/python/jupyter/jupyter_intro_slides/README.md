## Jupyter at NREL Slide Presentation

This directory contains the introductory talk "Jupyter on NREL HPC Resources"
first given on March 30th, 2021.

If you have conda on your computer, you can run the notebook locally.

The conda environment used by the presenter with all of the packages necessary to run jupyter, create and display 
slides, and make nice plots is defined in jupyter-slides.yml. 

To create the environment and get started, git clone the NREL/HPC repository to your
computer, copy these files into a working directory (~/jupyter in the example below), and:

`[user@laptop: ~/jupyter]$ conda create -f jupyter-slides.yml`

`[user@laptop: ~/jupyter]$ source activate jupyterenv`

`[user@laptop: ~/jupyter]$ jupyter-notebook`

Then follow the instructions that Jupyter displays. 


You can also view the notebook on Europa by placing it in your home directory
on Eagle and opening it via https://europa.hpc.nrel.gov/  -- all of the output
is pre-generated and contained in the .ipynb notebook files, but you may not 
be able to re-execute some cells since they use python modules that are not 
present on Europa. 


