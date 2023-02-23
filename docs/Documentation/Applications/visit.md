# VisIT

*VisIT is a free interactive parallel visualization and graphical analysis tool for viewing scientific data on Unix and PC platforms.*

With VisIt, users can quickly generate visualizations from their data, animate them through time, manipulate them, and save the resulting images for presentations. It contains a rich set of visualization features so that you can view your data in a variety of ways. 
Also, it can be used to visualize scalar and vector fields defined on two- and three-dimensional (2D and 3D) structured and unstructured meshes.

VisIt was designed to handle very large data set sizes in the terascale range, and yet can also handle small data sets in the kilobyte range.

For more information on VisIt, see their [Lawrence Livermore National Laboratory website](https://wci.llnl.gov/simulation/computer-codes/visit).

# Using VisIT

VisIt features a robust remote visualization capability.  To enable remote visualization (client/server), follow these steps.

1. On Eagle, add:
    ```
    module use /nopt/nrel/apps/modules/centos74/modulefiles
    module load visit
    ```
    to your `.bashrc` file in the home directory
2. On a local machine, download VisIt 2.13.3 for the appropriate platform from the [Lawrence Livermore National Laboratory VisIt site](https://wci.llnl.gov/simulation/computer-codes/visit/executables).
3. Copy the file `host-eagle.xml` to `$HOME/.visit/hosts` on your local machine.
4. Restart VisIt.
5. The installed profile can be viewed and edited by clicking on 'Options --> Host profiles ... '. A remote host profile should appear.
6. Go to Launch Profiles.
7. Go to the Parallel tab, set up the job parameters, select sbatch/srun for ‘Parallel launch method’ and then click Apply.
8. 