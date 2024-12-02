# VisIT

*VisIT is a free interactive parallel visualization and graphical analysis tool for viewing scientific data on Unix and PC platforms.*

With VisIt, users can quickly generate visualizations from their data, animate them through time, manipulate them, and save the resulting images for presentations. It contains a rich set of visualization features so that you can view your data in a variety of ways. 
Also, it can be used to visualize scalar and vector fields defined on two- and three-dimensional (2D and 3D) structured and unstructured meshes.

VisIt was designed to handle very large data set sizes in the terascale range, and yet can also handle small data sets in the kilobyte range.

For more information on VisIt, see their [Lawrence Livermore National Laboratory website](https://wci.llnl.gov/simulation/computer-codes/visit).

## Using VisIT
VisIT can be used through the GUI on Dav nodes 

```
module load visit 
visit 
```

VisIt features a robust remote visualization capability.  To enable remote visualization (client/server), follow these steps.

1. On Kestrel, add:
    ```
    module load visit
    ```
    to your `.bashrc` file in the home directory
2. On a local machine, download VisIt 3.3.3 for the appropriate platform from the [Lawrence Livermore National Laboratory VisIt site](https://wci.llnl.gov/simulation/computer-codes/visit/executables).
3. The installed profile can be viewed and edited by clicking on 'Options → Host profiles ... '. A remote host profile should appear.
![Alt text](/assets/images/VisIT/kestrel-5a.png)
![Alt text](/assets/images/VisIT/kestrel-5b.png)
4. Go to Launch Profiles.
![Alt text](/assets/images/VisIT/kestrel-6.png)
5. Go to the Parallel tab, set up the job parameters, select sbatch/srun for ‘Parallel launch method’ and then click Apply.
![Alt text](/assets/images/VisIT/kestrel-software-visit-step7.png)
6. To connect to VisIt, go to File → Open file
![Alt text](/assets/images/VisIT/kestrel-8.png)
7. In the Host option, click on the drop down menu and choose the host kestrel.hpc.nrel.gov
![Alt text](/assets/images/VisIT/kestrel-9.png)
8. It will display a window with an option to change the username, if the username is not correct, then click on change username. *This is your HPC username*
9. Type your HPC username and click Confirm username.
10. Enter your HPC password and click OK.
11. Wait for visit client to connect to the server on Kestrel.
12. Enter the directory where your data is located into Path.
![Alt text](/assets/images/VisIT/kestrel-14.png)
13. Once you choose your data file, VisIt will display the job information; you can change them and then click OK.
14. Once the job is submitted, you can start applying visualization filters to your data. For the job information:
    - Bank / Account: enter the project name you are charging to.
    - Time limit: enter the time you need for the job in the following format H:M:S.
