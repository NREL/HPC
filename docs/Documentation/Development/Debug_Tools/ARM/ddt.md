# Linaro (ARM) DDT

**Documentation:** [Linaro (ARM) DDT]( https://developer.arm.com/documentation/101136/22-1-3/DDT?lang=en)

*Linaro DDT (formerly ARM DDT) is a powerful GUI-based parallel debugger. It is part of the Linaro Forge suite of parallel tools, alongside Linaro MAP and Linaro Performance Reports.*

The focus of this page is on setting up and running DDT, rather than the useful features of DDT. For an overview of parallel debuggers, with a focus on DDT and its capabilities, see our [parallel debugging overview](/Documentation/Development/Debug_tools) page. For links to in-depth tutorials and guides on DDT, see our [resources](#resources) section. For help setting up ARM DDT, contact [HPC help](mailto:hpc-help@nrel.gov)

## Compiling for debugging

In order to effectively use any debugger on compiled code, including DDT, we must compile with the `-g` flag and, preferably, with the `-O0` optimization flag.

The `-g` flag produces debugging information.

The `-O0` optimization flag ensures that no variables or functions get optimized out, which simplifies debugging.

An example compile including the proper debug flags might look like:
` mpicc -O0 -g application.c -o application.exe`

## Remote GUI Set-Up

DDT involves working with a GUI. Thus, we first need to connect to the cluster in a way that supports fast and efficient visualization of applications.

To do this, we’ll use the FastX application. Follow the instructions on the [FastX page](/Documentation/Viz_Analytics/FastX/fastx) to set up the application and connect to the cluster.

## Launching an application with DDT

Once we’re connected to the cluster via FastX, we want to open a terminal inside FastX and initiate an interactive session:

`salloc --nodes=1 –account=<your account handle> --time=1:00:00`

Then, once the interactive session launches, load the arm module:

`module load arm`

Next, make sure that you have loaded any additional modules that you need to run your application.

Finally, launch DDT by typing `ddt` or `vglrun ddt`. This will launch the DDT GUI. From the GUI, click the `run` button. This will generate a box that allows you to specify the path to the executable and your working directory and choose how many tasks and threads to run your application with, among other settings.

!!! note 

	There are multiple ways to launch DDT. If your job will take a long time to run, it may be better to submit a job that launches DDT in offline mode. To do this, open DDT in GUI mode, set the desired debug points and save the sessionfile (mysession.session). Then, submit a job with the following command:

	`ddt --offline --session=<YOUR SESSION FILE.session> <other flags> -o <NAME of OUTPUT FILE.txt.html>   ./executable`
	
	This will produce a debugging report you can read when the job completes		

Be sure to set your working directory and application directory correctly:
 
![DDT setup](/assets/images/Debugging/ddt_app_path.png)

In this example, the application to debug is the epsilon executable of BerkeleyGW (epsilon.cplx.x). The development version of this code that we wish to debug can be found in /projects/scatter/OH/BerkeleyGW/bin. Be sure to set this `Application` line correctly. We want to run a particular BerkeleyGW calculation out of the /projects/scatter/OH/GaN/eps_ff directory, which contains all of the input files needed for BerkeleyGW to execute properly. Be sure that your `Working Directory` is set correctly.

Next, we want to choose how many MPI tasks to launch the application with, and provide any additional arguments to srun:
 
![DDT srun](/assets/images/Debugging/ddt_srun_options.png)

Finally, you can set the number of OpenMP threads to launch, among other options, by checking the corresponding boxes (OpenMP, CUDA, Memory Debugging, etc.).

Then, click the “run” button. A window will appear that states “listening for your program.” When DDT is done listening, it will show a “paused” view of your source code. From here, you can add break points, etc. When you’re ready to run the program again, click the green triangle towards the top left corner. 

## Summary of steps
1.	Compile your code with the `-g` and `-O0` flags
2.	Connect to the HPC machine via the [FastX](/Documentation/Viz_Analytics/Fastx/fastx) program
3.	Launch an interactive session on the HPC machine with salloc
4.	Launch DDT with the command `vglrun ddt` or `ddt`
5.	Provide DDT the path to the executable and the path to your working directory

## Resources

* NERSC has an excellent in-depth DDT tutorial [here]( https://docs.nersc.gov/tools/debug/ddt/#basic-debugging-functionality)
* See [these slides]( https://www.alcf.anl.gov/sites/default/files/2020-05/Hulguin_Arm_DDT.pdf) for a high-level overview of the Linaro Forge toolkit, including DDT
* See [this tutorial]( https://www.bsc.es/support/DDT-ug.pdf) for a user-friendly walkthrough of DDT

