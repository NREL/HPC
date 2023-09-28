# Using the Parallel Computing Toolbox with MATLAB

*Learn how to use the Parallel Computing Toolbox (PCT) with MATLAB software on the NREL HPC systems.*

!!! note

    Due to an issue with the scheduler and software licenses, we strongly recommend
    the use of compiled MATLAB code for batch processing. Using the PCT with MATLAB
    in batch mode may lead to failed jobs due to unavailability of licenses.

PCT provides the simplest way for users to run parallel MATLAB code on a single,
multi-core compute node. Here, we describe how to configure your local MATLAB
settings to utilize the PCT and provide some basic examples of running parallel
code on NREL HPC systems.

For more extensive examples of PCT usage and code examples, see the [MathWorks
documentation](https://www.mathworks.com/products/parallel-computing.html).

## Configuration in MATLAB R2023a

Configuration of the PCT is done most easily through the interactive
GUI. However, the opening of parallel pools can be significantly slower in
interactive mode than in non-interactive (batch) mode. For this reason, the
interactive GUI will only be used to set up your local configuration. Runtime
examples will include batch scripts that submit jobs directly to the scheduler.

To configure your local parallel settings, start an interactive MATLAB session
with X11 forwarding (see [Running Interactive Jobs on
Kestrel](https://nrel.github.io/HPC/Documentation/Systems/Kestrel/running/) and [Environment
Modules on the Kestrel System](../../Systems/Kestrel/Environments/index.md)). Open MATLAB
R2023a and do the following:

1. Under the Home tab, go to Parallel > Parallel Preferences.
2. In the Parallel Pool box, set the "Preferred number of workers in a parallel
   pool" to at least 104 (the max number of cores currently available on a standard Kestrel
   compute node).
3. Click OK.
4. Exit MATLAB.

For various reasons, you might not have 104 workers available at runtime. In this
case, MATLAB will just use the largest number available.

!!! note

    Specifying the number of tasks for an interactive job (i.e., using `salloc
    --ntasks-per-node=<n>` to start your interactive job) will interfere with parallel
    computing toolbox. We recommend not specifying the number of tasks.

## Examples

Here we demonstrate how to use the PCT on a single compute node on NREL HPC systems. Learn
how to open a local parallel pool with some examples of how to use it for
parallel computations. Because the opening of parallel pools can be extremely
slow in interactive sessions, the examples here will be restricted to
non-interactive (batch) job submission.

!!! note

    Each example below will check out one "MATLAB" and one
    "Distrib_Computing_Toolbox" license at runtime.

### Hello World Example

In this example, a parallel pool is opened and each worker identifies itself via
[`spmd`](http://www.mathworks.com/help/distcomp/spmd.html) ("single program
multiple data"). Create the MATLAB script helloWorld.m:

??? example "MATLAB Hello World script"

    ```matlab
    % open the local cluster profile
    p = parcluster('Processes');

    % open the parallel pool, recording the time it takes
    tic;
    parpool(p); % open the pool
    fprintf('Opening the parallel pool took %g seconds.\n', toc)

    % "single program multiple data"
    spmd
      fprintf('Worker %d says Hello World!\n', labindex)
    end

    delete(gcp); % close the parallel pool
    exit
    ```

To run the script on a compute node, create the file helloWorld.sb:

??? example "Slurm batch script for Hello World"

    ```bash
    #!/bin/bash
    #SBATCH --time=05:00
    #SBATCH --nodes=1
    #SBATCH --job-name=helloWorld
    #SBATCH --account=<account_string>

    # load modules
    module purge
    module load matlab/R2023a

    # define an environment variable for the MATLAB script and output
    BASE_MFILE_NAME=helloWorld
    MATLAB_OUTPUT=${BASE_MFILE_NAME}.out

    # execute code
    cd $SLURM_SUBMIT_DIR
    matlab -nodisplay -r $BASE_MFILE_NAME > $MATLAB_OUTPUT
    ```

where, again, the fields in `< >` must be properly specified.  Finally, at the
terminal prompt, submit the job to the scheduler:

```
$ sbatch helloWorld.sb
```

The output file helloWorld.out should contain messages about the parallel pool
and a "Hello World" message from each of the available workers.

### Example of Speed-Up Using Parfor

MATLAB's [`parfor`](http://www.mathworks.com/help/matlab/ref/parfor.html)
("parallel for-loop") can be used to parallelize tasks that require no
communication between workers. In this example, the aim is to solve a stiff,
one-parameter system of ordinary differential equations (ODE) for different
(randomly sampled) values of the parameter and to compare the compute time when
using serial and parfor loops. This is a quintessential example of Monte Carlo
simulation that is suitable for parfor: the solution for each value of the
parameter is time-consuming to compute but can be computed independently of the
other values.

First, create a MATLAB function stiffODEfun.m that defines the right-hand side
of the ODE system:

??? example "MATLAB code stiffODEfun.m"

    ```matlab
    function dy = stiffODEfun(t,y,c)
      % This is a modified example from MATLAB's documentation at:
      % http://www.mathworks.com/help/matlab/ref/ode15s.html
      % The difference here is that the coefficient c is passed as an argument.
        dy = zeros(2,1);
        dy(1) = y(2);
        dy(2) = c*(1 - y(1)^2)*y(2) - y(1);
    end
    ```

Second, create a driver file stiffODE.m that samples the input parameter and
solves the ODE using the ode15s function.

??? example "MATLAB script stiffODE.m"

    ```matlab
    %{
       This script samples a parameter of a stiff ODE and solves it both in
       serial and parallel (via parfor), comparing both the run times and the
       max absolute values of the computed solutions. The code -- especially the
       serial part -- will take several minutes to run on Eagle.
    %}

    % open the local cluster profile
    p = parcluster('Processes');

    % open the parallel pool, recording the time it takes
    time_pool = tic;
    parpool(p);
    time_pool = toc(time_pool);
    fprintf('Opening the parallel pool took %g seconds.\n', time_pool)

    % create vector of random coefficients on the interval [975,1050]
    nsamples = 10000; % number of samples
    coef = 975 + 50*rand(nsamples,1); % randomly generated coefficients

    % compute solutions within serial loop
    time_ser = tic;
    y_ser = cell(nsamples,1); % cell to save the serial solutions
    for i = 1:nsamples
      if mod(i,10)==0
        fprintf('Serial for loop, i = %d\n', i);
      end
      [~,y_ser{i}] = ode15s(@(t,y) stiffODEfun(t,y,coef(i)) ,[0 10000],[2 0]);
    end
    time_ser = toc(time_ser);

    % compute solutions within parfor
    time_parfor = tic;
    y_par = cell(nsamples,1); % cell to save the parallel solutions
    err = zeros(nsamples,1); % vector of errors between serial and parallel solutions
    parfor i = 1:nsamples
      if mod(i,10)==0
        fprintf('Parfor loop, i = %d\n', i);
      end
      [~,y_par{i}] = ode15s(@(t,y) stiffODEfun(t,y,coef(i)) ,[0 10000],[2 0]);
      err(i) = norm(y_par{i}-y_ser{i}); % error between serial and parallel solutions
    end
    time_parfor = toc(time_parfor);
    time_par = time_parfor + time_pool;

    % print results
    fprintf('RESULTS\n\n')
    fprintf('Serial time : %g\n', time_ser)
    fprintf('Parfor time : %g\n', time_par)
    fprintf('Speedup : %g\n\n', time_ser/time_par)
    fprintf('Max error between serial and parallel solutions = %e\n', max(abs(err)))

    % close the parallel pool
    delete(gcp)
    exit
    ```

Finally, create the batch script stiffODE.sb:

??? example "Slurm batch script stiffODE.sb"

    ```bash
    #!/bin/bash
    #SBATCH --time=20:00
    #SBATCH --nodes=1
    #SBATCH --job-name=stiffODE
    #SBATCH --account=<account_string>

    # load modules
    module purge
    module load matlab/R2023a

    # define environment variables for MATLAB script and output
    BASE_MFILE_NAME=stiffODE
    MATLAB_OUTPUT=${BASE_MFILE_NAME}.out

    # execute code
    cd $SLURM_SUBMIT_DIR
    matlab -nodisplay -r $BASE_MFILE_NAME > MATLAB_OUTPUT
    ```

Next, submit the job (which will take several minutes to complete):

```
$ sbatch stiffODE.sb
```

If the code executed correctly, the end of the text file stiffODE.out should
contain the times needed to compute the solutions in serial and parallel as well
as the error between the serial and parallel solutions (which should be
0!). There should be a significant speed-up — how much depends on the runtime
environment — for the parallelized computation.
