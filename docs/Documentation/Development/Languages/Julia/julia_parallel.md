---
title: Julia Parallel Computing
postdate: November 9, 2022
layout: default
author: Jonathan Maack
description: Guide to parallelism in Julia
parent: Julia
grand_parent: Languages
---

# Parallel Computing in Julia

We will make use of the following basic Monte Carlo integration function throughout this presentation


```julia
using Statistics
using BenchmarkTools # for the `@btime` macro

function mc_integrate(f::Function, a::Real=0, b::Real=1, n::Int=100000)
    ihat = 0.0
    for k in 1:n
        x = (b - a)*rand() + a
        ihat += (f(x) - ihat) / k
    end
    return ihat
end

function intense_computation(t::Real)
    sleep(t)
    return rand()
end;
```

## Asynchronous Tasks

### What are Tasks?

**Tasks** are execution streams that do not depend on each other and can be done in any order. They can be executed asynchronously but they are not executed in parallel. That is, only one task is running at a given time but the order of execution is not predetermined.

Tasks are also known as **coroutines**.

### Creating and Running Tasks

Running a task is done in 3 steps:

1. Creation
2. Scheduling
3. Collect Results


Creating a task can be done directly with the `Task` object:


```julia
my_task = Task(()->mc_integrate(sin, -pi, pi))
```




    Task (runnable) @0x000000011ecc0ab0



Note the `Task` constructor takes a function with no arguments.

We can always define an zero argument anonymous function to pass to the `Task` constructor. The `@task` macro exists for this purpose:


```julia
my_task = @task mc_integrate(sin, -pi, pi)
```




    Task (runnable) @0x0000000136384cd0




Next we schedule the task to run using the `schedule` function


```julia
schedule(my_task)
```




    Task (done) @0x0000000136384cd0



Many times we want to create and schedule a task immediately. We can do this with the `@async` macro:


```julia
my_task = @async mc_integrate(sin, -pi, pi)
```




    Task (done) @0x000000011d14edc0




We can collect the results of the task once it has completed with the `fetch` function


```julia
fetch(my_task)
```




    0.0020294747408654656



There are a few helpful details to know about `fetch`:

1. If the task has not finished when `fetch` is called, the call to `fetch` will block until the task has completed.
2. If the task raises an exception, `fetch` will raise a `TaskFailedException` which wraps the original exception.


Remember that tasks are not inherently parallel, just asynchronous execution streams. 


```julia
function run_mci()
    N = 10
    result = zeros(N)
    for k in 1:N
        result[k] = mc_integrate(sin, -pi, pi)
    end
    return mean(result)
end

function run_mci_task()
    N = 10
    task_res = zeros(N)
    @sync for k in 1:N
        @async(task_res[k] = mc_integrate(sin, -pi, pi))
    end
    return mean(task_res)
end;
```


```julia
@btime run_mci()
@btime run_mci_task();
```

      22.094 ms (1 allocation: 160 bytes)
      24.318 ms (75 allocations: 4.78 KiB)


!!! note
    The `@sync` macro will block at the end of the code block until all enclosed `@async` statements have completed execution.

### Communicating Between Tasks

Sometimes we need to communicate between tasks. An easy way to accomplish this is to use Julia's `Channel` type. We can think of a `Channel` like a pipe or a queue: objects are put in at one end and taken off at the other.

Let's rewrite `run_mci_task` to use channels by dividing the `run_mci` workflow into two functions.


The first function will perform small Monte-Carlo integrations and put the results on a channel with the `put!` function. When it has finished the requested number of computations it will close the channel with `close` and return.


```julia
function integrator(output::Channel{Float64}, N::Int)
    for k in 1:N
        result = mc_integrate(sin, -pi, pi)
        put!(output, result)
    end
    close(output)
    return
end;
```

!!! note 
    If the channel is full, `put!` will block until space opens up.


The second function will take the results off the channel using the `take!` function and accumulate them into an average. We keep pulling results from the channel as long as there is a result or the channel is open. We can check the former with `isready` and the latter with `isopen`.


```julia
function accumulator(input::Channel{Float64})
    mean_val = 0.0
    k = 0
    while isready(input) || isopen(input)
        value = take!(input)
        k += 1
        mean_val += (value - mean_val) / k
    end
    return mean_val
end;
```

!!! note
    If the channel is empty, the `take!` function will block until there is an item available.


Now we create channel which can hold 10 results, create and schedule both tasks and finally fetch the result.


```julia
function run_mci_chan()
    comm_ch = Channel{Float64}(10)
    atask = @async accumulator(comm_ch)
    @async integrator(comm_ch, 10)
    result = fetch(atask)    
    return result
end;
```


```julia
@btime run_mci_chan();
```

      22.097 ms (25 allocations: 1.45 KiB)


### Why Tasks?

If tasks aren't parallel, why are we talking about them in a parallel computing tutorial?

Remeber that tasks are discrete computation units. They naturally define boundaries between computational tasks. Julia's native parallel capabilities are ways of scheduling tasks on other processors.

## Multi-Threading

### Starting Julia with Multiple Threads

Julia (v1.3 or greater) has multithreading built into the language. By default, Julia starts with a single thread.  To start Julia with multiple threads either
* set the environment variable `JULIA_NUM_THREADS` to some value > 1
* start Julia with `--threads` or `-t` option (Julia v1.5 or greater)

Once started, we can see how many threads are running with the function `Threads.nthreads`


```julia
Threads.nthreads()
```




    2



### `@threads` Macro

Many computations take the form of looping over an array where the result of the computation is put into an element in the array and these computations do not interact. In this case, we can make use of the `Threads.@threads` macro.

Lets apply this to our Monte-Carlo integration.


```julia
function run_mci_mt()
    N = 10
    mt_res = zeros(N)
    Threads.@threads for k in 1:N
        mt_res[k] = mc_integrate(sin, -pi, pi)
    end
    return mean(mt_res)
end;
```


```julia
@btime run_mci_mt();
```

      11.118 ms (12 allocations: 1.00 KiB)


### `@spawn` Macro

Some applications require dispatching individual tasks on different threads. We can do this using the `Threads.@spawn` macro. This is like the `@async` macro but will schedule the task on an available thread. That is, it creates a `Task` and schedules it but on an available thread.


```julia
function run_mci_mt2()
    N = 10
    mt_res = Vector{Float64}(undef, N)
    @sync for k in 1:N
        @async(mt_res[k] = fetch(Threads.@spawn mc_integrate(sin, -pi, pi)))
    end
    return mean(mt_res)
end;
```


```julia
@btime run_mci_mt2();
```

      11.385 ms (126 allocations: 8.80 KiB)


There are a couple of oddities about Julia's multi-threading capability to remember:

1. An available thread is any thread that has completed all assigned tasks or any remaining tasks are *blocked*.
2. As of Julia 1.6, once a task has been assigned to a thread, it remains on that thread even after blocking operations. This will likely change in future releases of Julia.

The combination of these two behaviors can lead to load imbalances amongst threads when there are blocking operations within a thread's tasks.

### Using Channels

Just as before, we can use a `Channel` to communicate between tasks in a multi-threaded environment. The only difference is that we replace `@async` with `Threads.@spawn`.


```julia
function run_mci_mt3()
    comm_ch = Channel{Float64}(10)
    itask = Threads.@spawn integrator(comm_ch, 10)
    atask = Threads.@spawn accumulator(comm_ch)
    result = fetch(atask)
    return result
end;
```


```julia
@btime run_mci_mt3();
```

      22.183 ms (35 allocations: 1.61 KiB)


!!! note
    We can see from the timing results this is not the best way to distribute the work since the `integrator` function has much more computational work than the `accumulator` function.

## Distributed Computing with Distributed.jl

### Architecture

Communication patterns are one-sided, so users only manage one process. Communication itself takes the form of function or macro calls rather than explicit send and receive calls.

Distributed.jl is built on two basic types: remote calls and remote references. A remote call is a directive to execute a particular function on a particular process. A remote reference is a reference to a variable stored on a particular process.

There is a strong resemblance to the way Julia handles tasks: Function calls (wrapped in appropriate types) are scheduled on worker processes through remote calls which return remote references. The results of these calls are then retrieved by fetching the values using the remote references.

### Setting Up

We can launch more Julia processes on the same or other machines with the `addprocs` function. Here we launch 2 worker processes on the local machine:


```julia
using Distributed
addprocs(2);
```

Each Julia process is identified by a (64-bit) integer. We can get a list of all active processes with `procs`:


```julia
@show procs();
```

    procs() = [1, 2, 3]


There is a distinction between the original Julia process and those we launched. The original Julia process is often called the **master** process and always has id equal to 1. The launched processes are called **workers**. We can obtain a list of workers with the `workers` function:


```julia
@show workers();
```

    workers() = [2, 3]


By default, distributed processing operations use the workers only.

We can also start up worker processes from the command lines using the `-p` or `--procs` option.

In order to launch Julia processes on other machines, we give `addprocs` a vector of tuples where each tuple is the hostname as a string paired with the number of processes to start on that host.

The Julia global state is not copied in the new processes. We need to manually load any modules and define any functions we need. This is done with the `Distributed.@everywhere` macro:


```julia
@everywhere using Statistics
@everywhere function mc_integrate(f::Function, a::Real=0, b::Real=1, n::Int=100000)
    ihat = 0.0
    for k in 1:n
        x = (b - a)*rand() + a
        ihat += (f(x) - ihat) / k
    end
    return ihat
end;
```

### `@distributed` Macro

The `@distributed` macro is the distributed memory equivalent of the `Threads.@threads` macro. This macro partitions the range of the for loop and executes the computation on all worker processes. 


```julia
function run_mci_dist()
    N = 10
    total = @distributed (+) for k in 1:N
        mc_integrate(sin, -pi, pi)
    end
    return total/N
end;
```


```julia
@btime run_mci_dist();
```

      11.224 ms (157 allocations: 7.16 KiB)


Between the macro and the for loop is an optional reduction. Here we have used `+` but this can be any valid reduction operator including a user defined function. The values given to the reduction are the values of the last expression in the loop.

!!! note
    If we do not provide a reduction, `@distributed` creates a task for each element of the loop and schedules them on worker processes and returns without waiting for the tasks to complete. To wait for completion of the tasks, the whole block can be wrapped with `@sync` macro.

### `@spawnat` Macro

Julia also provides more fine grained control for launching tasks on workers with the `@spawnat` Macro:


```julia
function run_mci_dist2()
    N = 10
    futures = Vector{Future}(undef, N)
    for k in 1:N
        futures[k] = @spawnat(:any, mc_integrate(sin, -pi, pi))
    end
    return mean(fetch.(futures))
end;
```

The first argument to `@spawnat` is the worker to run the computation on. Here we have used `:any` indicating that Julia should pick a process for us. If we wanted to execute the computation on a particular worker, we could specify which one with the worker id value. The second argument is the expression to compute.

`@spawnat` returns a `Future` which is a remote reference. We call `fetch` on it to retrieve the value of the computation. Note that `fetch` will block until the computation is complete.


```julia
@btime run_mci_dist2();
```

      13.020 ms (1119 allocations: 44.34 KiB)



!!! warning
    The entire expression is sent to the worker process before anything in the expression is executed. This can cause performance issues if we need a small part of a big object or array.


```julia
@everywhere struct MyData
    Data::Vector{Float64}
    N::Int
end
function slow(my_data::MyData)
    return fetch(@spawnat(2, mean(rand(my_data.N))))
end;
```


```julia
large_data = MyData(rand(1000000), 5)
@btime slow(large_data);
```

      1.731 ms (108 allocations: 4.08 KiB)


This is easily fixed using a local variable:


```julia
function fast(my_data::MyData)
    n = my_data.N
    return fetch(@spawnat(2, mean(rand(n))))
end;
```


```julia
@btime fast(large_data);
```

      192.843 μs (100 allocations: 3.80 KiB)


### Remote Channels

As suggested by the name, these are the remote versions of the `Channel` type we've already seen. If you look at the source code, they are actually wrap an `AbstractChannel` to provide the needed remote functionality. We can effectively treat them just like a `Channel`.

Let's redo our `integrator` - `accumulator` workflow, but this time let's do a better job of distributing the work:


```julia
@everywhere function integrator(output::RemoteChannel{Channel{Float64}}, N::Int)
    for k in 1:N
        result = mc_integrate(sin, -pi, pi)
        put!(output, result)
    end
    put!(output, NaN)
    return
end;
@everywhere function accumulator(input::RemoteChannel{Channel{Float64}}, nworkers::Int)
    mean_val = 0.0
    k = 0
    finished = 0
    while finished < nworkers
        value = take!(input)
        if value === NaN
            finished += 1
        else
            k += 1
            mean_val += (value - mean_val) / k
        end
    end
    return mean_val
end;
```


```julia
function run_mci_rc()
    comm_ch = RemoteChannel(()->Channel{Float64}(10), 1)
    @spawnat(2, integrator(comm_ch, 5))
    @spawnat(3, integrator(comm_ch, 5))
    atask = @async accumulator(comm_ch, nworkers())
    return fetch(atask)
end;
```

Here we create a `RemoteChannel` on the master process, divide the computationally intensive `integrator` function into two calls and remotely execute them on the worker processes. Then we start a task on the master process to accumulate the values and call fetch to wait for and retrieve the result.


```julia
@btime run_mci_rc();
```

      12.328 ms (1066 allocations: 41.97 KiB)


### Shutting Down

To shutdown the worker processes we can use `rmprocs`.


```julia
rmprocs(workers())
```




    Task (done) @0x000000011cd3cde0



Alternatively, we can also just exit Julia and the workers will be shutdown as part of the exit process.

## Distributed Computing with MPI.jl

<!-- **TODO:** Stuff:
1. Outline
2. describe MPI.jl api and capabilities
2. breakup script
3. describe what we're doing -->

### Overview of MPI.jl

[`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) is a Julia wrapper around an MPI library. By default it will download an MPI library suitable for running on the installing system. However, it is easily configured to use an existing system MPI implementation (e.g. one of the MPI modules on Eagle). See the documentation for [instructions on how to do this](https://juliaparallel.github.io/MPI.jl/stable/configuration/).

`MPI.jl` mostly requires transmitted things to be buffers of basic types (types that are easily converted to C). Some functions can transmit arbitrary data by serializing them, but this functionality is not as fleshed out as in mpi4py.

### Example

We first need to load and initialize MPI.

```julia
using MPI
MPI.Init()
```

`MPI.Init` loads the MPI library and calls `MPI_Init` as well as sets up types for that specific MPI library.


Now we can implement our Monte-Carlo integration workflow using MPI

```julia
function run_mci_mpi()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if rank == 0
        N = 10
        num = [N]
    else
        num = Vector{Int}(undef, 1)
    end
    MPI.Bcast!(num, 0, comm)

    rank_sum = 0.0
    for k in rank+1:size:num[1]
        rank_sum += mc_integrate(sin, -pi, pi)
    end

    total = MPI.Reduce([rank_sum], MPI.SUM, 0, comm)
    if rank == 0
        result = total / N
    else
        result = nothing
    end

    return result
end
```


To benchmark this we time it many (10000) times and track the minimal value (this is similar to what the `@btime` macro does).

```julia
function run_loop(nruns::Int)
    
    min_time = 1e10
    result = 0.0
    
    for _ in 1:nruns
        MPI.Barrier(MPI.COMM_WORLD)
        start = time()
        result = run_mci_mpi()
        stop = time()
        elapsed = stop - start
        if elapsed < min_time
            min_time = elapsed
        end
    end

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println("Elapsed time: ", min_time)
    end

    return
end

run_loop(10000)
```


Here are the results:

```shell
mpirun -n 2 julia mpi_mci.jl
  Activating environment at `~/HPC_Apps/julia-tutorial/Project.toml`
  Activating environment at `~/HPC_Apps/julia-tutorial/Project.toml`
Elapsed time: 0.01108694076538086
```

## GPU Computing

We provide a brief survey of available packages that can be used to get started.

Packages exist for [NVIDIA's CUDA](https://github.com/JuliaGPU/CUDA.jl), [AMD's ROCm](https://github.com/JuliaGPU/AMDGPU.jl), and [Intel's oneAPI](https://github.com/JuliaGPU/oneAPI.jl). CUDA.jl is the most mature while the other two, as of this writing, are still underdevelopment.

The package [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) is an abstraction layer for enabling different GPU backends.

See the JuliaGPU organization's [webpage](https://juliagpu.org/) or [github repo](https://github.com/JuliaGPU) for a great place to get started.

## Additional Resources

The following are great resource for learning more

*  [Julia Documentation](https://docs.julialang.org/en/v1/) -- the manual discusses the inner workings of Julia including the native parallel computing capabilities
*  [Julia community](https://julialang.org/community/) especially the following discourse channels
    - [Julia discourse](https://discourse.julialang.org/) -- all channels
    - [Julia at Scale discourse](https://discourse.julialang.org/c/domain/parallel/34) -- for scalable Julia
    - [Julia GPU discourse](https://discourse.julialang.org/c/domain/gpu/11) -- for GPU Julia computing
*  [Julia Youtube Channel](https://www.youtube.com/c/TheJuliaLanguage) -- tutorials for Julia and Julia packages
*  MPI.jl [package repo](https://github.com/JuliaParallel/MPI.jl) and [documentation](https://juliaparallel.github.io/MPI.jl/stable/)
*  JuliaGPU [webpage](https://juliagpu.org/) and [github repo](https://github.com/JuliaGPU)
