# C++

*"C++ is a general-purpose programming language providing a direct and efficient model of hardware combined with facilities for defining lightweight abstractions."*
  - Bjarne Stroustrup, "The C++ Programming Language, Fourth Edition"

## Getting Started

This section illustrates the process to compile and run a basic C++ program on the HPC systems.

### Hello World

Begin by creating a source file named `hello.cpp` with the following contents:

```c++
#include <iostream>

int main(void) {
  std::cout << "Hello, World!\n";
  return 0;
}
```

Next, we must select the compiler to use for compiling our program.  We can choose among GNU, Intel, and Cray compilers, depending on the system that we are using (see [Compilers and Toolchains](#compilers-and-toolchains)).  To see available modules and versions, use `module avail`.  For this example, we will use the `g++` compiler, which is part of GNU's `gcc` package.  We will load the default version of the compiler, which in this case is gcc 10.1:

```
$ module load gcc
$ module list
Currently Loaded Modules:
  1) gcc/10.1.0
$ gcc --version | head -1
gcc (Spack GCC) 10.1.0
```

With the `gcc` package, the C++ compiler is provided by the `g++` command.  To compile the program, run:

```
$ g++ hello.cpp -o hello
```

This creates an executable named `hello`.  Now run the program and observe the output:

```
$ ./hello
Hello, World!
```

### CUDA Vector Operations Example

 This code demonstrates basic vector operations using CUDA:

 1. Scaling a vector using two different kernel strategies
 2. Adding two vectors element-wise on the GPU
 3. Managing memory and data transfer between host (CPU) and device (GPU)
 4. Verifying results on the host
 
Each CUDA kernel shows a different way to parallelize vector operations.
The main function walks through allocation, kernel launches, and cleanup.
Create a source file named `matrixMultiplication.cu` with the following contents:

??? example "matrixMultiplication.cu"
    ```
    #include <iostream>
    #include <cassert>

    // Define the size of the vectors
    #define N 10240
    typedef float dType;

    // CUDA kernel to scale a vector (each thread handles one element)
    __global__ void scaleVector(dType scale, dType *input, dType *output)
    {
        // Calculate global thread index
        int tid = threadIdx.x + blockIdx.x*blockDim.x;
        
        // Ensure thread index is within bounds
        if (tid < N)
        {
            output[tid] = input[tid]*scale;
        }
    };

    // Alternate CUDA kernel to scale a vector (each thread handles multiple elements)
    __global__ void scaleVectorAlternate(dType scale, dType *input, dType *output)
    {
        // Calculate global thread index
        int tid = threadIdx.x + blockIdx.x*blockDim.x;
        
        // Loop over elements assigned to this thread
        for (int i=tid; i<N; i=i+blockDim.x*gridDim.x)
        {
            output[i] = input[i]*scale;
        }
    }

    // CUDA kernel to add two vectors element-wise
    __global__ void addVectors(dType *a,dType *b, dType *c)
    {
        int tid = threadIdx.x + blockIdx.x*blockDim.x; // Calculate global thread index

        // Each thread processes multiple elements spaced by total number of threads
        for (int i=tid; i<N; i=i+blockDim.x*gridDim.x)
        {
            c[i] = a[i] + b[i]; // Add corresponding elements from a and b, store in c
        }

    }

    int main()
    {
        // Allocate host memory for input and output vectors
        dType *a = new dType[N];
        dType *b = new dType[N];

        // Device pointers
        dType *dev_a;
        dType *dev_b;

        // Scaling factor
        const dType scale = 2;

        // Initialize input vector on host
        for (int i=0; i<N; i++) a[i] = (dType)i / 2.0;

        // Allocate device memory
        cudaMalloc( (void**)&dev_a, N*sizeof(dType));
        cudaMalloc( (void**)&dev_b, N*sizeof(dType));

        // Copy input data from host to device
        cudaMemcpy(dev_a, a, N*sizeof(dType), cudaMemcpyHostToDevice);

        // Set CUDA kernel launch parameters
        const int blockSize=128;
        int nBlocks = (N-1)/blockSize + 1;

        // Launch the scaleVector kernel on the device
        scaleVector <<<nBlocks, blockSize>>> (scale,dev_a,dev_b);

        // Copy result from device to host
        cudaMemcpy(b, dev_b, N*sizeof(dType), cudaMemcpyDeviceToHost);

        // Verify results on host
        for (int i=0; i<N; i++) assert(a[i]*scale == b[i]);

        // Reset device memory for the next kernel
        cudaMemset(dev_b, (dType) 0, N*sizeof(dType));

        // Change the number of blocks such that each thread handles multiple elements
        nBlocks = 4;

        // Launch the alternate scaleVector kernel on the device
        scaleVectorAlternate <<<nBlocks, blockSize>>> (scale,dev_a,dev_b);

        // Copy result from device to host
        cudaMemcpy(b, dev_b, N*sizeof(dType), cudaMemcpyDeviceToHost);

        // Verify results on host
        for (int i=0; i<N; i++) assert(a[i]*scale == b[i]);

        dType *c;      // Host pointer for result vector
        dType *dev_c;  // Device pointer for result vector

        c = (dType *) malloc(N*sizeof(dType)); // Allocate host memory for output vector
        
        cudaMalloc( (void**)&dev_c, N*sizeof(dType)); // Allocate device memory for output vector

        addVectors <<<nBlocks, blockSize>>> (dev_a, dev_b, dev_c); // Launch kernel to add vectors

        // Copy result from device to host
        cudaMemcpy(c, dev_c, N*sizeof(dType), cudaMemcpyDeviceToHost);

        // Verify results on host
        for (int i=0; i<N; i++) assert(c[i] == a[i] + b[i]); // Check if addition is correct

        std::cout << "All operations completed successfully!" << std::endl;

        // Free host and device memory
        delete [] a;      // Free host memory for input vector a
        delete [] b;      // Free host memory for input/output vector b
        cudaFree(dev_a);  // Free device memory for input vector a
        cudaFree(dev_b);  // Free device memory for input/output vector b
        free(c);          // Free host memory for output vector c
        cudaFree(dev_c);  // Free device memory for output vector c

        return 0; // Return success

    }
    ```

With the `nvhpc` package, the CUDA C++ compiler is provided by the `nvcc` command.  To compile the program, run:

```
$ salloc -A <project_name> -t 00:30:00 --nodes=1 --ntasks-per-node=10 --gres=gpu:1 --partition=debug
$ module load nvhpc
$ nvcc -o cuda_matrixMultiplication matrixMultiplication.cu
```

This creates an executable named `cuda_matrixMultiplication`.  Now run the program and observe the output:

```
$ ./cuda_matrixMultiplication
All operations completed successfully!
```

### MPI + CUDA 2D Diffusion Equation Example

In this example, each MPI process computes a subdomain of a 2D array using CUDA.
Processes are arranged in a 2D Cartesian grid (MPI_Cart_create).
Each process evolves its local subdomain using a simple explicit diffusion scheme.
Boundary data is exchanged using host buffers and MPI_Sendrecv.
At the end, all subdomains are gathered to rank 0 and saved as a CSV file.
 
Create a source file named `mpi_diffusion.cu` with the following contents:

??? example "mpi_diffusion.cu"
    ```
    #include <iostream>
    #include <fstream>
    #include <mpi.h>
    #include <cuda_runtime.h>
    #include <vector>
    #include <cstring>

    #define N 8 // Global domain size (NxN)
    #define BLOCK_SIZE 4 // Local domain size per process (BLOCK_SIZE x BLOCK_SIZE)
    #define NSTEPS 10 // Number of time steps
    #define DT 0.1f
    #define DX 1.0f
    #define DIFF 1.0f // Diffusion coefficient

    typedef float dType;

    // CUDA kernel: explicit 2D diffusion step
    __global__ void diffusionStep(
        dType* u, dType* u_new,
        int n, dType dt, dType dx, dType diff,
        const dType* top, const dType* bottom, const dType* left, const dType* right)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y; // row index in local domain
        int j = blockIdx.x * blockDim.x + threadIdx.x; // column index in local domain

        if (i < n && j < n) {
            // Get neighbor values, using boundary arrays if on edge
            dType up    = (i == 0)      ? top[j]    : u[(i-1)*n + j];           // Top neighbor
            dType down  = (i == n-1)    ? bottom[j] : u[(i+1)*n + j];           // Bottom neighbor
            dType leftv = (j == 0)      ? left[i]   : u[i*n + (j-1)];           // Left neighbor
            dType rightv= (j == n-1)    ? right[i]  : u[i*n + (j+1)];           // Right neighbor
            dType center= u[i*n + j];                                            // Center value

            // Explicit diffusion update formula
            u_new[i*n + j] = center + diff * dt / (dx*dx) *
                (up + down + leftv + rightv - 4.0f * center);
        }
    }

    int main(int argc, char** argv)
    {
        MPI_Init(&argc, &argv); // Initialize MPI

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
        MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

        // Create a 2D Cartesian communicator for domain decomposition
        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims); // Let MPI decide grid shape
        int periods[2] = {0, 0}; // No periodic boundaries
        MPI_Comm cart_comm;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

        int coords[2];
        MPI_Cart_coords(cart_comm, rank, 2, coords); // Get coordinates of this rank in the grid

        // Allocate local domain arrays on host
        dType *local = new dType[BLOCK_SIZE * BLOCK_SIZE];      // Current solution
        dType *local_new = new dType[BLOCK_SIZE * BLOCK_SIZE];  // Next solution

        // Initialize: set center of global domain to 100, rest to 0
        for (int i = 0; i < BLOCK_SIZE; ++i)
            for (int j = 0; j < BLOCK_SIZE; ++j)
                local[i * BLOCK_SIZE + j] = (coords[0] == dims[0]/2 && coords[1] == dims[1]/2 && i == BLOCK_SIZE/2 && j == BLOCK_SIZE/2) ? 100.0f : 0.0f;

        // Allocate device memory for local domains
        dType *d_local, *d_local_new;
        cudaMalloc(&d_local, BLOCK_SIZE * BLOCK_SIZE * sizeof(dType));
        cudaMalloc(&d_local_new, BLOCK_SIZE * BLOCK_SIZE * sizeof(dType));
        cudaMemcpy(d_local, local, BLOCK_SIZE * BLOCK_SIZE * sizeof(dType), cudaMemcpyHostToDevice);

        // Allocate device buffers for boundaries (top, bottom, left, right)
        dType *d_top, *d_bottom, *d_left, *d_right;
        cudaMalloc(&d_top, BLOCK_SIZE * sizeof(dType));
        cudaMalloc(&d_bottom, BLOCK_SIZE * sizeof(dType));
        cudaMalloc(&d_left, BLOCK_SIZE * sizeof(dType));
        cudaMalloc(&d_right, BLOCK_SIZE * sizeof(dType));

        // Host buffers for MPI boundary exchange
        dType h_top[BLOCK_SIZE], h_bottom[BLOCK_SIZE], h_left[BLOCK_SIZE], h_right[BLOCK_SIZE];

        // Find ranks of neighboring processes in the grid
        int north, south, west, east;
        MPI_Cart_shift(cart_comm, 0, 1, &north, &south); // Find north/south neighbors (row direction)
        MPI_Cart_shift(cart_comm, 1, 1, &west, &east);   // Find west/east neighbors (col direction)

        // CUDA kernel launch configuration
        dim3 block(2,2), grid((BLOCK_SIZE+1)/2,(BLOCK_SIZE+1)/2);

        // Main time-stepping loop
        for (int step = 0; step < NSTEPS; ++step) {
            // Extract boundaries from device to host for MPI exchange
            cudaMemcpy(h_top,    d_local, BLOCK_SIZE * sizeof(dType), cudaMemcpyDeviceToHost); // Copy top row to host
            cudaMemcpy(h_bottom, d_local + (BLOCK_SIZE-1)*BLOCK_SIZE, BLOCK_SIZE * sizeof(dType), cudaMemcpyDeviceToHost); // Copy bottom row to host
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                cudaMemcpy(&h_left[i],  d_local + i*BLOCK_SIZE, sizeof(dType), cudaMemcpyDeviceToHost); // Copy left column to host
                cudaMemcpy(&h_right[i], d_local + i*BLOCK_SIZE + (BLOCK_SIZE-1), sizeof(dType), cudaMemcpyDeviceToHost); // Copy right column to host
            }

            // Exchange boundaries with neighbors using MPI_Sendrecv
            // Each boundary is sent to the neighbor and received from the same neighbor
            // The received data overwrites the host buffer for use in the next kernel launch

            // North neighbor exchange (top row)
            if (north != MPI_PROC_NULL)
                MPI_Sendrecv(h_top, BLOCK_SIZE, MPI_FLOAT, north, 0,   // Send top row to north
                            h_top, BLOCK_SIZE, MPI_FLOAT, north, 0,   // Receive top row from north
                            cart_comm, MPI_STATUS_IGNORE);

            // South neighbor exchange (bottom row)
            if (south != MPI_PROC_NULL)
                MPI_Sendrecv(h_bottom, BLOCK_SIZE, MPI_FLOAT, south, 0, // Send bottom row to south
                            h_bottom, BLOCK_SIZE, MPI_FLOAT, south, 0, // Receive bottom row from south
                            cart_comm, MPI_STATUS_IGNORE);

            // West neighbor exchange (left column)
            if (west != MPI_PROC_NULL)
                MPI_Sendrecv(h_left, BLOCK_SIZE, MPI_FLOAT, west, 0,    // Send left column to west
                            h_left, BLOCK_SIZE, MPI_FLOAT, west, 0,    // Receive left column from west
                            cart_comm, MPI_STATUS_IGNORE);

            // East neighbor exchange (right column)
            if (east != MPI_PROC_NULL)
                MPI_Sendrecv(h_right, BLOCK_SIZE, MPI_FLOAT, east, 0,   // Send right column to east
                            h_right, BLOCK_SIZE, MPI_FLOAT, east, 0,   // Receive right column from east
                            cart_comm, MPI_STATUS_IGNORE);

            // Copy received boundaries from host to device for use in kernel
            cudaMemcpy(d_top,    h_top,    BLOCK_SIZE * sizeof(dType), cudaMemcpyHostToDevice);    // Top boundary
            cudaMemcpy(d_bottom, h_bottom, BLOCK_SIZE * sizeof(dType), cudaMemcpyHostToDevice);     // Bottom boundary
            cudaMemcpy(d_left,   h_left,   BLOCK_SIZE * sizeof(dType), cudaMemcpyHostToDevice);     // Left boundary
            cudaMemcpy(d_right,  h_right,  BLOCK_SIZE * sizeof(dType), cudaMemcpyHostToDevice);     // Right boundary

            // Perform one explicit diffusion step on GPU
            diffusionStep<<<grid, block>>>(
                d_local, d_local_new, BLOCK_SIZE, DT, DX, DIFF,
                d_top, d_bottom, d_left, d_right);

            // Swap pointers for next step (so d_local always points to current solution)
            std::swap(d_local, d_local_new);
        }

        // Copy final local domain from device to host for gathering
        cudaMemcpy(local, d_local, BLOCK_SIZE * BLOCK_SIZE * sizeof(dType), cudaMemcpyDeviceToHost);

        // Gather all subdomains to rank 0 for output
        int total_size = BLOCK_SIZE * BLOCK_SIZE * size;
        std::vector<dType> global;
        if (rank == 0)
            global.resize(total_size);

        // Gather all local domains into the global array on rank 0
        MPI_Gather(local, BLOCK_SIZE * BLOCK_SIZE, MPI_FLOAT,
                  rank == 0 ? global.data() : nullptr, BLOCK_SIZE * BLOCK_SIZE, MPI_FLOAT,
                  0, MPI_COMM_WORLD);

        // Save the full solution to CSV on rank 0
        if (rank == 0) {
            std::ofstream fout("diffusion_result.csv");
            fout << "x,y,u\n"; // CSV header
            int px = dims[0], py = dims[1]; // Number of processes in x and y
            for (int proc = 0; proc < size; ++proc) {
                int cx, cy;
                MPI_Cart_coords(cart_comm, proc, 2, coords); // Get process coordinates for proc
                cx = coords[0]; cy = coords[1];
                for (int i = 0; i < BLOCK_SIZE; ++i) {
                    int gi = cx * BLOCK_SIZE + i; // Global row index
                    for (int j = 0; j < BLOCK_SIZE; ++j) {
                        int gj = cy * BLOCK_SIZE + j; // Global col index
                        int idx = proc * BLOCK_SIZE * BLOCK_SIZE + i * BLOCK_SIZE + j; // Index in gathered array
                        fout << gi << "," << gj << "," << global[idx] << "\n"; // Write x,y,u
                    }
                }
            }
            fout.close();
            std::cout << "Saved diffusion_result.csv\n";
        }

        // Free device memory
        cudaFree(d_local);
        cudaFree(d_local_new);
        cudaFree(d_top);
        cudaFree(d_bottom);
        cudaFree(d_left);
        cudaFree(d_right);

        // Free host memory
        delete[] local;
        delete[] local_new;

        MPI_Finalize(); // Finalize MPI
        return 0;
    }
    ```

With the `nvhpc` package, the MPI+CUDA C++ compiler is provided by the `mpicxx` command.  To compile the program, run:

```
$ salloc -A <project_name> -t 00:30:00 --nodes=1 --ntasks-per-node=10 --gres=gpu:1 --partition=debug
$ module load nvhpc
$ mpicxx -o cuda_mpi_diffusion mpi_diffusion.cu -lcudart -lmpi
```

This creates an executable named `cuda_mpi_diffusion`.  Now run the program and observe the output:

```
$ srun -n 4 ./cuda_mpi_diffusion
Saved diffusion_result.csv
```


## Compilers and Toolchains

The following is a summary of available compilers and toolchains.  User are encouraged to run `module avail` to check for the most up-to-date information on a particular system.

| Toolchain | C++ Compiler | Module                   | Systems                   |
|-----------|--------------|--------------------------|---------------------------|
| gcc       | `g++`        | `gcc`                    | All                       |
| Intel     | `icpc`       | `intel-oneapi-compilers` | Swift, Vermilion, Kestrel |
| Cray      | `CC`         | `PrgEnv-cray`            | Kestrel                   |

Note that Kestrel also provides the `PrgEnv-intel` and `PrgEnv-gnu` modules, which combine the Intel or gcc compilers together with Cray MPICH.  Please refer to [Kestrel Programming Environments Overview](../../Systems/Kestrel/Environments/index.md) for details about the programming environments available on Kestrel.

For information specific to compiling MPI applications, refer to [MPI](../Programming_Models/mpi.md).
