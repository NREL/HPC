# Using the Intel Math Kernel Library

*Learn how to use the Intel Math Kernel Library (MKL).*

MKL includes a wealth of routines to accelerate technical application performance on modern multicore architectures. Core math functions in MKL include BLAS, LAPACK, ScaLAPACK, sparse solvers, fast Fourier transforms, vector math, and data fitting. 

!!! tip "Note"
	
	If you are mixing an Anaconda environment with modules to build, always activate the conda environment *before* loading any library modules like MKL. cmake discovery, for example, is very sensitive to the order in which these actions are taken. 

With the Intel toolchain, linking against MKL is as simple as adding `-mkl` to the link command. This by default links in the threaded MKL routines. To limit to strictly sequential (*i.e.*, not threaded) routines, use `-mkl=sequential`; to enable multi-process Scalapack routines, use `-mkl=cluster`. 
To link MKL with GCC, the `mkl` module includes some convenience environment variables defined as the appropriate `LDFLAGS` setting. See the `module show mkl` output; the variable naming is intended to be self-explanatory. 

If you have needs not covered by these, use Intel's interactive [MKL Link Line Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html) website to discover the appropriate linking options. **Don't** use mkl_link_tool in your build automation, as Intel only provides a 32-bit version of this tool which will cause builds to fail. 

# User Tips
MKL will provide optimized library code based on the most advanced instruction set able to run on discovered hardware. So for floating point math, although GNU and Intel compilers will generate application code with SSE 4.2 instructions by default, MKL libraries will use AVX-512 float point instructions available on Skylake processors.  

As the code executes, rapid transition between different such floating point instruction sets may cause a significant performance penalty. Consider compiling the base code optimized for AVX instructions, i.e., adding “-xcore-AVX512” for Intel and “-march=skylake-avx512” for GNU.

Using `-mkl` by default generates the code to use multithreaded MKL routines. There is an extra initialization overhead associated with using multithreaded MKL. With the smaller problem size or with sparce vectors it may be more beneficial from the performance standpoint to use sequential MKL routines ( `-mkl=sequential`). 

