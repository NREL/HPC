/*
  STREAM benchmark implementation in CUDA.

    COPY:       a(i) = b(i)                 
    SCALE:      a(i) = q*b(i)               
    SUM:        a(i) = b(i) + c(i)          
    TRIAD:      a(i) = b(i) + q*c(i)        

  It measures the memory system on the device.
  The implementation is in double precision.

  Code based on the code developed by John D. McCalpin
  http://www.cs.virginia.edu/stream/FTP/Code/stream.c

  Written by: Massimiliano Fatica, NVIDIA Corporation

  Further modifications by: Ben Cumming, CSCS; Andreas Herten (JSC/FZJ)
  MPI - Tim Kaiser

  For Kestrel's 80 GB gpus...
  Runs in about 85 seconds
CC -gpu=cc90  -DNTIMES=1000  mstream.cu -o mstream
export VSIZE=3300000000
srun --tasks-per-node=4  ./mstream -n $VSIZE

*/
#ifndef NGPUS
#define NGPUS 4
#endif
#ifndef NTIMES
#define NTIMES  200000
#endif
#include <string>
#include <vector>

#ifndef SERIAL
#include <mpi.h>
#endif

#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <unistd.h>
#include <sys/time.h>

#include <sys/time.h>

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

typedef double real;

static double   avgtime[4] = {0}, maxtime[4] = {0},
        mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};


void print_help()
{
    printf(
        "Usage: stream [-s] [-n <elements>] [-b <blocksize>]\n\n"
        "  -g"
        "        GPU target\n\n"
        "  -s\n"
        "        Print results in SI units (by default IEC units are used)\n\n"
        "  -n <elements>\n"
        "        Put <elements> values in the arrays\n"
        "        (defaults to 1<<26)\n\n"
        "  -b <blocksize>\n"
        "        Use <blocksize> as the number of threads in each block\n"
        "        (defaults to 192)\n"
    );
}

void parse_options(int argc, char** argv, int& GPU,bool& SI, long& N, int& blockSize)
{
    // Default values
    SI = false;
    N = 1<<26;
    blockSize = 192;
    GPU=-1;
    int c;

    while ((c = getopt (argc, argv, "g:sn:b:h")) != -1)
        switch (c)
        {
            case 'g':
                GPU = std::atoi(optarg);
                break;
            case 's':
                SI = true;
                break;
            case 'n':
                N = std::atol(optarg);
                break;
            case 'b':
                blockSize = std::atoi(optarg);
                break;
            case 'h':
                print_help();
                std::exit(0);
                break;
            default:
                print_help();
                std::exit(1);
        }
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */


double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


template <typename T>
__global__ void set_array(T * __restrict__ const a, T value, long len)
{
    long idx = (long)threadIdx.x + (long)blockIdx.x * (long)blockDim.x;
    if (idx < len)
        a[idx] = value;
}

template <typename T>
__global__ void STREAM_Copy(T const * __restrict__ const a, T * __restrict__ const b, long len)
{
    long idx = (long)threadIdx.x + (long)blockIdx.x * (long)blockDim.x;
    if (idx < len)
        b[idx] = a[idx];
}

template <typename T>
__global__ void STREAM_Scale(T const * __restrict__ const a, T * __restrict__ const b, T scale,  long len)
{
    long idx = (long)threadIdx.x + (long)blockIdx.x * (long)blockDim.x;
    if (idx < len)
        b[idx] = scale * a[idx];
}

template <typename T>
__global__ void STREAM_Add(T const * __restrict__ const a, T const * __restrict__ const b, T * __restrict__ const c, long len)
{
    long idx = (long)threadIdx.x + (long)blockIdx.x * (long)blockDim.x;
    if (idx < len)
        c[idx] = a[idx] + b[idx];
}

template <typename T>
__global__ void STREAM_Triad(T const * __restrict__ a, T const * __restrict__ b, T * __restrict__ const c, T scalar, long len)
{
    long idx = (long)threadIdx.x + (long)blockIdx.x * (long)blockDim.x;
    if (idx < len)
        c[idx] = a[idx] + scalar * b[idx];
}

int main(int argc, char** argv)
{
    real *d_a, *d_b, *d_c;
    int j,k;
    double times[4][NTIMES];
    FILE *FOUT;
    char fname[128];

    real scalar;
    std::vector<std::string> label{"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

    // Parse arguments
    bool SI;
    int GPU;
    int SGPU;
    long N;
    int  blockSize;
    int myid,ntasks,mpi_err;
    char hostbuffer[256];
    cudaDeviceProp prop;
    gethostname(hostbuffer, sizeof(hostbuffer));
    ntasks=1;
    myid=ntasks-1;
    mpi_err=0;
#ifndef SERIAL
    mpi_err=MPI_Init(&argc,&argv);
    mpi_err=MPI_Comm_size(MPI_COMM_WORLD,&ntasks);
    mpi_err=MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif
    sprintf(fname,"strm.%4.4d",myid);
    FOUT=fopen(fname,"w");
    if(myid == 0 ) {
        parse_options(argc, argv, SGPU,SI, N, blockSize);
    }
    // int NGPUS; cudaGetDeviceCount(&NGPUS);
#ifndef SERIAL
    mpi_err=MPI_Bcast(&SGPU,1,MPI_INT,0,MPI_COMM_WORLD);
    mpi_err=MPI_Bcast(&SI,1,MPI_INT,0,MPI_COMM_WORLD);
    mpi_err=MPI_Bcast(&N,1,MPI_LONG,0,MPI_COMM_WORLD);
    mpi_err=MPI_Bcast(&blockSize,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
    GPU=myid % NGPUS;
    if (SGPU >= 0 ) GPU=SGPU;
    if (SGPU >= NGPUS ) GPU= (SGPU  % NGPUS);
    cudaSetDevice(GPU);
    cudaGetDeviceProperties(&prop, GPU);
    double st=mysecond();

    fprintf(FOUT," STREAM Benchmark implementation in CUDA on device %d of %s\n",GPU,hostbuffer);
    fprintf(FOUT," Device name: %s\n", prop.name);
    fprintf(FOUT," Array elements %ld",N);
    fprintf(FOUT," Array size (%s precision) =%7.2f MB\n", sizeof(double)==sizeof(real)?"double":"single", double(N)*double(sizeof(real))/1.e6);
    fprintf(FOUT," Total memory for 3 arrays =%7.2f GB\n",3.0*double(N)*double(sizeof(real))/1.e9);
    fprintf(FOUT," NTIMES %d\n",NTIMES);

    /* Allocate memory on device */
    int ierr;
    ierr=cudaMalloc((void**)&d_a, sizeof(real)*N);if (ierr > 0) fprintf(FOUT,"d_a ierr %d\n",ierr);
    ierr=cudaMalloc((void**)&d_b, sizeof(real)*N);if (ierr > 0) fprintf(FOUT,"d_b ierr %d\n",ierr);
    ierr=cudaMalloc((void**)&d_c, sizeof(real)*N);if (ierr > 0) fprintf(FOUT,"d_c ierr %d\n",ierr);

    /* Compute execution configuration */
    dim3 dimBlock(blockSize);
    dim3 dimGrid(N/dimBlock.x );
    if( N % dimBlock.x != 0 ) dimGrid.x+=1;

    fprintf(FOUT," using %d threads per block, %d blocks\n",dimBlock.x,dimGrid.x);

    if (SI)
        fprintf(FOUT," output in SI units (KB = 1000 B)\n");
    else
        fprintf(FOUT," output in IEC units (KiB = 1024 B)\n");

    /* Initialize memory on the device */
    set_array<real><<<dimGrid,dimBlock>>>(d_a, 2.f, N);
    set_array<real><<<dimGrid,dimBlock>>>(d_b, .5f, N);
    set_array<real><<<dimGrid,dimBlock>>>(d_c, .5f, N);
    double st2=mysecond();

    /*  --- MAIN LOOP --- repeat test cases NTIMES times --- */

    scalar=3.0f;
    for (k=0; k<NTIMES; k++)
    {
        times[0][k]= mysecond();
        STREAM_Copy<real><<<dimGrid,dimBlock>>>(d_a, d_c, N);
        cudaDeviceSynchronize();
        times[0][k]= mysecond() -  times[0][k];

        times[1][k]= mysecond();
        STREAM_Scale<real><<<dimGrid,dimBlock>>>(d_b, d_c, scalar,  N);
        cudaDeviceSynchronize();
        times[1][k]= mysecond() -  times[1][k];

        times[2][k]= mysecond();
        STREAM_Add<real><<<dimGrid,dimBlock>>>(d_a, d_b, d_c,  N);
        cudaDeviceSynchronize();
        times[2][k]= mysecond() -  times[2][k];

        times[3][k]= mysecond();
        STREAM_Triad<real><<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar,  N);
        cudaDeviceSynchronize();
        times[3][k]= mysecond() -  times[3][k];
    }

    /*  --- SUMMARY --- */

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
    {
        for (j=0; j<4; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    double bytes[4] = {
        2 * sizeof(real) * (double)N,
        2 * sizeof(real) * (double)N,
        3 * sizeof(real) * (double)N,
        3 * sizeof(real) * (double)N
    };

    // Use right units
    const double G = SI ? 1.e9 : static_cast<double>(1<<30);
    double f0,f1,f2,f3,f4;
    double tt;
    f0=mysecond();
    fprintf(FOUT,"\nFunction      Rate %s  Avg time(s)  Min time(s)  Max time(s)\n",
           SI ? "(GB/s) " : "(GiB/s)" );
    fprintf(FOUT,"-----------------------------------------------------------------\n");
    f1=mysecond();
    for (j=0; j<4; j++) {
        avgtime[j] = avgtime[j]/(double)(NTIMES-1);

        fprintf(FOUT,"%s%11.4f     %11.8f  %11.8f  %11.8f\n", label[j].c_str(),
                bytes[j]/mintime[j] / G,
                avgtime[j],
                mintime[j],
                maxtime[j]);
    }

    f2=mysecond();
    tt=mysecond()-st;
    fprintf(FOUT," Total time %10.2f seconds\n",tt);
    f3=mysecond();
    fclose(FOUT);
    f4=mysecond();
    /* Free memory on device */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    printf("%4.4d %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n",myid,st,st2,f0,f1,f2,f3,f4,tt);
#ifdef DEBUG
    sprintf(fname,"times.%4.4d",myid);
    FOUT=fopen(fname,"w");
    for (k=0; k<NTIMES; k++) 
    {
        fprintf(FOUT,"%15.9f %15.9f %15.9f %15.9f\n",times[0][k],times[1][k],times[2][k],times[3][k]);
    }
    fclose(FOUT);
#endif    
#ifndef SERIAL
    mpi_err = MPI_Finalize();
#endif
    return (mpi_err);
}

