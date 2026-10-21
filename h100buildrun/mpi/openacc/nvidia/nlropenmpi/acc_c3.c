/* 
 *     Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

/*
 * Modified the original program to use long instead of int for various
 * values to enable bigger matrix size
 */


// can hard code the number of GPUs if -1 autodetect
#ifndef DEVS
#define DEVS -1
#endif



/*
 * Jacobi iteration example using OpenACC in C
 * Build with
 *   module load nvhpc
 *   mpicc -acc -Minfo=accel -fast acc_c3.c -o jacobi
 *
 *   mpirun -N 4 ./jacobi 46000 46000 5 nvidia
 *
 * Hack to build with Intel MPI and nvc
 *   module load nvhpc
 *   module load intel-oneapi-mpi
 *   ln -s `which nvc` `pwd`/gcc
 *   export I_MPI_CC=`pwd`/gcc
 *   mpicc  -acc -Minfo=accel -fast acc_c3.c -o jacobi_i
 *   rm `pwd`/gcc
 *
 *   srun --tasks-per-node=4 ./jacobi_i 46000 46000 10 nvidia
 */
 
/*
 * To target cores:
 *   mpicc  -acc -Minfo=accel -fast -target=multicore acc_c3.c -o jacobi_c
 *   export ACC_NUM_CORES=10
 *   mpirun -N 4 ./jacobi_c 46000 46000 5 default
 */

/*
 * mpirun -N 4 ./jacobi n m iter device
 * n      = matrix size
 * m      = matrix size
 * iter   = number of iterations
 * device = host or nvidia
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <openacc.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#ifdef USEINT
#define INT int
#define MYINT MPI_INT
#else
#define INT long
#define MYINT MPI_LONG
#endif




#if defined(_WIN32) || defined(_WIN64)
#include <sys/timeb.h>
#define gettime(a) _ftime(a)
#define usec(t1,t2) ((((t2).time-(t1).time)*1000+((t2).millitm-(t1).millitm))*100)
typedef struct _timeb timestruct;
#else
#include <sys/time.h>
#define gettime(a) gettimeofday(a,NULL)
#define usec(t1,t2) (((t2).tv_sec-(t1).tv_sec)*1000000+((t2).tv_usec-(t1).tv_usec))
typedef struct timeval timestruct;
#endif

void
smooth( float*restrict a, float*restrict b, float w0, float w1, float w2, INT n, INT m, INT niters )
{
    INT i, j, iter;
    float* tmp;
    for( iter = 1; iter <= niters; ++iter ){
	#pragma acc kernels loop copyin(b[0:n*m]) copy(a[0:n*m]) independent
	for( i = 1; i < n-1; ++i )
	    for( j = 1; j < m-1; ++j )
		a[i*m+j] = w0 * b[i*m+j] + 
		    w1*(b[(i-1)*m+j] + b[(i+1)*m+j] + b[i*m+j-1] + b[i*m+j+1]) +
		    w2*(b[(i-1)*m+j-1] + b[(i-1)*m+j+1] + b[(i+1)*m+j-1] + b[(i+1)*m+j+1]);
	tmp = a;  a = b;  b = tmp;
    }
}

void
smoothhost( float*restrict a, float*restrict b, float w0, float w1, float w2, INT n, INT m, INT niters )
{
    INT i, j, iter;
    float* tmp;
    for( iter = 1; iter <= niters; ++iter ){
	for( i = 1; i < n-1; ++i ){
	    for( j = 1; j < m-1; ++j ){
		a[i*m+j] = w0 * b[i*m+j] + 
		    w1*(b[(i-1)*m+j] + b[(i+1)*m+j] + b[i*m+j-1] + b[i*m+j+1]) +
		    w2*(b[(i-1)*m+j-1] + b[(i-1)*m+j+1] + b[(i+1)*m+j-1] + b[(i+1)*m+j+1]);
		}
	}
	tmp = a;  a = b;  b = tmp;
    }
}

void
doprt( char* s, float*restrict a, float*restrict ah, INT i, INT j, INT n, INT m )
{
    printf( "%s[%d][%d] = %g  =  %g\n", s, i, j, a[i*m+j], ah[i*m+j] );
}

int getpid();

int
main( int argc, char* argv[] )
{
    float *aa, *bb, *aahost, *bbhost;
    INT i,j;
    float w0, w1, w2;
    INT n, m, aerrs, berrs, iters;
    float dif, rdif, tol;
    timestruct t1, t2, t3;
    long long cgpu, chost;
    long long gcgpu, gchost;
    long ld;
    int thedevice;
    int myid,numprocs,resultlen;
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    char myname[MPI_MAX_PROCESSOR_NAME] ;
    int vlan;
    int devs;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(myname,&resultlen); 
    printf("Hello from %s running task %d of %d with pid %d\n",myname,myid,numprocs,getpid());

    
    if(myid == 0) {
    	n = 0;
		m = 0;
		iters = 0;
		thedevice=0;
		if( argc > 1 ){
		n = atoi( argv[1] );
		if( argc > 2 ){
			m = atoi( argv[2] );
			if( argc > 3 ){
			iters = atoi( argv[3] );
			if( argc > 4 ){
				if( !strcmp( argv[4], "host" ) ||
				!strcmp( argv[4], "HOST" ) ){
				//acc_set_device( acc_device_host );
				thedevice=3;
				printf( "using host\n" );
				}else
				if( !strcmp( argv[4], "nvidia" ) ||
				!strcmp( argv[4], "NVIDIA" ) ){
				//acc_set_device( acc_device_nvidia );
				//acc_init( acc_device_nvidia );
				thedevice=4;
				printf( "using nvidia\n" );
				}else{
				printf( "unknown device: %s\nUsing default\n", argv[4] );
				}
			}
			}
		}
		}
    }
    MPI_Bcast(&n,           1,MYINT,   0,MPI_COMM_WORLD);
    MPI_Bcast(&m,           1,MYINT,   0,MPI_COMM_WORLD);
    MPI_Bcast(&iters,       1,MYINT,   0,MPI_COMM_WORLD);
    MPI_Bcast(&thedevice,   1,MPI_INT, 0,MPI_COMM_WORLD);
    if (myid == 0 ) {
	MPI_Get_library_version(version, &vlan);
	printf("%s\n",version);
	ld=(long)n *(long)m * (long)2 * (long)4;
	ld=ld/(long)1048576;
	printf( "matrix %d x %d, %d iterations with memory/task %ld MB\n", n, m, iters, ld);
    }
    if(thedevice == 3){
        acc_set_device_type( acc_device_host );
    }
    if(thedevice == 4){
		// acc_set_device_type( acc_device_nvidia );
		if (DEVS > 0 ) {
			acc_set_device_num(myid % DEVS, acc_device_nvidia );
		}
		else {
			devs=acc_get_num_devices(acc_device_nvidia);
			acc_set_device_num(myid % devs, acc_device_nvidia );
		}
		acc_init( acc_device_nvidia );
    }
    if( n <= 0 ) n = 1000;
    if( m <= 0 ) m = n;
    if( iters <= 0 ) iters = 10;

    aa = (float*) malloc( sizeof(float) * n * m );
    aahost = (float*) malloc( sizeof(float) * n * m );
    bb = (float*)malloc( sizeof(float) * n * m );
    bbhost = (float*)malloc( sizeof(float) * n * m );
    for( i = 0; i < n; ++i ){
	for( j = 0; j < m; ++j ){
	    aa[i*m+j] = 0;
	    aahost[i*m+j] = 0;
	    bb[i*m+j] = i*1000 + j;
	    bbhost[i*m+j] = i*1000 + j;
	}
    }
    w0 = 0.5;
    w1 = 0.3;
    w2 = 0.2;
    gettime( &t1 );
    smooth( aa, bb, w0, w1, w2, n, m, iters );
    gettime( &t2 );
    smoothhost( aahost, bbhost, w0, w1, w2, n, m, iters );
    gettime( &t3 );

    cgpu = usec(t1,t2);
    chost = usec(t2,t3);
    
    MPI_Reduce(&cgpu,    &gcgpu,  1,  MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&chost,   &gchost, 1,  MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myid == 0){
        ld=(long)n *(long)m * (long)2 * (long)4;
        ld=ld/(long)1048576;
		
		printf( "%13ld microseconds optimized (max)\n", gcgpu );
		printf( "%13ld microseconds on host   (max)\n", gchost );
	}
	
    MPI_Reduce(&cgpu,    &gcgpu,  1,  MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&chost,   &gchost, 1,  MPI_LONG_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
		
    if (myid == 0){
		printf( "%13ld microseconds optimized (min)\n", gcgpu );
		printf( "%13ld microseconds on host   (min)\n", gchost );
    }

    aerrs = berrs = 0;
    tol = 0.000005;
    for( i = 0; i < n; ++i ){
	for( j = 0; j < m; ++j ){
	    rdif = dif = fabsf(aa[i*m+j] - aahost[i*m+j]);
	    if( aahost[i*m+j] ) rdif = fabsf(dif / aahost[i*m+j]);
	    if( rdif > tol ){
		++aerrs;
		if( aerrs < 10 ){
		    printf( "aa[%d][%d] = %12.7e != %12.7e, dif=%12.7e\n", i, j, (double)aa[i*m+j], (double)aahost[i*m+j], (double)dif );
		}
	    }
	    rdif = dif = fabsf(bb[i*m+j] - bbhost[i*m+j]);
	    if( bbhost[i*m+j] ) rdif = fabsf(dif / bbhost[i*m+j]);
	    if( rdif > tol ){
		++berrs;
		if( berrs < 10 ){
		    printf( "bb[%d][%d] = %12.7e != %12.7e, dif=%12.7e\n", i, j, (double)bb[i*m+j], (double)bbhost[i*m+j], (double)dif );
		}
	    }
	}
    }
    if( aerrs == 0 && berrs == 0 ){
	//printf( "Test PASSED\n" );
	MPI_Finalize();
	return 0;
    }else{
	printf( "Test FAILED\n" );
	printf( "%d %d ERRORS found\n", myid,aerrs + berrs );
	MPI_Finalize();
	return 1;
    }
}

