#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
	cudaError_t cuErr = call;                                                             \
	if(cudaSuccess != cuErr){                                                             \
		printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
		exit(0);                                                                            \
	}                                                                                     \
}while(0)


int main(int argc, char *argv[])
{
	/* -------------------------------------------------------------------------------------------
		MPI Initialization 
	--------------------------------------------------------------------------------------------*/
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Status stat;

	if(size != 2){
		if(rank == 0){
			printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
		}
		MPI_Finalize();
		exit(0);
	}
	if ( rank == 0 ) {
		char version[MPI_MAX_LIBRARY_VERSION_STRING];
		int vlan;
		MPI_Get_library_version(version, &vlan);
    		printf("%s\n",version);
	}
    char myname[MPI_MAX_PROCESSOR_NAME] ;
    char name1[MPI_MAX_PROCESSOR_NAME] ;
    int resultlen;
    MPI_Get_processor_name(myname,&resultlen);
    if(rank == 1)
            MPI_Send(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    else {  
            MPI_Recv(name1, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &stat); 
            if (strcmp(myname,name1) != 0) { 
                printf("Two nodes %s %s\n",myname,name1);
            }       
            else {  
                printf("One node %s\n",myname);
            }       
    }       

	// Map MPI ranks to GPUs
    int num_devices = 0;
    cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
	cudaErrorCheck( cudaSetDevice(rank % num_devices) );

	/* -------------------------------------------------------------------------------------------
		Loop from 8 B to 1 GB
	--------------------------------------------------------------------------------------------*/

	for(int i=0; i<=27; i++){

		long int N = 1 << i;
	
		// Allocate memory for A on CPU
		double *A = (double*)malloc(N*sizeof(double));

		// Initialize all elements of A to random values
		for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
		}

		double *d_A;
		cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
		cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
	
		int tag1 = 10;
		int tag2 = 20;
	
		int loop_count = 50;

		// Warm-up loop
		for(int i=1; i<=5; i++){
			if(rank == 0){
				MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
				MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
			}
			else if(rank == 1){
				MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
				MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
			}
		}

		// Time ping-pong for loop_count iterations of data transfer size 8*N bytes
		double start_time, stop_time, elapsed_time;
		start_time = MPI_Wtime();
	
		for(int i=1; i<=loop_count; i++){
			if(rank == 0){
				MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
				MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
			}
			else if(rank == 1){
				MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
				MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
			}
		}

		stop_time = MPI_Wtime();
		elapsed_time = stop_time - start_time;

		long int num_B = 8*N;
		long int B_in_GB = 1 << 30;
		double num_GB = (double)num_B / (double)B_in_GB;
		double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

		if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

		cudaErrorCheck( cudaFree(d_A) );
		free(A);
	}

	MPI_Finalize();

	return 0;
}
