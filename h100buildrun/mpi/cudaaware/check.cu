#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
void hold(const char *filename, float dt);


double mysecond();
// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
	cudaError_t cuErr = call;                                                             \
	if(cudaSuccess != cuErr){                                                             \
		printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
		exit(0);                                                                            \
	}                                                                                     \
}while(0)



#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

/**/
template <typename T>
__global__ void STREAM_Copy(T const * __restrict__ const a, T * __restrict__ const b, long len)
{
    long idx = (long)threadIdx.x + (long)blockIdx.x * (long)blockDim.x;
    if (idx < len)
        b[idx] = a[idx];
}
/**/

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
#ifdef HOLD
	hold("continue",2.0);
#endif
	// Map MPI ranks to GPUs
    int num_devices = 0;
    cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
	cudaErrorCheck( cudaSetDevice(rank % num_devices) );
	
int blockSize = 192;
int NT=4;


	/* -------------------------------------------------------------------------------------------
		Loop from 8 B to 1 GB
	--------------------------------------------------------------------------------------------*/

	for(int i=0; i<=27; i++){

		long int N = 1 << i;
/* Compute execution configuration */
NT=N;
dim3 dimBlock(blockSize);
dim3 dimGrid(NT/dimBlock.x );
if( NT % dimBlock.x != 0 ) dimGrid.x+=1;
int tomove;
#ifdef ASYNC
MPI_Request swait,rwait;
#endif

#ifdef DOFULL
tomove=N;
#else
tomove=MIN(N,8);
#endif


	
		// Allocate memory for A on CPU
		double *A = (double*)malloc(N*sizeof(double));
		double *B = (double*)malloc(N*sizeof(double));
		double *D = (double*)malloc(N*sizeof(double));

		// Initialize all elements of A to random values
		for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
            B[i] = -1.0;
		}

		double *d_A;
		double *d_B;
		double *d_C;
		double *d_D;
		cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
		cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
		cudaErrorCheck( cudaMalloc(&d_C, N*sizeof(double)) );
		cudaErrorCheck( cudaMalloc(&d_D, N*sizeof(double)) );
		cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
		cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(double), cudaMemcpyHostToDevice) );
	
		int tag1 = 10;
		int tag2 = 20;
	
		int loop_count = 50;

		// Warm-up loop
		for(int i=1; i<=5; i++){
			if(rank == 0){
#ifdef ASYNC
				MPI_Isend(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD,&swait);
				MPI_Irecv(d_B, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD,&rwait);
				MPI_Wait(&swait,&stat);
				MPI_Wait(&rwait,&stat);
				STREAM_Copy<double><<<dimGrid,dimBlock>>>(d_B, d_D, NT);
#else
				MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
				MPI_Recv(d_B, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
				STREAM_Copy<double><<<dimGrid,dimBlock>>>(d_B, d_D, NT);
#endif
			}
			else if(rank == 1){
#ifdef ASYNC
				MPI_Irecv(d_C, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD,&rwait);
				MPI_Isend(d_C, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD,&swait);
				MPI_Wait(&rwait,&stat);
				MPI_Wait(&swait,&stat);
				STREAM_Copy<double><<<dimGrid,dimBlock>>>(d_C, d_D, NT);
#else
				MPI_Recv(d_C, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
				MPI_Send(d_C, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
				STREAM_Copy<double><<<dimGrid,dimBlock>>>(d_C, d_D, NT);
#endif
			}
		}

		// Time ping-pong for loop_count iterations of data transfer size 8*N bytes
		double start_time, stop_time, elapsed_time;
		start_time = MPI_Wtime();
	
		for(int i=1; i<=loop_count; i++){
			if(rank == 0){
#ifdef ASYNC
				MPI_Isend(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD,&swait);
				MPI_Irecv(d_B, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD,&rwait);
				MPI_Wait(&swait,&stat);
				MPI_Wait(&rwait,&stat);
#else
				MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
				MPI_Recv(d_B, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
#endif
#ifdef DOCOPY
				 STREAM_Copy<double><<<dimGrid,dimBlock>>>(d_B, d_D, NT);
#ifdef COPYOUT
				 cudaErrorCheck( cudaMemcpy(D, d_D, tomove*sizeof(double), cudaMemcpyDeviceToHost) );
				 if (i == loop_count){fprintf(stderr,"%d %g %g\n",rank,D[0],D[tomove-1]);}
#endif
#endif
				 
			}
			else if(rank == 1){
#ifdef ASYNC
				MPI_Irecv(d_C, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD,&rwait);
				MPI_Isend(d_C, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD,&swait);
				MPI_Wait(&rwait,&stat);
				MPI_Wait(&swait,&stat);
#else
				MPI_Recv(d_C, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
				MPI_Send(d_C, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
#endif
#ifdef DOCOPY
				 STREAM_Copy<double><<<dimGrid,dimBlock>>>(d_C, d_D, NT);
#ifdef COPYOUT
				 cudaErrorCheck( cudaMemcpy(D, d_D, tomove*sizeof(double), cudaMemcpyDeviceToHost) );
				 if (i == loop_count){fprintf(stderr,"%d %g %g\n",rank,D[0],D[tomove-1]);}
#endif
#endif
			}
		}

		stop_time = MPI_Wtime();
		elapsed_time = stop_time - start_time;
		if (rank == 0){
			cudaErrorCheck( cudaMemcpy(B, d_B, N*sizeof(double), cudaMemcpyDeviceToHost) );
			for(int i=0; i<N; i++){
				if((A[i]-B[i]) != 0) {
					printf("%d %d %g %g %g\n",N,i,A[i],B[i],A[i]-B[i]);
					i=N;
				}
			}
		}

		long int num_B = 8*N;
		long int B_in_GB = 1 << 30;
		double num_GB = (double)num_B / (double)B_in_GB;
		double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

		if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

		cudaErrorCheck( cudaFree(d_A) );
		free(A);
		cudaErrorCheck( cudaFree(d_B) );
		free(B);
		cudaErrorCheck( cudaFree(d_C) );
		cudaErrorCheck( cudaFree(d_D) );
		free(D);
		// cudaFree(dimBlock);
		// cudaFree(dimGrid);
	}

	MPI_Finalize();

	return 0;
}
