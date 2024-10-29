#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char *argv[]) {
	/* -------------------------------------------------------------------------------------------
		MPI Initialization 
	--------------------------------------------------------------------------------------------*/
	MPI_Init(&argc, &argv);

	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// we want the same, or different values for each task
// comment out as desired
        srand(1234);
      //srand(100+rank);
//


	double sum;
// which rank reports results, -1 for all
	int rcheck;
#ifdef RCHECK
	rcheck=RCHECK
#else
	rcheck=size-1;
#endif
	if (rank == 0 && argc > 1){
		sscanf(argv[1],"%d",&rcheck);
	}
	MPI_Bcast(&rcheck,1,MPI_INT,  0,MPI_COMM_WORLD);	

	/* -------------------------------------------------------------------------------------------
		Loop from 8 B to 1 GB
	--------------------------------------------------------------------------------------------*/
int j;
	for(int i=0; i<=27; i++){

		long int N = 1 << i;
	
		// Allocate memory for A on CPU
		double *A = (double*)malloc(N*size*sizeof(double));
		double *B = (double*)malloc(N*size*sizeof(double));

		// Initialize all elements of A to random values
		for(int i=0; i<N*size; i++){
            		A[i] = rank+(double)rand()/(double)RAND_MAX;
		}

	
	
		int loop_count = 50;

		// Warm-up loop
		for(j=1; j<=5; j++){
			MPI_Alltoall(A, N, MPI_DOUBLE, B, N, MPI_DOUBLE,MPI_COMM_WORLD);
		}

		// Time all-to-all for loop_count iterations of data transfer size 8*N bytes/task
		double start_time, stop_time, elapsed_time;
		start_time = MPI_Wtime();
	
		for(j=1; j<=loop_count; j++){
			MPI_Alltoall(A, N, MPI_DOUBLE, B, N, MPI_DOUBLE,MPI_COMM_WORLD);
		}

		stop_time = MPI_Wtime();
		elapsed_time = stop_time - start_time;

		long int num_B = 8*N;
		long int B_in_GB = 1 << 30;
		double num_GB = (double)num_B / (double)B_in_GB;
		double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

		if(rank == 0) printf("Transfer size/Task (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

		free(A);
			int id;
			for (id=0 ; id<size;id++) {
				sum=0.0;
				for(int k=0; k<N;k++) {
					sum=sum+B[id*N+k];
				}
				sum=sum/N;
				// Should be about N+0.5 for larger message sizes  
				// with fractional portion the same.
				if(rank == rcheck || rcheck == -1){
					printf("%d %d %ld %15.7f\n",rank,id,N,sum);
					if (id == (size-1))printf("\n");
				}
			};
		free(B);
	}

	MPI_Finalize();

	return 0;
}
