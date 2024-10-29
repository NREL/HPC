#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
void hold(const char *filename, float dt);
void mone(double *A, long N) {
     for (int i=0 ; i< N;i++) A[i]=-1.0;
}





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
    int modify=0;
    int async=0;
    if (rank == 0 ){
    	printf("\nRunning %s\n",argv[0]);
    	char version[MPI_MAX_LIBRARY_VERSION_STRING];
    	int vlan;
    	MPI_Get_library_version(version, &vlan);
	    printf("%s\n",version);    	
    	if(argc > 1)sscanf(argv[1],"%d",&modify);
    	if (modify) {
    		printf("Modifying vector\n");
    	}
    	else {
    		printf("NOT Modifying vector\n");
    	}
		if(argc > 2)sscanf(argv[2],"%d",&async);
		if (async) {
			printf("Calling MPI_Isend\n");
			}
		else {
			printf("Calling MPI_Send\n");
		}
    }
    MPI_Bcast(&async, 1, MPI_INT, 0,MPI_COMM_WORLD);
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
   
    
#ifdef HOLD
	hold("continue",2.0);
#endif


	/* -------------------------------------------------------------------------------------------
		Loop from 8 B to 1 GB
	--------------------------------------------------------------------------------------------*/

	for(int i=0; i<=27; i++){

		long int N = 1 << i;
/* Compute execution configuration */
MPI_Request swait,rwait;



	
		// Allocate memory for A on CPU
		double *A = (double*)malloc(N*sizeof(double));
		double *B = (double*)malloc(N*sizeof(double));
		double *C = (double*)malloc(N*sizeof(double));

		// Initialize all elements of A to random values
		// B to 0
		for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
            B[i] = 0.0;
		}

	
		int tag1 = 10;
		int tag2 = 20;
		
	
		// Do a single send
		if(rank == 0){
			printf("N=%ld\n",N);
			if (async) {
				MPI_Isend(A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD,&swait);
				MPI_Wait(&swait,&stat);
				if (modify) {
					// MONE<double><<<dimGrid,dimBlock>>>(d_A,  N);
					mone(A,N);
				}
				MPI_Irecv(B, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD,&rwait);
				MPI_Wait(&rwait,&stat);
			}
			else {
				MPI_Send(A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
				if (modify) {
				 	// MONE<double><<<dimGrid,dimBlock>>>(d_A,  N);
					mone(A,N);
				}
				MPI_Recv(B, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
			}
		}
		else if(rank == 1){
		if (async) {
			MPI_Irecv(C, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD,&rwait);
			MPI_Wait(&rwait,&stat);
			MPI_Isend(C, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD,&swait);
			MPI_Wait(&swait,&stat);
			}
			else {
				MPI_Recv(C, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
				MPI_Send(C, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
			}		
		}
		if (rank == 0){
			for(int i=0; i<N; i++){
				if((B[i]) <= 0.0) {
					printf(" FAIL %ld %d %g\n",N,i,B[i]);
					i=N;
				}
			}
		}


		free(A);
		free(B);
		free(C);
	}

	MPI_Finalize();

	return 0;
}
