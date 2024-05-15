#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
 
/************************************************************
This is a simple hello world program. Each processor prints
name, rank, and total run size.
************************************************************/
int main(int argc, char **argv)
{
    int myid,numprocs,resultlen;
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    char myname[MPI_MAX_PROCESSOR_NAME] ;
    int vlan;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(myname,&resultlen); 
    if (myid == 0 ) {
	    printf("Running: %s\n",argv[0]);
	    MPI_Get_library_version(version, &vlan);
	    printf("%s\n",version);
    }
    printf("Hello from %s %d %d\n",myname,myid,numprocs);
    MPI_Finalize();
}

