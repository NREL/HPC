#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
int cumain(int myid, int gx, int gy,int bx, int by, int bz);
void ptime(){
  time_t rawtime;
  struct tm * timeinfo;
  char buffer [80];
  time (&rawtime);
  timeinfo = localtime (&rawtime);
  strftime (buffer,80,"%c",timeinfo);
  //puts (buffer);
  printf("%s\n",buffer);
  }
  
#define GPUS 2
/************************************************************
This is a simple hello world program. Each processor prints
name, rank, and total run size.
************************************************************/
int main(int argc, char **argv)
{
    int myid,numprocs,resultlen;
    char myname[MPI_MAX_PROCESSOR_NAME] ;
	int gx,gy;
	int bx,by,bz;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(myname,&resultlen); 
    printf("C-> Hello from %s # %d of %d\n",myname,myid,numprocs);

    if (myid == 0) {
		ptime();
		//scanf("%d %d",&gx,&gy);
		//scanf("%d %d %d",&bx,&by,&bz);
		sscanf("2 3","%d %d",&gx,&gy);
		sscanf("2 3 4","%d %d %d",&bx,&by,&bz);
	}

	MPI_Bcast(&gx,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&gy,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&bx,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&by,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&bz,1,MPI_INT,0,MPI_COMM_WORLD);

    for (int ic=0; ic<30; ic++ ){
// call a routine that uses GPS - does nothing important
	cumain(myid % GPUS ,gx,gy,bx,by,bz);
    	sleep(1);
    }
    if(myid == 0) {
                ptime();
    }
    MPI_Finalize();
}
