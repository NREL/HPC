#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
void pass(int myid,int numprocs); 
void waste(double t);
void printenv_();
void chkerr(int ierr,int myid, char* routine);

/************************************************************
This is a simple hello world program. Each processor prints
name, rank, and total run size.
************************************************************/
int main(int argc, char **argv ,char **envp)
{
    int myid,numprocs,resultlen;
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    char myname[MPI_MAX_PROCESSOR_NAME] ;
    int vlan;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    if (myid == 0 ) {
	    MPI_Get_library_version(version, &vlan);
	    printf("%s\n",version);
	    // printenv_();
    }
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Get_processor_name(myname,&resultlen); 
    printf(" Hello from %s (C) %d of %d\n",myname,myid,numprocs);
    if(numprocs > 1)pass(myid,numprocs);
    if (myid == 0 )printf(" SUCCESS\n");
    waste(1.0);
    MPI_Finalize();
}
void pass(int myid,int numprocs) {
#include <mpi.h>
	MPI_Status status;
	int my_tag,to,from,i,ierr;
	my_tag=1234;
	i=myid;
	to=myid+1;
	from=myid-1;
	ierr=MPI_Bcast(&i,1,MPI_INT,0, MPI_COMM_WORLD); chkerr(ierr,myid,"bcast");
	if( i != 0) {
	    printf("bcast failed %d\n",myid);
	    ierr=MPI_Abort(MPI_COMM_WORLD,-1);
	    }
	if (myid == 0){
		from=numprocs-1;
		i=1;
		ierr=MPI_Send(&i,1,MPI_INT,to,my_tag,MPI_COMM_WORLD); chkerr(ierr,myid,"send");
		ierr=MPI_Recv(&i,1,MPI_INT,from,my_tag,MPI_COMM_WORLD,&status); chkerr(ierr,myid,"recv");
		if (i != numprocs){
	        printf("send /recv failed %d\n",i);
	        ierr=MPI_Abort(MPI_COMM_WORLD,-1);
		}
	}
	else {
		if (myid == numprocs-1)to=0;
		ierr=MPI_Recv(&i,1,MPI_INT,from,my_tag,MPI_COMM_WORLD,&status); chkerr(ierr,myid,"recv");
		i=i+1;
		ierr=MPI_Send(&i,1,MPI_INT,to,my_tag,MPI_COMM_WORLD); chkerr(ierr,myid,"send");
	}
}

void chkerr(int ierr,int myid, char* routine){
#include <mpi.h>
if(ierr != 0 ){
    printf("%s failed on %d\n",routine,myid);
    MPI_Abort(MPI_COMM_WORLD,-3);
    }
}

void waste(double t) {
	double t1;
	int *avect;
	int i,msize;
	msize=10000000;
	t1=MPI_Wtime()+t;
	while (MPI_Wtime() < t1) {
		avect=(int*)malloc(msize*sizeof(int));
		for (i=0;i<msize;i++){
			avect[i]=i;
		}
		int j;
		j=0;
		for (i=0;i<msize;i++){
			j=j+avect[i];
		}
		if (j == 0)printf("nope\n");
		free(avect);
	}
}
void printenv_() {
extern char **environ;
char **s = environ;
  for (; *s; s++) {
    printf("%s\n", *s);
  }
}

