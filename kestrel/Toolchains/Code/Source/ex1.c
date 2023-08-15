#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>
int sched_getcpu();
double dotriad();
double dotriad(int *myid);
/************************************************************
This is a simple hybrid hello world program.
Prints MPI information 
For each task/thread
  task id
  node name for task
  thread id
  # of threads for the task
  core on which the thread is running
************************************************************/
int main(int argc, char **argv)
{
    int myid,numtasks,resultlen;
    int did;
    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    char myname[MPI_MAX_PROCESSOR_NAME] ;
    int vlan;
    int mycore;
    double wait;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(myname,&resultlen); 
    if (myid == 0 ) {
	    printf(" C MPI TASKS %d\n",numtasks);
	    MPI_Get_library_version(version, &vlan);
            printf("%s\n",version);
            printf("%s\n",__VERSION__);
    }
// dotriad runs "triad", in parallel, for 4 seconds to give threads time to settle
//  if input to triad is negative run for -# seconds
//  if >=0 run "triad", in parallel one more time and give report to stderr
    did=-4;
    wait=dotriad(&did);
#pragma omp parallel 
  {
#pragma omp critical
    mycore=sched_getcpu();
    printf(" task %04d is running on %s  thread %3d of %3d is on core %03d\n",
		    myid,
		    myname,
		    omp_get_thread_num(),
		    omp_get_thread_num(),
		    mycore);
  }
  if (myid == 0)printf("ran triad for %10.2f seconds\n",wait);
// run "triad", in parallel one more time and give report to stderr
    did=myid;
//    wait=dotriad(&did);
    MPI_Finalize();
}

