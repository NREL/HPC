#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
//#include <utmpx.h>
#include <sys/time.h>

double omp_get_wtime(void);
void triad(int NTIMES, int val,int myid);
double dotriad(int *myid) ;

double dotriad(int *myid) {
  double dt,t1,t2;
  int val;
  int ml=*myid;
  if (ml < 0 ) {
	  dt=-ml;
	  val=10;
  } else {
	 dt=1e-15; 
	 val=-1;
  }
  if (dt > 0)
    {
      t1 = omp_get_wtime();
      t2 = t1;
      while (dt > t2 - t1)
        {
              triad(10, val,ml);
              t2 = omp_get_wtime();
        }
    }
  return(t2-t1);
  }

void triad(int NTIMES, int val,int myid) {
#include <float.h>

static double	avgtime, maxtime,mintime;
size_t bytes;
size_t j,k;
int ic;
# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif


#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif
#ifndef OFFSET
#   define OFFSET	0
#endif
#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif
ic=0;
STREAM_TYPE atot;
STREAM_TYPE scalar;
//printf("from triad %d\n",myid);
if (val >= 0)
    scalar=(STREAM_TYPE)val;
else
    scalar=3.0;

double		t,*times;
times=(double*)malloc(NTIMES*sizeof(double));

static STREAM_TYPE	a[STREAM_ARRAY_SIZE+OFFSET],
			b[STREAM_ARRAY_SIZE+OFFSET],
			c[STREAM_ARRAY_SIZE+OFFSET];

avgtime= 0; 
maxtime = 0;
mintime= FLT_MAX;
#pragma omp parallel for
    for (j=0; j<STREAM_ARRAY_SIZE; j++) {
	    a[j] = 1.0;
	    b[j] = 2.0;
	    c[j] = 0.0;
	}

    for (k=0; k<NTIMES; k++)
	{
	times[k] = omp_get_wtime();
#pragma omp parallel for
	for (j=0; j<STREAM_ARRAY_SIZE; j++)
	    a[j] = b[j]+scalar*c[j];
	times[k] = omp_get_wtime() - times[k];
	}
    atot=0;
    for (j=0; j<STREAM_ARRAY_SIZE; j++) atot=atot+a[j];
    if (atot < 0)printf("%g\n",(double)atot);

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	    avgtime= avgtime + times[k];
	    mintime= MIN(mintime, times[k]);
	    maxtime= MAX(maxtime, times[k]);
	}
        avgtime=avgtime;
	mintime=mintime;
	maxtime=maxtime;
	avgtime = avgtime/(double)(NTIMES-1);
	bytes=3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE;
	ic++;
	if (val == -1){
	if (myid == 0 )fprintf(stderr,"TASK   Function    Best Rate MB/s  Avg time     Min time     Max time  Called\n");
	fprintf(stderr,"%6d %s  %12.1f      %11.6f  %11.6f  %11.6f  %6d\n", myid,"triad",
	       1.0E-06 * bytes/mintime,
	       avgtime,
	       mintime,
	       maxtime,
	       ic);
	}
	free(times);

}


