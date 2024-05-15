#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#define REAL 0
#define IMAG 1

#define THREED

#ifdef ONED
#undef THREED
#endif

#ifdef TWOD
#undef THREED
#endif

#ifndef PMAX
#define PMAX 512
#endif

/* fftw_complex *an_array;
an_array = (fftw_complex*) fftw_malloc(5*12*27 * sizeof(fftw_complex));
Accessing the array elements, however, is more tricky—you can’t simply use 
multiple applications of the ‘[]’ operator like you could for fixed-size 
arrays. Instead, you have to explicitly compute the offset into the array 
using the formula given earlier for row-major arrays. For example, to 
reference the (i,j,k)-th element of the array allocated above, you would 
use the expression an_array[k + 27 * (j + 12 * i)].
See also https://www.fftw.org/fftw3_doc/Row_002dmajor-Format.html
*/

double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void printit(fftw_complex* result,int N,char x) {
    int i;
    for (i = 0; i < N; ++i) {
        double mag = sqrt(result[i][REAL] * result[i][REAL] +
                          result[i][IMAG] * result[i][IMAG]);
        printf("%c %23.12f %10.5f %10.5f\n", x,mag,result[i][REAL] ,result[i][IMAG]);
    }
}

void trig(fftw_complex* signal, int N){
//n=i^d-1 + n^d-1 * (i^d-2 + n^d-2 * (... + n^1 * i^0))
//n=i2 + n2 * (i1 + n1 * i0)
//n=i2 + i2max * (i1 + i1max * i0);

	double t0,t1,t2;
	double rp0,rp1,rp2;
	double pi2;
	int i0,i1,i2,n;
	int i0max,i1max,i2max;
	i0max=N;
	i1max=N;
	i2max=N;
	pi2=2.0*M_PI;
	printf("N=%d\n",N);
#ifdef ONED
	  for (i0=0;i0<i0max;i0++){
	  	t0 = (double)i0 / (double)i0max * pi2;
		rp0=1.00*cos(1.00*t0)*sin(1.00*t0);
		n=i0;
		signal[n][REAL]=rp0;
		signal[n][IMAG]=0.0;
#ifdef PRINT
		printf("+ %4d %4d %10.6f %10.6f\n",i0,n,t0,signal[n][REAL]);	
#endif
		}
#endif
#ifdef TWOD
	  for (i0=0;i0<i0max;i0++){
	  	t0 = (double)i0 / (double)i0max * pi2;
	  	rp0=1.00*cos(1.00*t0)*sin(1.00*t0);
	  	for(i1=0;i1<i1max;i1++){
	  		t1 = (double)i1 / (double)i1max * pi2;
	  		rp1=sin(1.00*t1);
	  		n=(i1 + i1max * i0);
	  		signal[n][REAL]=rp0*rp1;
	  		signal[n][IMAG]=0.0;
#ifdef PRINT
			printf("+ %4d %4d %4d %10.6f %10.6f %10.6f\n",i0,i1,n,t0,t1,signal[n][REAL]);	  		
#endif
	  	}
	  }
#endif
#ifdef THREED
	  for (i0=0;i0<i0max;i0++){
	  	t0 = (double)i0 / (double)i0max * pi2;
	  	rp0=1.00*cos(1.00*t0)*sin(1.00*t0);
	  	for(i1=0;i1<i1max;i1++){
	  		t1 = (double)i1 / (double)i1max * pi2;
	  		rp1=sin(1.00*t1);
			for(i2=0;i2<i2max;i2++){
				t2 = (double)i2 / (double)i2max * pi2;
				rp2=cos(1.00*t2);
	  			n=i2 + i2max * (i1 + i1max * i0);
	  			signal[n][REAL]=rp0*rp1*rp2;
	  			signal[n][IMAG]=0.0;
#ifdef PRINT
	  		 	printf("+ %4d %4d %4d %4d %10.6f %10.6f %10.6f %10.6f\n",i0,i1,i2,n,t0,t1,t2,signal[n][REAL]);		
#endif
	  		}  		
	  	}
	  }
#endif
}

#define rrand() (float)rand()/(float)(RAND_MAX); 
void randomv(fftw_complex* signal,int N) {
    int i;
    for (i = 0; i < N; ++i) {
        signal[i][REAL] = rrand();
        signal[i][IMAG] = rrand();
    }
}

int main(int argc, char **argv){
     unsigned long NUM_POINTS;
     int N=16;
     double c1,c2,s1,s2,d1,d2;
     if (argc > 1) N=atoi(argv[1]);
#ifdef ONED
     NUM_POINTS=N;
     printf("vector size %d elements %ld\n",N,NUM_POINTS);
#endif
#ifdef TWOD
     NUM_POINTS=N*N;
     printf("grid size %d elements %ld\n",N,NUM_POINTS);
#endif
#ifdef THREED
     NUM_POINTS=N*N*N;
     printf("cube size %d elements %ld\n",N,NUM_POINTS);
#endif
     fftw_complex *signal;
     fftw_complex *result;
     signal=(fftw_complex*)fftw_malloc(NUM_POINTS*sizeof(fftw_complex));
     result=(fftw_complex*)fftw_malloc(NUM_POINTS*sizeof(fftw_complex));

     fftw_plan p;

/*
	FFTW_ESTIMATE specifies that, instead of actual measurements of
different algorithms, a simple heuristic is used to pick a (probably
sub-optimal) plan quickly. With this flag, the input/output arrays
are not overwritten during planning.

	FFTW_MEASURE tells FFTW to find an optimized plan by actually
computing several FFTs and measuring their execution time. Depending
on your machine, this can take some time (often a few seconds).
FFTW_MEASURE is the default planning option.
*/
    printf(" create plan\n");
    c1=mysecond();
#ifdef ONED
    p = fftw_plan_dft_1d(N,     signal,result,FFTW_FORWARD,FFTW_ESTIMATE);
#endif
#ifdef TWOD
    p = fftw_plan_dft_2d(N,N,   signal,result,FFTW_FORWARD,FFTW_ESTIMATE);
#endif
#ifdef THREED
    p = fftw_plan_dft_3d(N,N,N, signal,result,FFTW_FORWARD,FFTW_ESTIMATE);
#endif
    c2=mysecond();
    printf(" stuff in data\n");
    s1=mysecond();
#ifdef TRIG
    trig(signal,N);
#else
    randomv(signal,NUM_POINTS);
#endif
    s2=mysecond();
    if (NUM_POINTS <= PMAX ){
	    printf("signal\n");
        printit(signal,NUM_POINTS,'>');
    }
    printf(" do it\n");
    d1=mysecond();
    fftw_execute(p);
    d2=mysecond();
    if (NUM_POINTS <= PMAX ) {
    	printf("result\n");
    	printit(result,NUM_POINTS,'<');
    }
    printf(" clean up\n");
    fftw_destroy_plan(p);  
#ifdef ONED    
    printf("create 1d plan %15.6f\n",c2-c1);
#endif
#ifdef TWOD
    printf("create 2d plan %15.6f\n",c2-c1);
#endif
#ifdef THREED
    printf("create 3d plan %15.6f\n",c2-c1);
#endif
    printf(" stuff in data %15.6f\n",s2-s1);
    printf("        do fft %15.6f\n",d2-d1);
}









