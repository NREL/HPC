#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <unistd.h>
#include <sys/time.h>
#include <sched.h>
#include <string>
#include <vector>



// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
        cudaError_t cuErr = call;                                                             \
        if(cudaSuccess != cuErr){                                                             \
                printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
                exit(0);                                                                            \
        }                                                                                     \
}while(0)


void FORCECORE (int *core) {
        int bonk;
        cpu_set_t set; 
        bonk=*core;
        bonk=abs(bonk) ;
        CPU_ZERO(&set);        // clear cpu mask
        CPU_SET(bonk, &set);      // set cpu 0
        if (*core < 0 ){
                sched_setaffinity(0, sizeof(cpu_set_t), &set);
        }else{
                sched_setaffinity(getpid(), sizeof(cpu_set_t), &set);
        }
} 

double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


int main(int argc, char *argv[]) {
double t1,t2,rate;
 // read core,gpu,N,count;
 // buffer is double so divide N by sizeof(double)
int core,gpu,count;
long N;
 
	if(argc != 5){
		printf("%s CORE GPU BYTES REPEAT_COUNT\n",argv[0]);
		printf("For example:\n");
		printf("%s 23 3 10000000000 5\n",argv[0]);
		exit(1);
	}
	core= std::atoi(argv[1]);
	gpu=  std::atoi(argv[2]);
	N= std::atol(argv[3]);
	N=N/sizeof(double);
	if ( N == 0 ) N=1;
	count= std::atoi(argv[4]);
	FORCECORE(&core);
	cudaSetDevice(gpu);
	// Allocate memory for A on CPU
	double *A = (double*)malloc(N*sizeof(double));
	double *d_A;
	// Initialize all elements of A to random values
	for(int i=0; i<N; i++){
		// A[i] = (double)rand()/(double)RAND_MAX;
		A[i] = (double)i/(double)RAND_MAX;
	}
	
	cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
	t1=mysecond();
	for (int i=0; i<count; i++) {
		cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
	}
	t2=mysecond();
	N=N*sizeof(double);
	rate=(count*N)/(t2-t1);
	printf("      core        gpu                bytes         bytes/sec\n");
	printf("%10d %10d %20ld        %10.4g\n",core,gpu,N,rate);

}
