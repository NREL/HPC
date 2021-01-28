#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
void checkCUDAError(const char *msg);
 __global__ void Kernel(int *dat);
extern "C" void  cumain(int myid, int gx, int gy,int bx, int by, int bz) {
	int *dat_local, *dat_remote;
	//int gx,gy;
	//int bx,by,bz;
	int size;
	int numthreads,j;
        int mydev;

	cudaSetDevice(myid);
	cudaGetDevice(&mydev);
	
	dim3 dimGrid(gx,gy);
	dim3 dimBlock(bx,by,bz);
	
	numthreads=gx*gy*bx*by*bz;
	
	size=6*sizeof(int)*numthreads;
	cudaMalloc((void**) &dat_remote, size);
        checkCUDAError("cudaMalloc");
	dat_local=(int*)malloc(size);
	
	Kernel<<<dimGrid,dimBlock>>>(dat_remote);
        checkCUDAError("Kernel");
	cudaMemcpy(dat_local, dat_remote, size,cudaMemcpyDeviceToHost);
        checkCUDAError("copy");
	
	for(int i=0;i<numthreads;i++) {
		j=i*6;
		printf("%4.4d %2.2d %6d      %3d %3d      %3d %3d %3d\n",myid,mydev,
		dat_local[j],
		dat_local[j+1],dat_local[j+2],
		dat_local[j+3],dat_local[j+4],dat_local[j+5]);
	}
}

// To run at normal speed define SLOW as blank.
// The program should return the same results independent of
// the setting for kmax and jmax.
// Set jmax to a larger value to slow it down more.
#ifndef SLOW
#define SLOW slow 
#endif
int __device__ slow(int input){
  int i;
  int jmax,kmax;
  jmax=5;
  kmax=100000;
  for (int j=1; j <=jmax ; j++) {
    i=j;
    if(j == jmax)i=input;
    for (int k=1; k< kmax; k++) {
     i= int(i*(1.00001*(sin((double)i)*sin((double)i)+cos((double)i)*cos((double)i))));
    }
  }
  return(i);
}

 __global__ void Kernel(int *dat) {
/* get my block within a grid */
    int myblock=blockIdx.x+blockIdx.y*gridDim.x;
/* how big is each block within a grid */
    int blocksize=blockDim.x*blockDim.y*blockDim.z;
/* get thread within a block */
    int subthread=threadIdx.z*(blockDim.x*blockDim.y)+threadIdx.y*blockDim.x+threadIdx.x;
/* find my thread */
    int thread=myblock*blocksize+subthread;
#if __DEVICE_EMULATION__
	printf("gridDim=(%3d %3d) blockIdx=(%3d %3d)     blockDim=(%3d %3d %3d)  threadIdx=(%3d %3d %3d)  %6d\n",    
	  gridDim.x,gridDim.y,
	  blockIdx.x,blockIdx.y,
	  blockDim.x,blockDim.y,blockDim.z,
	  threadIdx.x,threadIdx.y,threadIdx.z,thread);
#endif
/* starting index into array */
	int index=thread*6;
	dat[index]=SLOW(thread);
	dat[index+1]=SLOW(blockIdx.x);
	dat[index+2]=SLOW(blockIdx.y);
	dat[index+3]=SLOW(threadIdx.x);
	dat[index+4]=SLOW(threadIdx.y);
	dat[index+5]=SLOW(threadIdx.z);
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg,
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}
