/* ---------------------------------------------------
   My Hello world for CUDA programming
   --------------------------------------------------- */

#include <stdio.h>        // C programming header file
#include <unistd.h>       // C programming header file
#include <cuda_runtime.h>
#include "hello_cuda.cuh"                            // cude.h is automatically included by nvcc...

inline void __checkMsg(cudaError_t code, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", cudaGetErrorString(code), file, line, cudaGetErrorString(err));
    exit(-1);
  }
}
inline void __checkMsgNoFail(cudaError_t code, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "checkMsg() CUDA warning: %s in file <%s>, line %i : %s.\n", cudaGetErrorString(code), file, line, cudaGetErrorString(err));
  }
}

/* ------------------------------------
   Your first kernel (= GPU function)
   ------------------------------------ */
__global__ void hello( )
{
   printf("Hello World GPU!\n");
}

void start_hello()
{

   hello<<< 1, 4 >>>( );

   cudaDeviceProp prop;
   checkMsg(cudaGetDeviceProperties(&prop,0));

   printf("I am the CPU: Hello World ! \n");

   sleep(1);   // Necessary to give time to let GPU threads run !!!

}
