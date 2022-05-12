#include <stdio.h>
 
#define NUM_BLOCKS 16
#define BLOCK_SIZE 1
 
__global__ void hello()
{
    int idx = blockIdx.x;
    printf("Hello world! I'm a thread in block %d\n", idx);
}
 
 
int main(int argc,char **argv)
{
    // launch the kernel
    hello<<<NUM_BLOCKS, BLOCK_SIZE>>>();
 
    // force the printf()s to flush
    cudaDeviceSynchronize();
 
    printf("That's all!\n");
 
    return 0;
}
