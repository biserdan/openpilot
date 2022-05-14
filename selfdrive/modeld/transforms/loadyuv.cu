#include "loadyuv.cuh"   

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

__global__ void loadys(__global__ u_int8_t const * const Y,
                     __global__ float * out,
                     int out_offset, int TRANSFORMED_WIDTH,
                     int TRANSFORMED_HEIGHT, int UV_SIZE)
{
   const int gid = threadIdx.x;
   const int ois = gid * 8;
   const int oy = ois / TRANSFORMED_WIDTH;
   const int ox = ois % TRANSFORMED_WIDTH;

   const uchar8 ys = Y[gid];
   const float8 ysf = convert_float8(ys);

    // 02
    // 13

    __global float* outy0;
    __global float* outy1;
    if ((oy & 1) == 0) {
      outy0 = out + out_offset; //y0
      outy1 = out + out_offset + UV_SIZE*2; //y2
    } else {
      outy0 = out + out_offset + UV_SIZE; //y1
      outy1 = out + out_offset + UV_SIZE*3; //y3
    }

    vstore4(ysf.s0246, 0, outy0 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
    vstore4(ysf.s1357, 0, outy1 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
}

__global__ void loaduv(__global uchar8 const * const in,
                     __global float8 * out,
                     int out_offset)
{
  const int gid = get_global_id(0);
  const uchar8 inv = in[gid];
  const float8 outv  = convert_float8(inv);
  out[gid + out_offset / 8] = outv;
}

__global__ void copy(__global__ float * inout,
                   int in_offset)
{
  const int gid = get_global_id(0);
  inout[gid] = inout[gid + in_offset / 8];
}

void start_loadys(uint16_t *y_cuda_d, uint16_t *out_cuda, 
    size_t *global_out_off, const int loadys_work_size,
    int TRANSFORMED_WIDTH, int TRANSFORMED_HEIGHT)
{
   int UV_SIZE = ((TRANSFORMED_WIDTH/2)*(TRANSFORMED_HEIGHT/2));
   loadys<<< 1, loadys_work_size >>>(&y_cuda_d,&out_cuda,static_cast<int>(*global_out_off),TRANSFORMED_WIDTH,TRANSFORMED_HEIGHT,UV_SIZE);

   sleep(1);   // Necessary to give time to let GPU threads run !!!

}
