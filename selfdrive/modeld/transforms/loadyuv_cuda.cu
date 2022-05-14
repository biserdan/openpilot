#include "loadyuv_cuda.cuh"   

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

/*__global__ void loadys(__global__ u_int8_t const * const Y,
                     __global__ float * out,
                     int out_offset, int TRANSFORMED_WIDTH,
                     int TRANSFORMED_HEIGHT, int UV_SIZE)*/
__global__ void loadys(u_int8_t const * const Y,
                     float * out,
                     int out_offset, int TRANSFORMED_WIDTH,
                     int TRANSFORMED_HEIGHT, int UV_SIZE)
{
   const int gid = threadIdx.x;
   const int ois = gid * 8;
   const int oy = ois / TRANSFORMED_WIDTH;
   const int ox = ois % TRANSFORMED_WIDTH;

  const u_int8_t ys[8] = {
    Y[gid],
    Y[gid+1],
    Y[gid+2],
    Y[gid+3],
    Y[gid+4],
    Y[gid+5],
    Y[gid+6],
    Y[gid+7]
  };

  const float_t ysf[8] = {
    static_cast<float_t>(ys[0]),
    static_cast<float_t>(ys[1]),
    static_cast<float_t>(ys[2]),
    static_cast<float_t>(ys[3]),
    static_cast<float_t>(ys[4]),
    static_cast<float_t>(ys[5]),
    static_cast<float_t>(ys[6]),
    static_cast<float_t>(ys[7])
  };

    // 02
    // 13

    // __global__ float* outy0;
    // __global__ float* outy1;
    float *outy0;
    float *outy1;
    if ((oy & 1) == 0) {
      outy0 = out + out_offset; //y0
      outy1 = out + out_offset + UV_SIZE*2; //y2
    } else {
      outy0 = out + out_offset + UV_SIZE; //y1
      outy1 = out + out_offset + UV_SIZE*3; //y3
    }
    
    //vstore4(ysf.s0246, 0, outy0 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
    //vstore4(ysf.s1357, 0, outy1 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
}

/*__global__ void loaduv(__global uchar8 const * const in,
                     __global float8 * out,
                     int out_offset)*/
__global__ void loaduv(uint8_t const * const in,
                     float * out,
                     int out_offset)
{
  const int gid = threadIdx.x;
  const u_int8_t inv[8] = {
    in[gid],
    in[gid+1],
    in[gid+2],
    in[gid+3],
    in[gid+4],
    in[gid+5],
    in[gid+6],
    in[gid+7]
  };

  const float_t outv[8] = {
    static_cast<float_t>(inv[0]),
    static_cast<float_t>(inv[1]),
    static_cast<float_t>(inv[2]),
    static_cast<float_t>(inv[3]),
    static_cast<float_t>(inv[4]),
    static_cast<float_t>(inv[5]),
    static_cast<float_t>(inv[6]),
    static_cast<float_t>(inv[7])
  };

  // toDo
  out[gid + out_offset / 8] = outv[0];
}
/*
__global__ void copy(__global__ float * inout,
                   int in_offset)
{
  const int gid = get_global_id(0);
  inout[gid] = inout[gid + in_offset / 8];
}
*/
void start_loadys(uint8_t *y_cuda_d, float_t *out_cuda, 
    size_t *global_out_off, const int loadys_work_size,
    int TRANSFORMED_WIDTH, int TRANSFORMED_HEIGHT)
{
   int UV_SIZE = ((TRANSFORMED_WIDTH/2)*(TRANSFORMED_HEIGHT/2));
   loadys<<< 1, loadys_work_size >>>(y_cuda_d,out_cuda,static_cast<int>(*global_out_off),TRANSFORMED_WIDTH,TRANSFORMED_HEIGHT,UV_SIZE);

   sleep(1);   // Necessary to give time to let GPU threads run !!!

}

void start_loaduv(uint8_t *u_cuda_d, float_t *out_cuda, 
    size_t *global_out_off, const int loaduv_work_size,
    int TRANSFORMED_WIDTH, int TRANSFORMED_HEIGHT)
{
   int UV_SIZE = ((TRANSFORMED_WIDTH/2)*(TRANSFORMED_HEIGHT/2));
   loaduv<<< 1, loaduv_work_size >>>(u_cuda_d,out_cuda,static_cast<int>(*global_out_off));

   sleep(1);   // Necessary to give time to let GPU threads run !!!

}
