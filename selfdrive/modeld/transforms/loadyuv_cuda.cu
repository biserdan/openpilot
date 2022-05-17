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
   const int gid = threadIdx.x * blockDim.x * threadIdx.x;
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
    
    // copy vector 0246 (even indexes)
    //checkMsg(cudaMemcpy((void *)outy0 + 0 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2,(void*)&ysf[0],sizeof(float),cudaMemcpyDeviceToDevice));
    //checkMsg(cudaMemcpy((void *)outy0 + 1 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2,(void*)&ysf[2],sizeof(float),cudaMemcpyDeviceToDevice));
    //checkMsg(cudaMemcpy((void *)outy0 + 2 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2,(void*)&ysf[4],sizeof(float),cudaMemcpyDeviceToDevice));
    //checkMsg(cudaMemcpy((void *)outy0 + 3 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2,(void*)&ysf[5],sizeof(float),cudaMemcpyDeviceToDevice));
    //vstore4(ysf.s0246, 0, outy0 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
    *(outy0 + 0 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2) = ysf[0];
    *(outy0 + 1 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2) = ysf[2];
    *(outy0 + 2 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2) = ysf[4];
    *(outy0 + 3 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2) = ysf[6];
    // copy vector 1357 (odd indexes)
    // checkMsg(cudaMemcpy((void *)outy1 + 0 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2,(void*)&ysf[1],sizeof(float),cudaMemcpyDeviceToDevice));
    // checkMsg(cudaMemcpy((void *)outy1 + 1 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2,(void*)&ysf[3],sizeof(float),cudaMemcpyDeviceToDevice));
    // checkMsg(cudaMemcpy((void *)outy1 + 2 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2,(void*)&ysf[5],sizeof(float),cudaMemcpyDeviceToDevice));
    // checkMsg(cudaMemcpy((void *)outy1 + 3 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2,(void*)&ysf[7],sizeof(float),cudaMemcpyDeviceToDevice));
    //vstore4(ysf.s1357, 0, outy1 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2);
    *(outy1 + 0 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2) = ysf[1];
    *(outy1 + 1 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2) = ysf[3];
    *(outy1 + 2 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2) = ysf[5];
    *(outy1 + 3 + (oy/2) * (TRANSFORMED_WIDTH/2) + ox/2) = ysf[7];
}

/*__global__ void loaduv(__global uchar8 const * const in,
                     __global float8 * out,
                     int out_offset)*/
__global__ void loaduv(uint8_t const * const in,
                     float * out,
                     int out_offset)
{
  const int gid = threadIdx.x * blockDim.x * threadIdx.x;
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
  out[gid + 0 + out_offset / 8] = outv[0];
  out[gid + 1 + out_offset / 8] = outv[0];
  out[gid + 2 + out_offset / 8] = outv[0];
  out[gid + 3 + out_offset / 8] = outv[0];
  out[gid + 4 + out_offset / 8] = outv[0];
  out[gid + 5 + out_offset / 8] = outv[0];
  out[gid + 6 + out_offset / 8] = outv[0];
  out[gid + 7 + out_offset / 8] = outv[0];
}

__global__ void copy(float * inout,
                   int in_offset)
{
  const int gid = threadIdx.x * blockDim.x * threadIdx.x;
  inout[gid + 0] = inout[gid + 0 + in_offset / 8];
  inout[gid + 1] = inout[gid + 1 + in_offset / 8];
  inout[gid + 2] = inout[gid + 2 + in_offset / 8];
  inout[gid + 3] = inout[gid + 3 + in_offset / 8];
  inout[gid + 4] = inout[gid + 4 + in_offset / 8];
  inout[gid + 5] = inout[gid + 5 + in_offset / 8];
  inout[gid + 6] = inout[gid + 6 + in_offset / 8];
  inout[gid + 7] = inout[gid + 7 + in_offset / 8];
}

void start_loadys(uint8_t *y_cuda_d, float_t *out_cuda, 
    size_t *global_out_off, const int loadys_work_size,
    int TRANSFORMED_WIDTH, int TRANSFORMED_HEIGHT)
{
   int UV_SIZE = ((TRANSFORMED_WIDTH/2)*(TRANSFORMED_HEIGHT/2));
   loadys<<< loadys_work_size, 1>>>(y_cuda_d,out_cuda,static_cast<int>(*global_out_off),TRANSFORMED_WIDTH,TRANSFORMED_HEIGHT,UV_SIZE);
   sleep(1);   // Necessary to give time to let GPU threads run !!!
}

void start_loaduv(uint8_t *u_cuda_d, float_t *out_cuda, 
    size_t *global_out_off, const int loaduv_work_size)
{
   loaduv<<< loaduv_work_size, 1>>>(u_cuda_d,out_cuda,static_cast<int>(*global_out_off));
   sleep(1);   // Necessary to give time to let GPU threads run !!!
}

void start_copy(float_t *inout, 
    size_t *in_offset, const int copy_work_size)
{
  
  copy<<< copy_work_size, 1>>>(inout,static_cast<int>(*in_offset));
  sleep(1);   // Necessary to give time to let GPU threads run !!!
}