#include <fcntl.h>
#include <getopt.h>
#include <memory.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef ANDROID

#define MAXE 0
#include <unistd.h>

#else
// The libyuv implementation on ARM is slightly different than on x86
// Our implementation matches the ARM version, so accept errors of 1
#define MAXE 1

#endif

// #include <CL/cl.h>

#include "libyuv.h"
#include "selfdrive/camerad/transforms/rgb_to_yuv.h"
// #include "selfdrive/common/clutil.h"
#include "/usr/local/cuda/include/cuda_runtime.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static inline double millis_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec * 1e-6;
}


bool compare_results(uint8_t *a, uint8_t *b, int len, int stride, int width, int height, uint8_t *rgb) {
  int min_diff = 0., max_diff = 0., max_e = 0.;
  int e1 = 0, e0 = 0;
  int e0y = 0, e0u = 0, e0v = 0, e1y = 0, e1u = 0, e1v = 0;
  int max_e_i = 0;
  for (int i = 0;i < len;i++) {
    int e = ((int)a[i]) - ((int)b[i]);
    if(e < min_diff) {
      min_diff = e;
    }
    if(e > max_diff) {
      max_diff = e;
    }
    int e_abs = std::abs(e);
    if(e_abs > max_e) {
      max_e = e_abs;
      max_e_i = i;
    }
    if(e_abs < 1) {
      e0++;
      if(i < stride * height)
        e0y++;
      else if(i < stride * height + stride * height / 4)
        e0u++;
      else
        e0v++;
    } else {
      e1++;
      if(i < stride * height)
        e1y++;
      else if(i < stride * height + stride * height / 4)
        e1u++;
      else
        e1v++;
    }
  }
  //printf("max diff : %d, min diff : %d, e < 1: %d, e >= 1: %d\n", max_diff, min_diff, e0, e1);
  //printf("Y: e < 1: %d, e >= 1: %d, U: e < 1: %d, e >= 1: %d, V: e < 1: %d, e >= 1: %d\n", e0y, e1y, e0u, e1u, e0v, e1v);
  if(max_e <= MAXE) {
    return true;
  }
  int row = max_e_i / stride;
  if(row < height) {
    printf("max error is Y: %d = (libyuv: %u - cl: %u), row: %d, col: %d\n", max_e, a[max_e_i], b[max_e_i], row, max_e_i % stride);
  } else if(row >= height && row < (height + height / 4)) {
    printf("max error is U: %d = %u - %u, row: %d, col: %d\n", max_e, a[max_e_i], b[max_e_i], (row - height) / 2, max_e_i % stride / 2);
  } else {
    printf("max error is V: %d = %u - %u, row: %d, col: %d\n", max_e, a[max_e_i], b[max_e_i], (row - height - height / 4) / 2, max_e_i % stride / 2);
  }
  return false;
}

int main(int argc, char** argv) {
  srand(1337);

  int width = 1164;
  int height = 874;

  int opt = 0;
  while ((opt = getopt(argc, argv, "f")) != -1)
    {
      switch (opt)
        {
        case 'f':
          std::cout << "Using front camera dimensions" << std::endl;
          width = 1152;
          height = 846;
        }
  }

  std::cout << "Width: " << width << " Height: " << height << std::endl;
  uint8_t *rgb_frame = new uint8_t[width * height * 3];

  Rgb2Yuv rgb_to_yuv_state(width,height, width * 3);

  int frame_yuv_buf_size = width * height * 3 / 2;
  // create yuv CUDA device buffer
  void * d_yuv_cuda;
  cudaMalloc((void**)&d_yuv_cuda, frame_yuv_buf_size);

  // cl_mem yuv_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, frame_yuv_buf_size, (void*)NULL, &err));
  uint8_t *frame_yuv_buf = new uint8_t[frame_yuv_buf_size];
  uint8_t *frame_yuv_ptr_y = frame_yuv_buf;
  uint8_t *frame_yuv_ptr_u = frame_yuv_buf + (width * height);
  uint8_t *frame_yuv_ptr_v = frame_yuv_ptr_u + ((width/2) * (height/2));
  
  // create rgb CUDA device buffer
  uint8_t * d_rgb_cuda;
  cudaMalloc((void**)&d_rgb_cuda, width * height * 3);

  // cl_mem rgb_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * 3, (void*)NULL, &err));
  int mismatched = 0;
  int counter = 0;
  srand (time(NULL));

  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < width * height * 3; j++) {
      rgb_frame[j] = (uint8_t)rand();
    }

    double t1 = millis_since_boot();
    libyuv::RGB24ToI420((uint8_t*)rgb_frame, width * 3,
                        frame_yuv_ptr_y, width,
                        frame_yuv_ptr_u, width/2,
                        frame_yuv_ptr_v, width/2,
                        width, height);
    double t2 = millis_since_boot();
    //printf("Libyuv: rgb to yuv: %.2fms\n", t2-t1);

    // copy buffer from host to device
    cudaMemcpy(d_rgb_cuda, (void *)rgb_frame, width * height * 3, cudaMemcpyHostToDevice);

    // clEnqueueWriteBuffer(q, rgb_cl, CL_TRUE, 0, width * height * 3, (void *)rgb_frame, 0, NULL, NULL);
    t1 = millis_since_boot();

    rgb_to_yuv_state.queue(d_rgb_cuda, d_yuv_cuda);
    // rgb_to_yuv_queue(&rgb_to_yuv_state, q, rgb_cl, yuv_cl);
    t2 = millis_since_boot();

    //printf("OpenCL: rgb to yuv: %.2fms\n", t2-t1);
    // uint8_t *yyy = (uint8_t *)clEnqueueMapBuffer(q, yuv_cl, CL_TRUE,
    //                                              CL_MAP_READ, 0, frame_yuv_buf_size,
    //                                              0, NULL, NULL, &err);

    uint8_t *yyy;
    yyy   = (uint8_t *)malloc(frame_yuv_buf_size);
    // copy CUDA buffer from device to host
    gpuErrchk(cudaMemcpy(yyy, d_yuv_cuda, frame_yuv_buf_size, cudaMemcpyDeviceToHost)); 

    // gpuErrchk(cudaHostAlloc((void **)&yyy, frame_yuv_buf_size, cudaHostAllocMapped));
    // gpuErrchk(cudaHostGetDevicePointer((void **)&d_yuv_cuda, (void *)yyy, 0));


    if(!compare_results(frame_yuv_ptr_y, (uint8_t *)yyy, frame_yuv_buf_size, width, width, height, (uint8_t*)rgb_frame))
      mismatched++;
    // clEnqueueUnmapMemObject(q, yuv_cl, yyy, 0, NULL, NULL);

    // std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if(counter++ % 100 == 0)
      printf("Matched: %d, Mismatched: %d\n", counter - mismatched, mismatched);

  }
  printf("Matched: %d, Mismatched: %d\n", counter - mismatched, mismatched);

  delete[] frame_yuv_buf;
  // rgb_to_yuv_destroy(&rgb_to_yuv_state);
  // clReleaseContext(context);
  delete[] rgb_frame;

  if (mismatched == 0)
    return 0;
  else
    return -1;
}
