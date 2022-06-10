#include "selfdrive/modeld/transforms/test_transform.h"

#include "selfdrive/modeld/transforms/transform_cuda.cuh"

#include <cassert>

#include <cstring>

#include <time.h>

#include <iostream>

#include <fstream>

#include <stdlib.h>

using namespace std;

inline void __checkMsg(cudaError_t code,
  const char * file,
    const int line) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", cudaGetErrorString(code), file, line, cudaGetErrorString(err));
    exit(-1);
  }
}
inline void __checkMsgNoFail(cudaError_t code,
  const char * file,
    const int line) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "checkMsg() CUDA warning: %s in file <%s>, line %i : %s.\n", cudaGetErrorString(code), file, line, cudaGetErrorString(err));
  }
}

void test_transform() {
  printf("test_transform\n");

  // create buffer on host memory
  // uint8_t *input = static_cast<uint8_t*>(malloc(1928*1208*3/2));
  uint8_t * input = 0;
  checkMsg(cudaHostAlloc((void ** ) & input, 1928 * 1208 * 3 / 2, cudaHostAllocMapped));
  /*uint8_t *data = 0; 
  checkMsg(cudaHostAlloc((void**)&data, 1928*1208*3/2 * sizeof(uint8_t), cudaHostAllocMapped));
  uint8_t *output_y = 0; 
  checkMsg(cudaHostAlloc((void**)&output_y, 131072, cudaHostAllocMapped));
  uint8_t *output_u = 0; 
  checkMsg(cudaHostAlloc((void**)&output_y, 32768, cudaHostAllocMapped));
  uint8_t *output_v = 0; 
  checkMsg(cudaHostAlloc((void**)&output_v, 32768, cudaHostAllocMapped));
  float_t * projection_y_cpu = 0;
  checkMsg(cudaHostAlloc((void ** ) & projection_y_cpu, 3 * 3 * sizeof(float_t), cudaHostAllocMapped));
  float_t * projection_uv_cpu = 0;
  checkMsg(cudaHostAlloc((void ** ) & projection_uv_cpu, 3 * 3 * sizeof(float_t), cudaHostAllocMapped));*/
  
  // read file from OpenCL and fill up input buffer, write data to file
  FILE * openclf = fopen("test_opencl.txt", "r");
  FILE * inputf = fopen("test_input.txt", "w");
  int x = 0;
  int y = 0;

  fscanf(openclf, "%d", & x);
  input[y] = x;
  fprintf(inputf, "%d ", input[y]);
  fprintf(inputf, "\n");
  while (!feof(openclf)) {
    if (y < 1928 * 1208 * 3 / 2 - 1) {
      y += 1;
      fscanf(openclf, "%d", & x);
      input[y] = x;
      fprintf(inputf, "%d ", input[y]);
      if (y % 1000 == 0) {
        fprintf(inputf, "\n");
      }
    } else {
      break;
    }
  }
  fclose(inputf);
  fclose(openclf);

  // define parameters for kernel
  uint8_t * in_yuv_test_h;
  uint8_t * in_yuv_test_d;
  uint8_t * out_y_h;
  uint8_t * out_y_d;
  uint8_t * out_u_h;
  uint8_t * out_u_d;
  uint8_t * out_v_h;
  uint8_t * out_v_d;
  float_t * m_y_cuda_h;
  float_t * m_y_cuda_d;
  float_t * m_uv_cuda_h;
  float_t * m_uv_cuda_d;

  // create buffers shared memory host and device
  //printf("cuda malloc\n");
  checkMsg(cudaHostAlloc((void ** ) & in_yuv_test_h, 1928 * 1208 * 3 / 2, cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & in_yuv_test_d, (void * ) in_yuv_test_h, 0));
  //checkMsg(cudaMalloc((void**)&in_yuv_test_d, 1928*1208*3/2 * sizeof(uint8_t)));

  checkMsg(cudaHostAlloc((void ** ) & out_y_h, 131072, cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & out_y_d, (void * ) out_y_h, 0));
  //checkMsg(cudaMalloc((void**)&out_y_d, 131072 * sizeof(uint8_t)));
  checkMsg(cudaHostAlloc((void ** ) & out_u_h, 32768, cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & out_u_d, (void * ) out_u_h, 0));
  checkMsg(cudaHostAlloc((void ** ) & out_v_h, 32768, cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & out_v_d, (void * ) out_v_h, 0));

  checkMsg(cudaHostAlloc((void ** ) & m_y_cuda_h, 3 * 3 * sizeof(float), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & m_y_cuda_d, (void * ) m_y_cuda_h, 0));
  //checkMsg(cudaMalloc((void ** ) & m_y_cuda_d, 3 * 3 * sizeof(float_t)));

  checkMsg(cudaHostAlloc((void ** ) & m_uv_cuda_h, 3 * 3 * sizeof(float), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & m_uv_cuda_d, (void * ) m_uv_cuda_h, 0));
  //checkMsg(cudaMalloc((void ** ) & m_uv_cuda_d, 3 * 3 * sizeof(float_t)));

  const int zero = 0;

  //mat3 projection_y, projection_uv;
  //printf("projection\n");

  // fill up projection matrix for y and uv with data
  for (int i = 0; i < 10; i++) {
    m_y_cuda_h[i] = 1.0 + i;
    m_uv_cuda_h[i] = 0.5 + i;
  }
  /*for(int i=0; i<10; i++) {
      printf("projection_y: %d\t%f\n",i,*(projection_y_cpu+i));
      printf("projection_uv: %d\t%f\n",i,*(projection_uv_cpu+i));
  }*/
  //printf("cudaMemcpy\n");
  // copy data from host to device
  checkMsg(cudaMemcpy((void * ) in_yuv_test_d, (void * ) input, 1928 * 1208 * 3 / 2, cudaMemcpyHostToDevice));
  // checkMsg(cudaMemcpy((void *)data,(void*)in_yuv_test_d,1928*1208*3/2,cudaMemcpyDeviceToHost));
  //checkMsg(cudaMemcpy((void * ) m_y_cuda_d, (void * ) projection_y_cpu, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
  //checkMsg(cudaMemcpy((void * ) m_uv_cuda_d, (void * ) projection_uv_cpu, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

  // ouput file with data in shared buffer
  FILE * dataf = fopen("test_data.txt", "w");
  fprintf(dataf, "Data: \n");
  //dataf << "Data: \n";
  for (int i = 0; i < 1928 * 1208 * 3 / 2; i++) {
    ;
    if (i % 100 == 0) {
      fprintf(dataf, "%d ,", in_yuv_test_h[i]);
      //dataf << input[i] << ", ";
      if (i % 1000 == 0) {
        fprintf(dataf, "\n");
      }
    }
    //printf("Data: %d\n",input[i]);
  }
  fclose(dataf);

  // initial parameters with fixed values
  const int in_y_width = 1928;
  const int in_y_height = 1208;
  const int in_uv_width = 1928 / 2;
  const int in_uv_height = 1208 / 2;
  const int in_y_offset = 0;
  const int in_u_offset = in_y_offset + in_y_width * in_y_height;
  const int in_v_offset = in_u_offset + in_uv_width * in_uv_height;

  const int out_y_width = 512;
  const int out_y_height = 256;
  const int out_uv_width = 512 / 2;
  const int out_uv_height = 256 / 2;

  //printf("Process test_input\n");

  // y component: initial two dimensional work size
  const size_t work_size_y[2] = {
    (size_t) out_y_width,
    (size_t) out_y_height
  };

  // y component: start kernel
  start_warpPerspective(in_yuv_test_d, in_y_width, in_y_offset, in_y_height, in_y_width,
    out_y_d, out_y_width, zero, out_y_height, out_y_width, m_y_cuda_d,
    (const size_t * ) & work_size_y);
  //printf("finish y\n");
  //checkMsg(cudaMemcpy((void *)output_y,(void*)out_y_d,131072,cudaMemcpyDeviceToHost));

  // y component: output data to file
  FILE * outputfy = fopen("test_output_y.txt", "w");
  fprintf(outputfy, "output_y: \n");
  //dataf << "Data: \n";
  for (int i = 0; i < 131072; i++) {
    ;
    if (i % 100 == 0) {
      fprintf(outputfy, "%d ,", out_y_h[i]);
      //dataf << input[i] << ", ";
      if (i % 1000 == 0) {
        fprintf(outputfy, "\n");
      }
    }
    //printf("Data: %d\n",input[i]);
  }
  fclose(outputfy);

  // u and v component: define two dimensional work_size
  const size_t work_size_uv[2] = {
    (size_t) out_uv_width,
    (size_t) out_uv_height
  };

  // u component: start kernel
  start_warpPerspective(in_yuv_test_d, in_uv_width, in_u_offset, in_uv_height, in_uv_width,
    out_u_d, out_uv_width, zero, out_uv_height, out_uv_width, m_uv_cuda_d,
    (const size_t * ) & work_size_uv);
  //printf("finish u\n");

  // u component: output data to file
  FILE * outputfu = fopen("test_output_u.txt", "w");
  fprintf(outputfu, "output_u: \n");
  //dataf << "Data: \n";
  for (int i = 0; i < 32768; i++) {
    ;
    if (i % 100 == 0) {
      fprintf(outputfu, "%d ,", out_u_h[i]);
      //dataf << input[i] << ", ";
      if (i % 1000 == 0) {
        fprintf(outputfu, "\n");
      }
    }
    //printf("Data: %d\n",input[i]);
  }
  fclose(outputfu);

  // v component: start kernel
  start_warpPerspective(in_yuv_test_d, in_uv_width, in_v_offset, in_uv_height, in_uv_width,
    out_v_d, out_uv_width, zero, out_uv_height, out_uv_width, m_uv_cuda_d,
    (const size_t * ) & work_size_uv);
  //printf("finish v\n");

  // v component: output data to file
  FILE * outputfv = fopen("test_output_v.txt", "w");
  fprintf(outputfv, "output_u: \n");
  //dataf << "Data: \n";
  for (int i = 0; i < 32768; i++) {
    ;
    if (i % 100 == 0) {
      fprintf(outputfv, "%d ,", out_v_h[i]);
      //dataf << input[i] << ", ";
      if (i % 1000 == 0) {
        fprintf(outputfv, "\n");
      }
    }
    //printf("Data: %d\n",input[i]);
  }
  fclose(outputfv);

  // should not be needed since no other kernels are active
  //cudaDeviceSynchronize();

  // clean up buffers
  //checkMsg(cudaFreeHost((void *)m_y_cuda_h));
  checkMsg(cudaFreeHost((void * ) m_y_cuda_h));
  checkMsg(cudaFreeHost((void * ) m_uv_cuda_h));
  //checkMsg(cudaFree((void * ) m_uv_cuda_d));
  checkMsg(cudaFreeHost((void * ) input));
  //checkMsg(cudaFreeHost((void *)data));
  checkMsg(cudaFreeHost((void * ) in_yuv_test_h));
  //checkMsgNoFail(cudaFreeHost((void * ) projection_uv_cpu));
  //checkMsgNoFail(cudaFreeHost((void * ) projection_y_cpu));
  //checkMsg(cudaFree((void *)in_yuv_test_d));
  //checkMsg(cudaFree((void *)out_y_d));
  checkMsg(cudaFreeHost((void * ) out_y_h));
  checkMsg(cudaFreeHost((void * ) out_u_h));
  checkMsg(cudaFreeHost((void * ) out_v_h));

  printf("test_transform finished\n");
}