#include "selfdrive/modeld/transforms/test_loadyuv.h"

#include "selfdrive/modeld/transforms/loadyuv_cuda.cuh"

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

void test_loadyuv() {
  printf("test_loadyuv\n");

  // create buffers shared memory host and device
  // uint8_t * test = 0;
  // checkMsg(cudaHostAlloc((void ** ) & test, 131072, cudaHostAllocMapped));
  uint8_t * y_cuda_h, * y_cuda_d;
  checkMsg(cudaHostAlloc((void ** ) & y_cuda_h, 131072, cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & y_cuda_d, (void * ) y_cuda_h, 0));
  uint8_t * v_cuda_h, * v_cuda_d;
  checkMsg(cudaHostAlloc((void ** ) & v_cuda_h, 32768, cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & v_cuda_d, (void * ) v_cuda_h, 0));
  uint8_t * u_cuda_h, * u_cuda_d;
  checkMsg(cudaHostAlloc((void ** ) & u_cuda_h, 32768, cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & u_cuda_d, (void * ) u_cuda_h, 0));
  float * io_buffer_h, * io_buffer_d;
  checkMsg(cudaHostAlloc((void ** ) & io_buffer_h, 196608 * sizeof(float), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void ** ) & io_buffer_d, (void * ) io_buffer_h, 0));

  // y component: read file from OpenCL and fill up input buffer, write data to file
  FILE * openclyf = fopen("test_yuv_opencl_y.txt", "r");
  FILE * inputf = fopen("test_yuvinput_y.txt", "w");
  int x = 0;
  int y = 0;
  fscanf(openclyf, "%d", & x);
  y_cuda_h[y] = x;
  fprintf(inputf, "%d ", y_cuda_h[y]);
  fprintf(inputf, "\n");
  while (!feof(openclyf)) {
    if (y < 131072 - 1) {
      y += 1;
      fscanf(openclyf, "%d", & x);
      y_cuda_h[y] = x;
      fprintf(inputf, "%d ", y_cuda_h[y]);
      if (y % 1000 == 0) {
        fprintf(inputf, "\n");
      }
    } else {
      break;
    }
  }
  fclose(inputf);
  fclose(openclyf);

  // u and v component: read file from OpenCL and fill up the input buffer, write data to file
  FILE * opencluf = fopen("test_yuv_opencl_u.txt", "r");
  FILE * inputuf = fopen("test_yuvinput_u.txt", "w");
  FILE * openclvf = fopen("test_yuv_opencl_v.txt", "r");
  FILE * inputvf = fopen("test_yuvinput_v.txt", "w");
  int xu = 0;
  int xv = 0;
  y = 0;
  fscanf(opencluf, "%d", & xu);
  fscanf(openclvf, "%d", & xv);
  u_cuda_h[y] = xu;
  v_cuda_h[y] = xv;
  fprintf(inputuf, "%d ", u_cuda_h[y]);
  fprintf(inputuf, "\n");
  fprintf(inputvf, "%d ", v_cuda_h[y]);
  fprintf(inputvf, "\n");
  while (!feof(opencluf) && !feof(openclvf)) {
    if (y < 32768 - 1) {
      y += 1;
      fscanf(opencluf, "%d", & xu);
      fscanf(openclvf, "%d", & xv);
      u_cuda_h[y] = xu;
      v_cuda_h[y] = xv;
      fprintf(inputuf, "%d ", u_cuda_h[y]);
      fprintf(inputvf, "%d ", v_cuda_h[y]);
      if (y % 1000 == 0) {
        fprintf(inputuf, "\n");
        fprintf(inputvf, "\n");
      }
    } else {
      break;
    }
  }
  fclose(inputuf);
  fclose(inputvf);
  fclose(opencluf);
  fclose(openclvf);

  // y component: set offset and worksize for kernel
  int global_out_off = 0;
  const int loadys_work_size = (512 * 256);

  // y component: start kernel 
  start_loadys(y_cuda_d, io_buffer_d, global_out_off, loadys_work_size,
    512, 256);

  //printf("Output: %f, %f\n",io_buffer_h[0],io_buffer_h[1]);

  // y component: output data to file
  FILE * output_loadys_f = fopen("test_yuvloadys.txt", "w");
  fprintf(output_loadys_f, "Output ys: \n");
  for (int i = 0; i < 196608; i++) {
    if (i % 100 == 0) {
      fprintf(output_loadys_f, "%f ,", io_buffer_h[i]);
      if (i % 1000 == 0) {
        fprintf(output_loadys_f, "\n");
      }
    }
    //fprintf(output_loadys_f,"%f\n",io_buffer_h[i]);
  }
  fclose(output_loadys_f);

  // u component: set offset and worksize for kernel
  global_out_off += 131072;
  int loaduv_work_size = 256 * 128;

  // u component: start kernel
  start_loaduv(u_cuda_d, io_buffer_d, global_out_off, loaduv_work_size);

  // u component: output data to file
  FILE * output_loadu_f = fopen("test_yuvloadu.txt", "w");
  fprintf(output_loadu_f, "Output u: \n");
  for (int i = 0; i < 196608; i++) {
    if (i % 100 == 0) {
      fprintf(output_loadu_f, "%f ,", io_buffer_h[i]);
      if (i % 1000 == 0) {
        fprintf(output_loadu_f, "\n");
      }
    }
  }
  fclose(output_loadu_f);

  // v component: set offset
  global_out_off += 256 * 128;

  // v component start kernel
  start_loaduv(v_cuda_d, io_buffer_d, global_out_off, loaduv_work_size);

  // v component: output data to file
  FILE * output_loadv_f = fopen("test_yuvloadv.txt", "w");
  fprintf(output_loadv_f, "Output v: \n");
  for (int i = 0; i < 196608; i++) {
    if (i % 100 == 0) {
      fprintf(output_loadv_f, "%f ,", io_buffer_h[i]);
      if (i % 1000 == 0) {
        fprintf(output_loadv_f, "\n");
      }
    }
  }
  fclose(output_loadv_f);

  // set offset and worksize for copy kernel
  global_out_off = 196608;
  int copy_work_size = global_out_off / 8;

  // start copy kernel
  start_copy(io_buffer_d, global_out_off, copy_work_size);

  // copy: output data to file
  FILE * output_copy_f = fopen("test_yuvcopy.txt", "w");
  fprintf(output_copy_f, "Output copy: \n");
  for (int i = 0; i < 196608; i++) {
    fprintf(output_copy_f, "%f ,", io_buffer_h[i]);
    if (i % 8 == 0) {
      fprintf(output_copy_f, "\n");
    }
  }
  fclose(output_copy_f);

  // clean up buffers
  //printf("float_t: %lu\n",sizeof(float_t));
  checkMsg(cudaFreeHost((void * ) y_cuda_h));
  checkMsg(cudaFreeHost((void * ) v_cuda_h));
  checkMsg(cudaFreeHost((void * ) u_cuda_h));
  checkMsg(cudaFreeHost((void * ) io_buffer_h));
  //checkMsg(cudaFreeHost((void * ) test));
  printf("finsih test_loadyuv\n");
}