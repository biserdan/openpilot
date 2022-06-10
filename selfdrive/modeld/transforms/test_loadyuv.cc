#include "selfdrive/modeld/transforms/test_loadyuv.h"

#include <cassert>

#include <cstring>

#include <time.h>

#include <iostream>

#include <fstream>

#include <stdlib.h>

using namespace std;

#include "selfdrive/common/clutil.h"

void test_loadyuv() {

  printf("test_loadyuv\n");

  // create buffers on host memory
  u_char * inputy = (u_char * ) malloc(131072);
  u_char * inputu = (u_char * ) malloc(32768);
  u_char * inputv = (u_char * ) malloc(32768);
  float * output = (float * ) malloc(196608 * sizeof(float));
  
  // create OpenCL environment
  const cl_queue_properties props[] = {
    0
  }; //CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0};

  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, & device_id, NULL, NULL, & err));

  int error;
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, props, & error);
  if (error != 0) {
    printf("clCreateCommandQueueWithProperties error: %d\n", error);
  }

  char args[1024];
  snprintf(args, sizeof(args),
    "-cl-fast-relaxed-math -cl-denorms-are-zero "
    "-DTRANSFORMED_WIDTH=%d -DTRANSFORMED_HEIGHT=%d",
    512, 256);
  
  // load OpenCL kernels from external file and with variable based defines
  cl_program prg = cl_program_from_file(context, device_id, "transforms/loadyuv.cl", args);
  cl_kernel loadys_krnl = CL_CHECK_ERR(clCreateKernel(prg, "loadys", & err));
  cl_kernel loaduv_krnl = CL_CHECK_ERR(clCreateKernel(prg, "loaduv", & err));
  cl_kernel copy_krnl = CL_CHECK_ERR(clCreateKernel(prg, "copy", & err));;
  CL_CHECK(clReleaseProgram(prg));

  // create OpenCL buffers on device memory
  cl_mem y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 131072, NULL, & err));
  cl_mem u_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 32768, NULL, & err));
  cl_mem v_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 32768, NULL, & err));
  cl_mem io_buffer = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 196608 * sizeof(float), NULL, & err));

  // y component: write random test data to host buffer and file
  FILE * inputfy = fopen("test_yuvinput_y.txt", "w");
  //fprintf(inputfy,"Input: \n");
  for (int i = 0; i < 131072; i++) {
    inputy[i] = rand() % 256;
    fprintf(inputfy, "%d ", inputy[i]);
    if (i % 1000 == 0) {
      fprintf(inputfy, "\n");
    }
  }
  fclose(inputfy);

  // copy buffers from host to device memory
  CL_CHECK(clEnqueueWriteBuffer(queue, y_cl, CL_TRUE, 0, 131072, (void * ) inputy, 0, NULL, NULL));
  
  // u and v component: write random test data to host buffer and file
  FILE * inputfu = fopen("test_yuvinput_u.txt", "w");
  FILE * inputfv = fopen("test_yuvinput_v.txt", "w");
  // fprintf(inputfu,"Input: \n");
  // fprintf(inputfv,"Input: \n");
  for (int i = 0; i < 32768; i++) {
    //inputuv[i] = (i % 256);
    inputu[i] = rand() % 256;
    inputv[i] = rand() % 256;
    fprintf(inputfu, "%d ", inputu[i]);
    fprintf(inputfv, "%d ", inputv[i]);
    if (i % 1000 == 0) {
      fprintf(inputfu, "\n");
      fprintf(inputfv, "\n");
    }
  }
  fclose(inputfu);
  fclose(inputfv);
  
  // y component: set offset for kernel
  cl_int global_out_off = 0;
  
  // copy buffers from host to device memory
  CL_CHECK(clEnqueueWriteBuffer(queue, u_cl, CL_TRUE, 0, 32768, (void * ) inputu, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(queue, v_cl, CL_TRUE, 0, 32768, (void * ) inputv, 0, NULL, NULL));

  // y component: set kernel arguments
  CL_CHECK(clSetKernelArg(loadys_krnl, 0, sizeof(cl_mem), & y_cl));
  CL_CHECK(clSetKernelArg(loadys_krnl, 1, sizeof(cl_mem), & io_buffer));
  CL_CHECK(clSetKernelArg(loadys_krnl, 2, sizeof(cl_int), & global_out_off));
  
  // y component: set kernel one dimensional work_size buffer / patches
  const size_t loadys_work_size = 131072 / 8;

  // y component: start kernel 
  CL_CHECK(clEnqueueNDRangeKernel(queue, loadys_krnl, 1, NULL, &
    loadys_work_size, NULL, 0, 0, NULL));
  
  // y component: copy buffer from device to host
  CL_CHECK(clEnqueueReadBuffer(queue, io_buffer, CL_TRUE, 0, 196608 * sizeof(float), (void * ) output, 0, NULL, NULL));
  
  // y component: output data to file
  FILE * output_loadys_f = fopen("test_yuvloadys.txt", "w");
  fprintf(output_loadys_f, "Output ys: \n");
  for (int i = 0; i < 196608; i++) {
    //for(int i=0; i<100; i++) {
    if (i % 100 == 0) {
      fprintf(output_loadys_f, "%f ,", output[i]);
      if (i % 1000 == 0) {
        fprintf(output_loadys_f, "\n");
      }
    }
    //fprintf(output_loadys_f,"%f\n",output[i]);
  }
  fclose(output_loadys_f);
  
  // u and v component: set kernel one dimensional work_size buffer / patches
  const size_t loaduv_work_size = ((512 / 2) * (256 / 2)) / 8;
  
  // u component: set offset to half of the buffer
  global_out_off += 131072;
  
  // u component: set kernel arguments
  CL_CHECK(clSetKernelArg(loaduv_krnl, 0, sizeof(cl_mem), & u_cl));
  CL_CHECK(clSetKernelArg(loaduv_krnl, 1, sizeof(cl_mem), & io_buffer));
  CL_CHECK(clSetKernelArg(loaduv_krnl, 2, sizeof(cl_int), & global_out_off));

  // u component: start kernel
  CL_CHECK(clEnqueueNDRangeKernel(queue, loaduv_krnl, 1, NULL, &
    loaduv_work_size, NULL, 0, 0, NULL));
  
  // u component: copy buffer from device to host
  CL_CHECK(clEnqueueReadBuffer(queue, io_buffer, CL_TRUE, 0, 196608 * sizeof(float), (void * ) output, 0, NULL, NULL));
  
  // u component: output data to file
  FILE * output_loadu_f = fopen("test_yuvloadu.txt", "w");
  fprintf(output_loadu_f, "Output u: \n");
  for (int i = 0; i < 196608; i++) {
    if (i % 100 == 0) {
      fprintf(output_loadu_f, "%f ,", output[i]);
      if (i % 1000 == 0) {
        fprintf(output_loadu_f, "\n");
      }
    }
  }
  fclose(output_loadu_f);
  
  // v component: set offset to 3/4 of the buffer
  global_out_off += 256 * 128;

  CL_CHECK(clSetKernelArg(loaduv_krnl, 0, sizeof(cl_mem), & v_cl));
  CL_CHECK(clSetKernelArg(loaduv_krnl, 1, sizeof(cl_mem), & io_buffer));
  CL_CHECK(clSetKernelArg(loaduv_krnl, 2, sizeof(cl_int), & global_out_off));
  
  // v component: start kernel
  CL_CHECK(clEnqueueNDRangeKernel(queue, loaduv_krnl, 1, NULL, &
    loaduv_work_size, NULL, 0, 0, NULL));
  
  // v component: copy buffer from device to host
  CL_CHECK(clEnqueueReadBuffer(queue, io_buffer, CL_TRUE, 0, 196608 * sizeof(float), (void * ) output, 0, NULL, NULL));
  
  // v component: output data to file
  FILE * output_loadv_f = fopen("test_yuvloadv.txt", "w");
  fprintf(output_loadv_f, "Output v: \n");
  for (int i = 0; i < 196608; i++) {
    if (i % 100 == 0) {
      fprintf(output_loadv_f, "%f ,", output[i]);
      if (i % 1000 == 0) {
        fprintf(output_loadv_f, "\n");
      }
    }
  }
  fclose(output_loadv_f);
  
  // copy: set offset to start of second image
  global_out_off = 196608;
  
  // copy: set kernel arguments
  CL_CHECK(clSetKernelArg(copy_krnl, 0, sizeof(cl_mem), & io_buffer));
  CL_CHECK(clSetKernelArg(copy_krnl, 1, sizeof(cl_int), & global_out_off));
  
  // copy: set one dimensional work_size buffer / patches
  const size_t copy_work_size = global_out_off / 8;
  //const size_t copy_work_size = 1000;
  CL_CHECK(clEnqueueNDRangeKernel(queue, copy_krnl, 1, NULL, & copy_work_size, NULL, 0, 0, NULL));
  
  // copy: copy buffer from device to host
  CL_CHECK(clEnqueueReadBuffer(queue, io_buffer, CL_TRUE, 0, 196608 * sizeof(float), (void * ) output, 0, NULL, NULL));
  
  // copy: output data to file
  FILE * output_copy_f = fopen("test_yuvcopy.txt", "w");
  fprintf(output_copy_f, "Output copy: \n");
  for (int i = 0; i < 196608; i++) {
    if (i % 100 == 0) {
      fprintf(output_copy_f, "%f ,", output[i]);
      if (i % 1000 == 0) {
        fprintf(output_copy_f, "\n");
      }
    }
  }
  fclose(output_copy_f);
  
  // clean up host buffers
  free(output);
  free(inputy);
  free(inputu);
  free(inputv);

  // clean up OpenCL buffers and environment
  CL_CHECK(clReleaseKernel(loadys_krnl));
  CL_CHECK(clReleaseKernel(loaduv_krnl));
  CL_CHECK(clReleaseKernel(copy_krnl));

  CL_CHECK(clReleaseMemObject(io_buffer));
  CL_CHECK(clReleaseMemObject(y_cl));
  CL_CHECK(clReleaseMemObject(u_cl));
  CL_CHECK(clReleaseMemObject(v_cl));

  printf("test_loadyuv finished\n");
}