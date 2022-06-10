#include "selfdrive/modeld/transforms/test_transform.h"

#include <cassert>

#include <cstring>

#include <time.h>

#include <iostream>

#include <fstream>

#include <stdlib.h>

using namespace std;

#include "selfdrive/common/clutil.h"

void test_transform() {

  printf("test_transform\n");

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

  // load OpenCL kernel from external file
  cl_program prg = cl_program_from_file(context, device_id, "transforms/transform.cl", "");
  cl_kernel krnl = CL_CHECK_ERR(clCreateKernel(prg, "warpPerspective", & err));
  CL_CHECK(clReleaseProgram(prg));

  // create buffers on host memory
  u_char * input = (u_char * ) malloc(1928 * 1208 * 3 / 2);
  u_char * data = (u_char * ) malloc(1928 * 1208 * 3 / 2);
  u_char * output_y = (u_char * ) malloc(131072);
  u_char * output_u = (u_char * ) malloc(32768);
  u_char * output_v = (u_char * ) malloc(32768);

  // old output file
  /*input[0] = 0;
  input[1] = 255;
  input[2] = 10;
  printf("1: %d, 2: %d, 3: %d\n",input[0],input[1],input[2]);*/
  /*ofstream dataf ("test_data.txt");
  if (!dataf.is_open())
  {
      cout << "Unable to open file\n";        
  }*/

  // write random test data to host buffer and file
  FILE * inputf = fopen("test_input.txt", "w");
  //fprintf(inputfy,"Input: \n");
  for (int i = 0; i < 1928 * 1208 * 3 / 2; i++) {
    input[i] = rand() % 256;
    fprintf(inputf, "%d ", input[i]);
    if (i % 1000 == 0) {
      fprintf(inputf, "\n");
    }
  }
  fclose(inputf);

  // create OpenCL buffers on device memory
  //printf("size_of: %lu\n",sizeof(input));
  cl_mem m_y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * 3 * sizeof(float), NULL, & err));
  cl_mem m_uv_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * 3 * sizeof(float), NULL, & err));
  cl_mem in_yuv_test = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 1928 * 1208 * 3 / 2, NULL, & err));
  cl_mem out_y = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 131072, NULL, & err));
  cl_mem out_u = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 32768, NULL, & err));
  cl_mem out_v = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 32768, NULL, & err));

  const int zero = 0;

  mat3 projection_y, projection_uv;

  // fill projection matrix for y and yv with data
  for (int i = 0; i < 10; i++) {
    projection_y.v[i] = 1 + i;
    projection_uv.v[i] = 0.5 + i;
  }

  // copy buffers from host to device memory
  CL_CHECK(clEnqueueWriteBuffer(queue, m_y_cl, CL_TRUE, 0, 3 * 3 * sizeof(float), (void * ) projection_y.v, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(queue, m_uv_cl, CL_TRUE, 0, 3 * 3 * sizeof(float), (void * ) projection_uv.v, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(queue, in_yuv_test, CL_TRUE, 0, 1928 * 1208 * 3 / 2, (void * ) input, 0, NULL, NULL));

  // debugging: copy buffer from device to host memory 
  CL_CHECK(clEnqueueReadBuffer(queue, in_yuv_test, CL_TRUE, 0, 1928 * 1208 * 3 / 2, (void * ) data, 0, NULL, NULL));

  // debugging to check projection matrices
  /*for(int i=0; i<10; i++) {
      printf("projection_y: %d\t%f\n",i,*((float *)m_y_cl+i));
      //printf("projection_uv: %d\t%f\n",i,*(projection_uv_cpu+i));
  }*/

  // initialize parameters with fixed values
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

  // check if buffer copies were successful with file 
  FILE * dataf = fopen("test_data.txt", "w");
  for (int i = 0; i < 1928 * 1208 * 3 / 2; i++) {
    ;
    /*if(i%100==0) {
        fprintf(dataf,"%d ,",data[i]);
        //dataf << input[i] << ", ";
        if(i%1000==0) {
            fprintf(dataf,"\n");
        }
    }*/
    //printf("Data: %d\n",input[i]);
    fprintf(dataf, "%d ", data[i]);
    if (i % 1000 == 0) {
      fprintf(dataf, "\n");
    }
  }
  fclose(dataf);

  //printf("Start clSetKernelArg\n");
  // y component: set kernel arguments
  CL_CHECK(clSetKernelArg(krnl, 0, sizeof(cl_mem), & in_yuv_test));
  CL_CHECK(clSetKernelArg(krnl, 1, sizeof(cl_int), & in_y_width));
  CL_CHECK(clSetKernelArg(krnl, 2, sizeof(cl_int), & in_y_offset));
  CL_CHECK(clSetKernelArg(krnl, 3, sizeof(cl_int), & in_y_height));
  CL_CHECK(clSetKernelArg(krnl, 4, sizeof(cl_int), & in_y_width));
  CL_CHECK(clSetKernelArg(krnl, 5, sizeof(cl_mem), & out_y));
  CL_CHECK(clSetKernelArg(krnl, 6, sizeof(cl_int), & out_y_width));
  CL_CHECK(clSetKernelArg(krnl, 7, sizeof(cl_int), & zero));
  CL_CHECK(clSetKernelArg(krnl, 8, sizeof(cl_int), & out_y_height));
  CL_CHECK(clSetKernelArg(krnl, 9, sizeof(cl_int), & out_y_width));
  CL_CHECK(clSetKernelArg(krnl, 10, sizeof(cl_mem), & m_y_cl));

  // y component: initialize two dimensional work size
  const size_t work_size_y[2] = {
    (size_t) out_y_width,
    (size_t) out_y_height
  };

  // debugging  
  // printf("Start clEnqueueNDRangeKernel\n");
  // printf("in_yuv: start uint64_t= %" PRIx64 "\n",*((uint64_t*)in_yuv));

  // y component: start kernel
  CL_CHECK(clEnqueueNDRangeKernel(queue, krnl, 2, NULL,
    (const size_t * ) & work_size_y, NULL, 0, 0, NULL));
  
  // y component: copy buffer from device to host
  CL_CHECK(clEnqueueReadBuffer(queue, out_y, CL_TRUE, 0, 131072, (void * ) output_y, 0, NULL, NULL));

  // y component: output data to file
  FILE * outputfy = fopen("test_output_y.txt", "w");
  fprintf(outputfy, "output_y: \n");
  for (int i = 0; i < 131072; i++) {
    ;
    if (i % 100 == 0) {
      fprintf(outputfy, "%d ,", output_y[i]);
      //dataf << input[i] << ", ";
      if (i % 1000 == 0) {
        fprintf(outputfy, "\n");
      }
    }
    //printf("Data: %d\n",input[i]);
  }
  fclose(outputfy);

  // u and v component: initialize two dimensional work_size
  const size_t work_size_uv[2] = {
    (size_t) out_uv_width,
    (size_t) out_uv_height
  };
  
  // u component: set kernel arguments
  CL_CHECK(clSetKernelArg(krnl, 1, sizeof(cl_int), & in_uv_width));
  CL_CHECK(clSetKernelArg(krnl, 2, sizeof(cl_int), & in_u_offset));
  CL_CHECK(clSetKernelArg(krnl, 3, sizeof(cl_int), & in_uv_height));
  CL_CHECK(clSetKernelArg(krnl, 4, sizeof(cl_int), & in_uv_width));
  CL_CHECK(clSetKernelArg(krnl, 5, sizeof(cl_mem), & out_u));
  CL_CHECK(clSetKernelArg(krnl, 6, sizeof(cl_int), & out_uv_width));
  CL_CHECK(clSetKernelArg(krnl, 7, sizeof(cl_int), & zero));
  CL_CHECK(clSetKernelArg(krnl, 8, sizeof(cl_int), & out_uv_height));
  CL_CHECK(clSetKernelArg(krnl, 9, sizeof(cl_int), & out_uv_width));
  CL_CHECK(clSetKernelArg(krnl, 10, sizeof(cl_mem), & m_uv_cl));
  
  // u component: start kernel
  CL_CHECK(clEnqueueNDRangeKernel(queue, krnl, 2, NULL,
    (const size_t * ) & work_size_uv, NULL, 0, 0, NULL));

  // u component: copy buffer from device to host
  CL_CHECK(clEnqueueReadBuffer(queue, out_u, CL_TRUE, 0, 32768, (void * ) output_u, 0, NULL, NULL));
  
  // u component: output data to file
  FILE * outputfu = fopen("test_output_u.txt", "w");
  fprintf(outputfu, "output_u: \n");
  for (int i = 0; i < 32768; i++) {
    ;
    if (i % 100 == 0) {
      fprintf(outputfu, "%d ,", output_u[i]);
      if (i % 1000 == 0) {
        fprintf(outputfu, "\n");
      }
    }
    //printf("Data: %d\n",input[i]);
  }
  fclose(outputfu);

  // v component: set kernel arguments
  CL_CHECK(clSetKernelArg(krnl, 2, sizeof(cl_int), & in_v_offset));
  CL_CHECK(clSetKernelArg(krnl, 5, sizeof(cl_mem), & out_v));
  
  // v component: start kernel
  CL_CHECK(clEnqueueNDRangeKernel(queue, krnl, 2, NULL,
    (const size_t * ) & work_size_uv, NULL, 0, 0, NULL));
  
  // v component: copy buffer from device to host
  CL_CHECK(clEnqueueReadBuffer(queue, out_v, CL_TRUE, 0, 32768, (void * ) output_v, 0, NULL, NULL));
  
  // v component: output data to file
  FILE * outputfv = fopen("test_output_v.txt", "w");
  fprintf(outputfv, "output_u: \n");
  for (int i = 0; i < 32768; i++) {
    ;
    if (i % 100 == 0) {
      fprintf(outputfv, "%d ,", output_v[i]);
      if (i % 1000 == 0) {
        fprintf(outputfv, "\n");
      }
    }
    //printf("Data: %d\n",input[i]);
  }
  fclose(outputfv);
  
  // clean up OpenCL buffers and environment
  CL_CHECK(clReleaseMemObject(m_uv_cl));
  CL_CHECK(clReleaseMemObject(m_y_cl));
  CL_CHECK(clReleaseMemObject(out_y));
  CL_CHECK(clReleaseMemObject(out_u));
  CL_CHECK(clReleaseMemObject(in_yuv_test));

  CL_CHECK(clReleaseCommandQueue(queue));

  CL_CHECK(clReleaseKernel(krnl));
  
  // clean up host buffers
  free(data);
  free(input);
  free(output_y);
  free(output_u);

  printf("test_transform finished\n");
}