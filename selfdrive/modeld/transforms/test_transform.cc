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

    

    const cl_queue_properties props[] = {0}; //CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0};


    cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
    cl_context context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));

    int error; 
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device_id, props, &error);
    if(error != 0) {
        printf("clCreateCommandQueueWithProperties error: %d\n",error);
    }

    cl_program prg = cl_program_from_file(context, device_id, "transforms/transform.cl", "");
    cl_kernel krnl = CL_CHECK_ERR(clCreateKernel(prg, "warpPerspective", &err));
    CL_CHECK(clReleaseProgram(prg));

    u_char *input = (u_char *)malloc(1928*1208*3/2);
    u_char *data = (u_char *)malloc(1928*1208*3/2);
    u_char *output_y = (u_char *)malloc(131072);
    u_char *output_u = (u_char *)malloc(32768);
    u_char *output_v = (u_char *)malloc(32768);
    

    /*input[0] = 0;
    input[1] = 255;
    input[2] = 10;

    printf("1: %d, 2: %d, 3: %d\n",input[0],input[1],input[2]);*/

    /*ofstream dataf ("test_data.txt");
    if (!dataf.is_open())
    {
        cout << "Unable to open file\n";        
    }*/
    FILE *inputf = fopen ("test_input.txt","w");
    fprintf(inputf,"Input: \n");
    //dataf << "Data: \n";
    for(int i=0; i<1928*1208*3/2; i++) {
        //((u_char*)input+i) = 1;
        input[i] = (i % 256);
        if(i%100==0) {
            fprintf(inputf,"%d ,",input[i]);
            //dataf << input[i] << ", ";
            if(i%1000==0) {
                fprintf(inputf,"\n");
            }
        }
        //printf("Data: %d\n",input[i]);
    }
    fclose(inputf);
    //printf("size_of: %lu\n",sizeof(input));
    cl_mem m_y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err));
    cl_mem m_uv_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err));
    cl_mem in_yuv_test = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE , 1928*1208*3/2, NULL, &err));
    cl_mem out_y = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE , 131072, NULL, &err));
    cl_mem out_u = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE , 32768, NULL, &err));
    cl_mem out_v = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE , 32768, NULL, &err));

    const int zero = 0;

    mat3 projection_y, projection_uv;

    for(int i=0; i<10; i++) {
        projection_y.v[i] = 1;
        projection_uv.v[i] = 0.5;
    }
    

    CL_CHECK(clEnqueueWriteBuffer(queue, m_y_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_y.v, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, m_uv_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_uv.v, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, in_yuv_test, CL_TRUE, 0, 1928*1208*3/2, (void*)input, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(queue, in_yuv_test, CL_TRUE, 0, 1928*1208*3/2, (void*)data, 0, NULL, NULL));

    /*for(int i=0; i<10; i++) {
        printf("projection_y: %d\t%f\n",i,*((float *)m_y_cl+i));
        //printf("projection_uv: %d\t%f\n",i,*(projection_uv_cpu+i));
    }*/


    const int in_y_width = 1928;
    const int in_y_height = 1208;
    const int in_uv_width = 1928/2;
    const int in_uv_height = 1208/2;
    const int in_y_offset = 0;
    const int in_u_offset = in_y_offset + in_y_width*in_y_height;
    const int in_v_offset = in_u_offset + in_uv_width*in_uv_height;

    const int out_y_width = 512;
    const int out_y_height = 256;
    const int out_uv_width = 512/2;
    const int out_uv_height = 256/2;

    printf("Process test_input\n");

    FILE *dataf = fopen ("test_data.txt","w");
    fprintf(dataf,"Data: \n");
    //dataf << "Data: \n";
    for(int i=0; i<1928*1208*3/2; i++) {;
        if(i%100==0) {
            fprintf(dataf,"%d ,",data[i]);
            //dataf << input[i] << ", ";
            if(i%1000==0) {
                fprintf(dataf,"\n");
            }
        }
        //printf("Data: %d\n",input[i]);
    }
    fclose(dataf);


    printf("Start clSetKernelArg\n");
    CL_CHECK(clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_yuv_test));
    CL_CHECK(clSetKernelArg(krnl, 1, sizeof(cl_int), &in_y_width));
    CL_CHECK(clSetKernelArg(krnl, 2, sizeof(cl_int), &in_y_offset));
    CL_CHECK(clSetKernelArg(krnl, 3, sizeof(cl_int), &in_y_height));
    CL_CHECK(clSetKernelArg(krnl, 4, sizeof(cl_int), &in_y_width));
    CL_CHECK(clSetKernelArg(krnl, 5, sizeof(cl_mem), &out_y));
    CL_CHECK(clSetKernelArg(krnl, 6, sizeof(cl_int), &out_y_width));
    CL_CHECK(clSetKernelArg(krnl, 7, sizeof(cl_int), &zero));
    CL_CHECK(clSetKernelArg(krnl, 8, sizeof(cl_int), &out_y_height));
    CL_CHECK(clSetKernelArg(krnl, 9, sizeof(cl_int), &out_y_width));
    CL_CHECK(clSetKernelArg(krnl, 10, sizeof(cl_mem), &m_y_cl));

  const size_t work_size_y[2] = {(size_t)out_y_width, (size_t)out_y_height};

    printf("Start clEnqueueNDRangeKernel\n");
  // printf("in_yuv: start uint64_t= %" PRIx64 "\n",*((uint64_t*)in_yuv));

  CL_CHECK(clEnqueueNDRangeKernel(queue, krnl, 2, NULL,
                              (const size_t*)&work_size_y, NULL, 0, 0, NULL));

  CL_CHECK(clEnqueueReadBuffer(queue, out_y, CL_TRUE, 0, 131072, (void*)output_y, 0, NULL, NULL));
  
  FILE *outputfy = fopen ("test_output_y.txt","w");
    fprintf(outputfy,"output_y: \n");
    //dataf << "Data: \n";
    for(int i=0; i<131072; i++) {;
        if(i%100==0) {
            fprintf(outputfy,"%d ,",output_y[i]);
            //dataf << input[i] << ", ";
            if(i%1000==0) {
                fprintf(outputfy,"\n");
            }
        }
        //printf("Data: %d\n",input[i]);
    }
    fclose(outputfy);

  const size_t work_size_uv[2] = {(size_t)out_uv_width, (size_t)out_uv_height};

  CL_CHECK(clSetKernelArg(krnl, 1, sizeof(cl_int), &in_uv_width));
  CL_CHECK(clSetKernelArg(krnl, 2, sizeof(cl_int), &in_u_offset));
  CL_CHECK(clSetKernelArg(krnl, 3, sizeof(cl_int), &in_uv_height));
  CL_CHECK(clSetKernelArg(krnl, 4, sizeof(cl_int), &in_uv_width));
  CL_CHECK(clSetKernelArg(krnl, 5, sizeof(cl_mem), &out_u));
  CL_CHECK(clSetKernelArg(krnl, 6, sizeof(cl_int), &out_uv_width));
  CL_CHECK(clSetKernelArg(krnl, 7, sizeof(cl_int), &zero));
  CL_CHECK(clSetKernelArg(krnl, 8, sizeof(cl_int), &out_uv_height));
  CL_CHECK(clSetKernelArg(krnl, 9, sizeof(cl_int), &out_uv_width));
  CL_CHECK(clSetKernelArg(krnl, 10, sizeof(cl_mem), &m_uv_cl));
  
  CL_CHECK(clEnqueueNDRangeKernel(queue, krnl, 2, NULL,
                              (const size_t*)&work_size_uv, NULL, 0, 0, NULL));

  CL_CHECK(clEnqueueReadBuffer(queue, out_u, CL_TRUE, 0, 32768, (void*)output_u, 0, NULL, NULL));
  
  FILE *outputfu = fopen ("test_output_u.txt","w");
    fprintf(outputfu,"output_u: \n");
    //dataf << "Data: \n";
    for(int i=0; i<32768; i++) {;
        if(i%100==0) {
            fprintf(outputfu,"%d ,",output_u[i]);
            //dataf << input[i] << ", ";
            if(i%1000==0) {
                fprintf(outputfu,"\n");
            }
        }
        //printf("Data: %d\n",input[i]);
    }
    fclose(outputfu);

  CL_CHECK(clSetKernelArg(krnl, 2, sizeof(cl_int), &in_v_offset));
  CL_CHECK(clSetKernelArg(krnl, 5, sizeof(cl_mem), &out_v));

  CL_CHECK(clEnqueueNDRangeKernel(queue, krnl, 2, NULL,
                              (const size_t*)&work_size_uv, NULL, 0, 0, NULL));

CL_CHECK(clEnqueueReadBuffer(queue, out_v, CL_TRUE, 0, 32768, (void*)output_v, 0, NULL, NULL));
  
  FILE *outputfv = fopen ("test_output_v.txt","w");
    fprintf(outputfv,"output_u: \n");
    //dataf << "Data: \n";
    for(int i=0; i<32768; i++) {;
        if(i%100==0) {
            fprintf(outputfv,"%d ,",output_v[i]);
            //dataf << input[i] << ", ";
            if(i%1000==0) {
                fprintf(outputfv,"\n");
            }
        }
        //printf("Data: %d\n",input[i]);
    }
    fclose(outputfv);

  CL_CHECK(clReleaseMemObject(m_uv_cl));
  CL_CHECK(clReleaseMemObject(m_y_cl));
  CL_CHECK(clReleaseMemObject(out_y));
  CL_CHECK(clReleaseMemObject(out_u));
  CL_CHECK(clReleaseMemObject(in_yuv_test));

  CL_CHECK(clReleaseCommandQueue(queue));

  CL_CHECK(clReleaseKernel(krnl));

  free(data);
  free(input);
  free(output_y);
  free(output_u);

  printf("Test done\n");
}