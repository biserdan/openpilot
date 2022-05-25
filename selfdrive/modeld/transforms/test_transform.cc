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

    uint8_t *input = static_cast<uint8_t*>(malloc(1928*1208*3/2));

    ofstream dataf ("test_data.txt");
    if (!dataf.is_open())
    {
        cout << "Unable to open file\n";        
    }

    dataf << "Data: \n";
    for(int i=0; i<1928*1208*3/2; i++) {
        *((uint8_t*)input+i) = i % 255;
        if(i%100==0) {
            dataf << (int)input[i] << ", ";
            if(i%1000==0) {
                dataf << "\n";
            }
        }
        //printf("Data: %d\t%d\t%p\n",i,*((uint8_t*)input+i),((uint8_t*)input+i));
    }
    dataf.close();
    
    cl_mem m_y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err));
    cl_mem m_uv_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err));
    cl_mem in_yuv_test = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE , 1928*1208*3/2, NULL, &err));
    cl_mem out_y = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE , 131072, NULL, &err));

    const int zero = 0;

    mat3 projection_y, projection_uv;

    for(int i=0; i<10; i++) {
        projection_y.v[i] = 1;
        projection_uv.v[i] = 0.5;
    }
    

    CL_CHECK(clEnqueueWriteBuffer(queue, m_y_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_y.v, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, m_uv_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_uv.v, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(queue, in_yuv_test, CL_TRUE, 0, 1928*1208*3/2, (void*)input, 0, NULL, NULL));


    const int in_y_width = 1928;
    const int in_y_height = 1208;
    // const int in_uv_width = 1928/2;
    // const int in_uv_height = 1208/2;
    const int in_y_offset = 0;
    // const int in_u_offset = in_y_offset + in_y_width*in_y_height;
    // const int in_v_offset = in_u_offset + in_uv_width*in_uv_height;

    const int out_y_width = 512;
    const int out_y_height = 256;
    // const int out_uv_width = 512/2;
    // const int out_uv_height = 256/2;

    printf("Process test_input\n");

    ofstream inputf ("test_input.txt");
    if (!inputf.is_open())
    {
        cout << "Unable to open file\n";        
    }
    inputf << "Input: \n";
    for(int i=0; i<1928*1208*3/2; i++) {
        if(i%100==0) {
            inputf << (int)(*((uint8_t*)in_yuv_test+i)) << ", ";
            if(i%1000==0) {
                inputf << "\n";
            }
            //if((*((uint8_t*)in_yuv_test+i))<0 && (*((uint8_t*)in_yuv_test+i))>255){
            //   printf("Input: %d\t%d\t%p\n",*((uint8_t*)in_yuv_test+i),*((uint8_t*)input+i),((uint8_t*)in_yuv_test+i));     
            //}
        }
        
    }
    printf("Finish test_input\n");
    inputf.close();


    printf("Start clSetKernelArg\n");
    CL_CHECK(clSetKernelArg(krnl, 0, sizeof(cl_mem), &in_yuv_test));
    CL_CHECK(clSetKernelArg(krnl, 1, sizeof(cl_int), &in_y_width));
    CL_CHECK(clSetKernelArg(krnl, 2, sizeof(cl_int), &in_y_offset));
    CL_CHECK(clSetKernelArg(krnl, 3, sizeof(cl_int), &in_y_height));
    CL_CHECK(clSetKernelArg(krnl, 4, sizeof(cl_int), &in_y_width));
    // printf("Successfull");
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
  
  /*printf("Output out_y: \n");
    for(int i=0; i<1928*1208*3/2; i++) {
        *((uint8_t*)in_yuv+i) = rand() % 255;
        printf("%d ,",*((uint8_t*)in_yuv+i));
    }
  printf("\n");*/

  /*myfile << "Output: \n";
  for(int i=0; i<1928*1208*3/2; i++) {
    myfile << *((uint8_t*)out_y+i) << ", ";
    // printf("%d ,",*((uint8_t*)in_yuv+i));
    }
  myfile << "\n";*/

  //printf("out_y: start uint64_t= %" PRIx64 "\n",*((uint64_t*)out_y));
  //printf("s->m_y_cl: start uint64_t= %" PRIx64 "\n",*((uint64_t*)m_y_cl));

  ofstream outputf ("test_output.txt");
    if (!outputf.is_open())
    {
        cout << "Unable to open file\n";        
    }
    outputf << "Output: \n";
    for(int i=0; i<131072; i++) {
        //if(i%100==0) {
            outputf << (int)(*((uint8_t*)out_y+i)) << ", ";
            if(i%1000==0) {
                outputf << "\n";
            //}
        }
    }
    outputf.close();

  
  CL_CHECK(clReleaseMemObject(m_uv_cl));
  CL_CHECK(clReleaseMemObject(m_y_cl));
  CL_CHECK(clReleaseMemObject(out_y));
  CL_CHECK(clReleaseMemObject(in_yuv_test));

  CL_CHECK(clReleaseCommandQueue(queue));

  CL_CHECK(clReleaseKernel(krnl));

  printf("Test done\n");
}