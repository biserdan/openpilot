#include "selfdrive/modeld/transforms/test_transform.h"
#include "selfdrive/modeld/transforms/transform_cuda.cuh"

#include <cassert>
#include <cstring>
#include <time.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

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

void test_transform() {

    // uint8_t *input = static_cast<uint8_t*>(malloc(1928*1208*3/2));
    uint8_t *input = 0;
    checkMsg(cudaHostAlloc((void**)&input, 1928*1208*3/2 * sizeof(uint8_t), cudaHostAllocMapped));
    /*uint8_t *data = 0; 
    checkMsg(cudaHostAlloc((void**)&data, 1928*1208*3/2 * sizeof(uint8_t), cudaHostAllocMapped));*/
    uint8_t *output_cpu = 0; 
    checkMsg(cudaHostAlloc((void**)&output_cpu, 131072 * sizeof(uint8_t), cudaHostAllocMapped));

    float_t *projection_y_cpu = 0; 
    checkMsg(cudaHostAlloc((void**)&projection_y_cpu, 3 * 3 * sizeof(float_t), cudaHostAllocMapped));
    float_t *projection_uv_cpu = 0; 
    checkMsg(cudaHostAlloc((void**)&projection_uv_cpu, 3 * 3 * sizeof(float_t), cudaHostAllocMapped));




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


    uint8_t *in_yuv_test_h;
    uint8_t *in_yuv_test_d;
    //uint8_t *out_y_h, 
    uint8_t *out_y_d;
    //float_t *m_y_cuda_h;
    float_t *m_y_cuda_d;
    //float_t *m_uv_cuda_h,
    float_t *m_uv_cuda_d;

    checkMsg(cudaHostAlloc((void **)&in_yuv_test_h, 1928*1208*3/2, cudaHostAllocMapped));
    checkMsg(cudaHostGetDevicePointer((void **)&in_yuv_test_d, (void *)in_yuv_test_h, 0));
    //checkMsg(cudaMalloc((void**)&in_yuv_test_d, 1928*1208*3/2 * sizeof(uint8_t)));

    /*checkMsg(cudaHostAlloc((void **)&out_y_h, 131072, cudaHostAllocMapped));
    checkMsg(cudaHostGetDevicePointer((void **)&out_y_d, (void *)out_y_h, 0));*/
    checkMsg(cudaMalloc((void**)&out_y_d, 131072 * sizeof(uint8_t)));

    /*checkMsg(cudaHostAlloc((void **)&m_y_cuda_h, 3*3*sizeof(float), cudaHostAllocMapped));
    checkMsg(cudaHostGetDevicePointer((void **)&m_y_cuda_d, (void *)m_y_cuda_h, 0));*/
    checkMsg(cudaMalloc((void**)&m_y_cuda_d, 3*3 * sizeof(float_t)));

    /*checkMsg(cudaHostAlloc((void **)&m_uv_cuda_h, 3*3*sizeof(float), cudaHostAllocMapped));
    checkMsg(cudaHostGetDevicePointer((void **)&m_uv_cuda_d, (void *)m_uv_cuda_h, 0));*/
    checkMsg(cudaMalloc((void**)&m_uv_cuda_d, 3*3 * sizeof(float_t)));

    const int zero = 0;

    //mat3 projection_y, projection_uv;

    for(int i=0; i<10; i++) {
        projection_y_cpu[i] = 1;
        projection_uv_cpu[i] = 0.5;
    }
    /*for(int i=0; i<10; i++) {
        printf("projection_y: %d\t%f\n",i,*(projection_y_cpu+i));
        printf("projection_uv: %d\t%f\n",i,*(projection_uv_cpu+i));
    }*/


    checkMsg(cudaMemcpy((void *)in_yuv_test_d,(void*)input,1928*1208*3/2,cudaMemcpyHostToDevice));
    // checkMsg(cudaMemcpy((void *)data,(void*)in_yuv_test_d,1928*1208*3/2,cudaMemcpyDeviceToHost));
    checkMsg(cudaMemcpy((void *)m_y_cuda_d,(void*)projection_y_cpu,3*3*sizeof(float),cudaMemcpyHostToDevice));
    checkMsg(cudaMemcpy((void *)m_uv_cuda_d,(void*)projection_uv_cpu,3*3*sizeof(float),cudaMemcpyHostToDevice));

    const int in_y_width = 1928;
    const int in_y_height = 1208;
    // const int in_uv_width = 1928/2;
    // const int in_uv_height = 1208/2;
    const int in_y_offset = 0;
    // const int in_u_offset = in_y_offset + in_y_512;
    const int out_y_height = 512;
    // const int in_v_offset = in_u_offset + in_uv_width*in_uv_height;

    const int out_y_width = 256;
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
            inputf << (int)(*((uint8_t*)in_yuv_test_h+i)) << ", ";
            if(i%1000==0) {
                inputf << "\n";
            }
        }

    }
    printf("Finish test_input\n");
    inputf.close();

    const size_t work_size_y[2] = {(size_t)out_y_width, (size_t)out_y_height};

    start_warpPerspective(in_yuv_test_d,in_y_width,in_y_offset,in_y_height,in_y_width,
      out_y_d,out_y_width,zero,out_y_height,out_y_width,m_y_cuda_d,
      (const size_t*)&work_size_y);
    
    checkMsg(cudaMemcpy((void *)output_cpu,(void*)out_y_d,131072,cudaMemcpyDeviceToHost));

    ofstream outputf ("test_output.txt");
    if (!outputf.is_open())
    {
        cout << "Unable to open file\n";        
    }
    outputf << "Output: \n";
    for(int i=0; i<131072; i++) {
        //if(i%100==0) {
            outputf << (int)(*((uint8_t*)output_cpu+i)) << ", ";
            if(i%1000==0) {
                outputf << "\n";
            //}
        }
    }
    outputf.close();

  //checkMsg(cudaFreeHost((void *)m_y_cuda_h));
  checkMsg(cudaFree((void *)m_y_cuda_d));
  //checkMsg(cudaFreeHost((void *)m_uv_cuda_h));
  checkMsg(cudaFree((void *)m_uv_cuda_d));
  checkMsg(cudaFreeHost((void *)input));
  //checkMsg(cudaFreeHost((void *)data));
  checkMsg(cudaFreeHost((void *)in_yuv_test_h));
  checkMsg(cudaFreeHost((void *)projection_uv_cpu));
  checkMsg(cudaFreeHost((void *)projection_y_cpu));
  //checkMsg(cudaFree((void *)in_yuv_test_d));
  checkMsg(cudaFree((void *)out_y_d));
  //checkMsg(cudaFreeHost((void *)out_y_h));
}