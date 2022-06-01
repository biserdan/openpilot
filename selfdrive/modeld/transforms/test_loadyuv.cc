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
    printf("Test loadyuv\n");

    // uint8_t * test = 0;
    // checkMsg(cudaHostAlloc((void ** ) & test, 131072, cudaHostAllocMapped));

    uint8_t *y_cuda_h,*y_cuda_d;
    checkMsg(cudaHostAlloc((void ** ) &y_cuda_h, 131072, cudaHostAllocMapped));
    checkMsg(cudaHostGetDevicePointer((void ** ) & y_cuda_d, (void * ) y_cuda_h, 0));
    uint8_t *v_cuda_h,*v_cuda_d;
    checkMsg(cudaHostAlloc((void ** ) &v_cuda_h, 32768, cudaHostAllocMapped));
    checkMsg(cudaHostGetDevicePointer((void ** ) & v_cuda_d, (void * ) v_cuda_h, 0));
    uint8_t *u_cuda_h,*u_cuda_d;
    checkMsg(cudaHostAlloc((void ** ) &u_cuda_h, 32768, cudaHostAllocMapped));
    checkMsg(cudaHostGetDevicePointer((void ** ) & u_cuda_d, (void * ) u_cuda_h, 0));
    float *io_buffer_h,*io_buffer_d;
    checkMsg(cudaHostAlloc((void ** ) &io_buffer_h, 32768, cudaHostAllocMapped));
    checkMsg(cudaHostGetDevicePointer((void ** ) & io_buffer_d, (void * ) io_buffer_h, 0));

    FILE *inputfy = fopen ("test_yuvinput_y.txt","w");
    
    fprintf(inputfy,"Input: \n");
    for(int i=0; i<131072; i++) {
        y_cuda_h[i] = (i % 101);
        if(i%100==0) {
            fprintf(inputfy,"%d ,",y_cuda_h[i]);
            if(i%1000==0) {
                fprintf(inputfy,"\n");
            }
        }
    }
    fclose(inputfy);

    /*checkMsg(cudaMemcpy((void * ) test, (void * ) y_cuda_d, 131072, cudaMemcpyDeviceToHost));

    FILE *testf = fopen ("test_yuvinput_test.txt","w");
    fprintf(testf,"Input: \n");
    for(int i=0; i<131072; i++) {
        if(i%100==0) {
            fprintf(testf,"%d ,",test[i]);
            if(i%1000==0) {
                fprintf(testf,"\n");
            }
        }
    }
    fclose(testf);*/

    FILE *inputfu = fopen ("test_yuvinput_u.txt","w");
    FILE *inputfv = fopen ("test_yuvinput_v.txt","w");
    fprintf(inputfu,"Input: \n");
    fprintf(inputfv,"Input: \n");
    for(int i=0; i<32768; i++) {
        //inputuv[i] = (i % 256);
        u_cuda_h[i] = (i % 151)+100;
        v_cuda_h[i] = (i % 201)+150;
        if(i%100==0) {
            fprintf(inputfu,"%d ,",u_cuda_h[i]);
            fprintf(inputfv,"%d ," ,v_cuda_h[i]);
            if(i%1000==0) {
                fprintf(inputfu,"\n");
                fprintf(inputfv,"\n");
            }
        }
    }
    fclose(inputfu);
    fclose(inputfv);

    int global_out_off = 0;
    const int loadys_work_size = (512*256);

    start_loadys(y_cuda_d,io_buffer_d,global_out_off,loadys_work_size,
     512, 256);

    //printf("Output: %f, %f\n",io_buffer_h[0],io_buffer_h[1]);

    FILE *output_loadys_f = fopen ("test_yuvloadys.txt","w");
    fprintf(output_loadys_f,"Output ys: \n");
    for(int i=0; i<196608; i++) {
        if(i%100==0) {
            fprintf(output_loadys_f,"%f ,",io_buffer_h[i]);
            if(i%1000==0) {
                fprintf(output_loadys_f,"\n");
            }
        }
        //fprintf(output_loadys_f,"%f\n",io_buffer_h[i]);
    }
    fclose(output_loadys_f);

    global_out_off += 131072;
    int loaduv_work_size = 131072;

    start_loaduv(u_cuda_d,io_buffer_d,global_out_off,loaduv_work_size);
    
    FILE *output_loadu_f = fopen ("test_yuvloadu.txt","w");
    fprintf(output_loadu_f,"Output u: \n");
    for(int i=0; i<196608; i++) {
        if(i%100==0) {
            fprintf(output_loadu_f,"%f ,",io_buffer_h[i]);
            if(i%1000==0) {
                fprintf(output_loadu_f,"\n");
            }
        }
    }
    fclose(output_loadu_f);

    global_out_off += 256 * 128;

    start_loaduv(v_cuda_d,io_buffer_d,global_out_off,loaduv_work_size);

    FILE *output_loadv_f = fopen ("test_yuvloadv.txt","w");
    fprintf(output_loadv_f,"Output v: \n");
    for(int i=0; i<196608; i++) {
        if(i%100==0) {
            fprintf(output_loadv_f,"%f ,",io_buffer_h[i]);
            if(i%1000==0) {
                fprintf(output_loadv_f,"\n");
            }
        }
    }
    fclose(output_loadv_f);

    global_out_off = 196608;
    
    int copy_work_size = global_out_off;
    
    start_copy(io_buffer_d,global_out_off,copy_work_size);
    
    FILE *output_copy_f = fopen ("test_yuvcopy.txt","w");
    fprintf(output_copy_f,"Output copy: \n");
    for(int i=0; i<196608; i++) {
        if(i%100==0) {
            fprintf(output_copy_f,"%f ,",io_buffer_h[i]);
            if(i%1000==0) {
                fprintf(output_copy_f,"\n");
            }
        }
    }
    fclose(output_copy_f);

    //printf("float_t: %lu\n",sizeof(float_t));
    checkMsg(cudaFreeHost((void * ) y_cuda_h));
    checkMsg(cudaFreeHost((void * ) v_cuda_h));
    checkMsg(cudaFreeHost((void * ) u_cuda_h));
    checkMsg(cudaFreeHost((void * ) io_buffer_h));
    //checkMsg(cudaFreeHost((void * ) test));
    printf("finsih test_loadyuv\n");
}