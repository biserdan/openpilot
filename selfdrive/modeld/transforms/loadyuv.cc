#include "selfdrive/modeld/transforms/loadyuv.h"
#include "selfdrive/modeld/transforms/loadyuv_cuda.cuh"

#include <cassert>
#include <cstdio>
#include <cstring>

void loadyuv_init(LoadYUVState* s, int width, int height) {
  //fprintf(stdout, "loadyuv_init\n");
  memset(s, 0, sizeof(*s));

  s->width = width;
  s->height = height;

  /*char args[1024];
  snprintf(args, sizeof(args),
           "-cl-fast-relaxed-math -cl-denorms-are-zero "
           "-DTRANSFORMED_WIDTH=%d -DTRANSFORMED_HEIGHT=%d",
           width, height);
  cl_program prg = cl_program_from_file(ctx, device_id, "transforms/loadyuv.cl", args);

  s->loadys_krnl = CL_CHECK_ERR(clCreateKernel(prg, "loadys", &err));
  s->loaduv_krnl = CL_CHECK_ERR(clCreateKernel(prg, "loaduv", &err));
  s->copy_krnl = CL_CHECK_ERR(clCreateKernel(prg, "copy", &err));

  // done with this
  CL_CHECK(clReleaseProgram(prg));*/
}

/*void loadyuv_destroy(LoadYUVState* s) {
  CL_CHECK(clReleaseKernel(s->loadys_krnl));
  CL_CHECK(clReleaseKernel(s->loaduv_krnl));
  CL_CHECK(clReleaseKernel(s->copy_krnl));
}*/
// biserdan: openCL
/*
void loadyuv_queue(LoadYUVState* s, cl_command_queue q,
                   cl_mem y_cl, cl_mem u_cl, cl_mem v_cl,
                   cl_mem out_cl, bool do_shift) {
*/
// biserdan: CUDA
void loadyuv_queue(LoadYUVState* s,uint8_t *y_cuda_d,
                   uint8_t *u_cuda_d, uint8_t *v_cuda_d,
                   float_t *out_cuda, bool do_shift) {
  //cl_int global_out_off = 0;
  int global_out_off = 0;
  // fprintf(stdout, "do_shift=%s\n",do_shift ? "true":"false");
  // biserdan: not needed
  
  if (do_shift) {
    // shift the image in slot 1 to slot 0, then place the new image in slot 1
    global_out_off += (s->width*s->height) + (s->width/2)*(s->height/2)*2;
    // CL_CHECK(clSetKernelArg(s->copy_krnl, 0, sizeof(cl_mem), &out_cl));
    // CL_CHECK(clSetKernelArg(s->copy_krnl, 1, sizeof(cl_int), &global_out_off));
    // const size_t copy_work_size = global_out_off/8;
    const int copy_work_size = global_out_off / 8;
    //printf("copy_work_size=%lu\n",copy_work_size);
    //fprintf(stdout, "copy_work_size=%zu\n",copy_work_size);
    //CL_CHECK(clEnqueueNDRangeKernel(q, s->copy_krnl, 1, NULL,&copy_work_size, NULL, 0, 0, NULL));
    start_copy(out_cuda,global_out_off,copy_work_size);
  }
  // biserdan: openCL
  //CL_CHECK(clSetKernelArg(s->loadys_krnl, 0, sizeof(cl_mem), &y_cl));
  //CL_CHECK(clSetKernelArg(s->loadys_krnl, 1, sizeof(cl_mem), &out_cl));
  //CL_CHECK(clSetKernelArg(s->loadys_krnl, 2, sizeof(cl_int), &global_out_off));
  //const size_t loadys_work_size = (s->width*s->height)/8;
  const int loadys_work_size = (s->width*s->height);
  //fprintf(stdout, "loadys_work_size=%zu\n",loadys_work_size);
  //fprintf(stdout, "loadys_work_size=%d\n",loadys_work_size);
  /*CL_CHECK(clEnqueueNDRangeKernel(q, s->loadys_krnl, 1, NULL,
                               &loadys_work_size, NULL, 0, 0, NULL));*/

  // biserdan: cuda loadys
  start_loadys(y_cuda_d,out_cuda,global_out_off,loadys_work_size,
     s->width, s->height);

  
  //const size_t loaduv_work_size = ((s->width/2)*(s->height/2))/8;
  const int loaduv_work_size = ((s->width/2)*(s->height/2));
  global_out_off += (s->width*s->height);

  /*CL_CHECK(clSetKernelArg(s->loaduv_krnl, 0, sizeof(cl_mem), &u_cl));
  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 1, sizeof(cl_mem), &out_cl));
  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 2, sizeof(cl_int), &global_out_off));

  CL_CHECK(clEnqueueNDRangeKernel(q, s->loaduv_krnl, 1, NULL,
                               &loaduv_work_size, NULL, 0, 0, NULL));*/

  start_loaduv(u_cuda_d,out_cuda,global_out_off,loaduv_work_size);
  //printf("finish load yuv\n");
  
  global_out_off += (s->width/2)*(s->height/2);

  /*CL_CHECK(clSetKernelArg(s->loaduv_krnl, 0, sizeof(cl_mem), &v_cl));
  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 1, sizeof(cl_mem), &out_cl));
  CL_CHECK(clSetKernelArg(s->loaduv_krnl, 2, sizeof(cl_int), &global_out_off));

  CL_CHECK(clEnqueueNDRangeKernel(q, s->loaduv_krnl, 1, NULL,
                               &loaduv_work_size, NULL, 0, 0, NULL));*/
  start_loaduv(v_cuda_d,out_cuda,global_out_off,loaduv_work_size);
}
