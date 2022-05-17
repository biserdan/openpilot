#include "selfdrive/modeld/transforms/transform.h"
// #include "selfdrive/modeld/transforms/hello_cuda.cuh"
#include "selfdrive/modeld/transforms/transform_cuda.cuh"

#include <cassert>
#include <cstring>

#include "selfdrive/common/clutil.h"

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

// void transform_init(Transform* s, cl_context ctx, cl_device_id device_id) {
void transform_init(Transform* s) {
  memset(s, 0, sizeof(*s));

  /*cl_program prg = cl_program_from_file(ctx, device_id, "transforms/transform.cl", "");
  s->krnl = CL_CHECK_ERR(clCreateKernel(prg, "warpPerspective", &err));
  // done with this
  CL_CHECK(clReleaseProgram(prg));

  s->m_y_cl = CL_CHECK_ERR(clCreateBuffer(ctx, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err));
  s->m_uv_cl = CL_CHECK_ERR(clCreateBuffer(ctx, CL_MEM_READ_WRITE, 3*3*sizeof(float), NULL, &err));*/

  checkMsg(cudaHostAlloc((void **)&s->m_y_cuda_h, 3*3*sizeof(float), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&s->m_y_cuda_d, (void *)s->m_y_cuda_h, 0));
  checkMsg(cudaHostAlloc((void **)&s->m_uv_cuda_h, 3*3*sizeof(float), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&s->m_uv_cuda_d, (void *)s->m_uv_cuda_h, 0));
}

void transform_destroy(Transform* s) {
  /*CL_CHECK(clReleaseMemObject(s->m_y_cl));
  CL_CHECK(clReleaseMemObject(s->m_uv_cl));
  CL_CHECK(clReleaseKernel(s->krnl));*/

  checkMsg(cudaFreeHost((void *)s->m_y_cuda_h));
  checkMsg(cudaFreeHost((void *)s->m_uv_cuda_h));
}

/*void transform_queue(Transform* s,
                     cl_command_queue q,
                     cl_mem in_yuv, int in_width, int in_height,
                     cl_mem out_y, cl_mem out_u, cl_mem out_v,
                     int out_width, int out_height,
                     const mat3& projection) {*/
void transform_queue(Transform* s,cl_command_queue q,
                     uint8_t *in_yuv, int in_width, int in_height,
                     uint8_t *out_y, uint8_t *out_u, uint8_t *out_v,
                     int out_width, int out_height,
                     const mat3& projection) {

  const int zero = 0;

  // sampled using pixel center origin
  // (because thats how fastcv and opencv does it)

  mat3 projection_y = projection;

  // in and out uv is half the size of y.
  mat3 projection_uv = transform_scale_buffer(projection, 0.5);

  // from host to device

  // CL_CHECK(clEnqueueWriteBuffer(q, s->m_y_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_y.v, 0, NULL, NULL));
  // CL_CHECK(clEnqueueWriteBuffer(q, s->m_uv_cl, CL_TRUE, 0, 3*3*sizeof(float), (void*)projection_uv.v, 0, NULL, NULL));

  checkMsg(cudaMemcpy((void *)s->m_y_cuda_d,(void*)projection_y.v,3*3*sizeof(float),cudaMemcpyHostToDevice));
  checkMsg(cudaMemcpy((void *)s->m_uv_cuda_d,(void*)projection_uv.v,3*3*sizeof(float),cudaMemcpyHostToDevice));


  const int in_y_width = in_width;
  const int in_y_height = in_height;
  const int in_uv_width = in_width/2;
  const int in_uv_height = in_height/2;
  const int in_y_offset = 0;
  const int in_u_offset = in_y_offset + in_y_width*in_y_height;
  const int in_v_offset = in_u_offset + in_uv_width*in_uv_height;

  const int out_y_width = out_width;
  const int out_y_height = out_height;
  const int out_uv_width = out_width/2;
  const int out_uv_height = out_height/2;

  // CL_CHECK(clSetKernelArg(s->krnl, 0, sizeof(cl_mem), &in_yuv));
  // CL_CHECK(clSetKernelArg(s->krnl, 1, sizeof(cl_int), &in_y_width));
  // CL_CHECK(clSetKernelArg(s->krnl, 2, sizeof(cl_int), &in_y_offset));
  // CL_CHECK(clSetKernelArg(s->krnl, 3, sizeof(cl_int), &in_y_height));
  // CL_CHECK(clSetKernelArg(s->krnl, 4, sizeof(cl_int), &in_y_width));
  // CL_CHECK(clSetKernelArg(s->krnl, 5, sizeof(cl_mem), &out_y));
  // CL_CHECK(clSetKernelArg(s->krnl, 6, sizeof(cl_int), &out_y_width));
  // CL_CHECK(clSetKernelArg(s->krnl, 7, sizeof(cl_int), &zero));
  // CL_CHECK(clSetKernelArg(s->krnl, 8, sizeof(cl_int), &out_y_height));
  // CL_CHECK(clSetKernelArg(s->krnl, 9, sizeof(cl_int), &out_y_width));
  // CL_CHECK(clSetKernelArg(s->krnl, 10, sizeof(cl_mem), &s->m_y_cl));

  const size_t work_size_y[2] = {(size_t)out_y_width, (size_t)out_y_height};

  //CL_CHECK(clEnqueueNDRangeKernel(q, s->krnl, 2, NULL,
  //                            (const size_t*)&work_size_y, NULL, 0, 0, NULL));
  start_warpPerspective(in_yuv,in_y_width,in_y_offset,in_y_height,in_y_width,
      out_y,out_y_width,zero,out_y_height,out_y_width,s->m_y_cuda_d,
      (const size_t*)&work_size_y);

  const size_t work_size_uv[2] = {(size_t)out_uv_width, (size_t)out_uv_height};

  // CL_CHECK(clSetKernelArg(s->krnl, 1, sizeof(cl_int), &in_uv_width));
  // CL_CHECK(clSetKernelArg(s->krnl, 2, sizeof(cl_int), &in_u_offset));
  // CL_CHECK(clSetKernelArg(s->krnl, 3, sizeof(cl_int), &in_uv_height));
  // CL_CHECK(clSetKernelArg(s->krnl, 4, sizeof(cl_int), &in_uv_width));
  // CL_CHECK(clSetKernelArg(s->krnl, 5, sizeof(cl_mem), &out_u));
  // CL_CHECK(clSetKernelArg(s->krnl, 6, sizeof(cl_int), &out_uv_width));
  // CL_CHECK(clSetKernelArg(s->krnl, 7, sizeof(cl_int), &zero));
  // CL_CHECK(clSetKernelArg(s->krnl, 8, sizeof(cl_int), &out_uv_height));
  // CL_CHECK(clSetKernelArg(s->krnl, 9, sizeof(cl_int), &out_uv_width));
  // CL_CHECK(clSetKernelArg(s->krnl, 10, sizeof(cl_mem), &s->m_uv_cl));
  
  // CL_CHECK(clEnqueueNDRangeKernel(q, s->krnl, 2, NULL,
  //                            (const size_t*)&work_size_uv, NULL, 0, 0, NULL));
  start_warpPerspective(in_yuv,in_uv_width,in_u_offset,in_uv_height,in_uv_width,
      out_u,out_uv_width,zero,out_uv_height,out_uv_width,s->m_uv_cuda_d,
      (const size_t*)&work_size_uv);
  
  // CL_CHECK(clSetKernelArg(s->krnl, 2, sizeof(cl_int), &in_v_offset));
  // CL_CHECK(clSetKernelArg(s->krnl, 5, sizeof(cl_mem), &out_v));

  start_warpPerspective(in_yuv,in_uv_width,in_v_offset,in_uv_height,in_uv_width,
      out_v,out_uv_width,zero,out_uv_height,out_uv_width,s->m_uv_cuda_d,
      (const size_t*)&work_size_uv);

  // CL_CHECK(clEnqueueNDRangeKernel(q, s->krnl, 2, NULL,
  //                             (const size_t*)&work_size_uv, NULL, 0, 0, NULL));
}
