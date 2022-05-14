#include "selfdrive/modeld/models/commonmodel.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/mat.h"
#include "selfdrive/common/timing.h"

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

ModelFrame::ModelFrame(cl_device_id device_id, cl_context context) {
  input_frames = std::make_unique<float[]>(buf_size);

  q = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &err));
  y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, MODEL_WIDTH * MODEL_HEIGHT, NULL, &err));
  u_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));
  v_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));
  net_input_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, MODEL_FRAME_SIZE * sizeof(float), NULL, &err));

  checkMsg(cudaHostAlloc((void **)&y_cuda_h, MODEL_WIDTH * MODEL_HEIGHT, cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&y_cuda_d, (void *)y_cuda_h, 0));
  checkMsg(cudaHostAlloc((void **)&u_cuda_h, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&u_cuda_d, (void *)u_cuda_h, 0));
  checkMsg(cudaHostAlloc((void **)&v_cuda_h, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&v_cuda_d, (void *)v_cuda_h, 0));
  checkMsg(cudaHostAlloc((void **)&net_input_cuda_h, MODEL_FRAME_SIZE * sizeof(float), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&net_input_cuda_d, (void *)net_input_cuda_h, 0));

  transform_init(&transform, context, device_id);
  loadyuv_init(&loadyuv, context, device_id, MODEL_WIDTH, MODEL_HEIGHT);
}

float* ModelFrame::prepare(cl_mem yuv_cl, int frame_width, int frame_height, const mat3 &projection, cl_mem *output) {
  transform_queue(&this->transform, q,
                  yuv_cl, frame_width, frame_height,
                  y_cl, u_cl, v_cl, MODEL_WIDTH, MODEL_HEIGHT, projection);

  if (output == NULL) {
    // biserdan: openCL
    // loadyuv_queue(&loadyuv, q, y_cl, u_cl, v_cl, net_input_cl);
    // biserdan: CUDA
    loadyuv_queue(&loadyuv, y_cuda_d, u_cuda_d, v_cuda_d, net_input_cuda_d);

    std::memmove(&input_frames[0], &input_frames[MODEL_FRAME_SIZE], sizeof(float) * MODEL_FRAME_SIZE);
    CL_CHECK(clEnqueueReadBuffer(q, net_input_cl, CL_TRUE, 0, MODEL_FRAME_SIZE * sizeof(float), &input_frames[MODEL_FRAME_SIZE], 0, nullptr, nullptr));
    clFinish(q);
    return &input_frames[0];
  } else {
    loadyuv_queue(&loadyuv, q, y_cl, u_cl, v_cl, *output, true);
    // NOTE: Since thneed is using a different command queue, this clFinish is needed to ensure the image is ready.
    clFinish(q);
    return NULL;
  }
}

ModelFrame::~ModelFrame() {
  transform_destroy(&transform);
  loadyuv_destroy(&loadyuv);
  CL_CHECK(clReleaseMemObject(net_input_cl));
  CL_CHECK(clReleaseMemObject(v_cl));
  CL_CHECK(clReleaseMemObject(u_cl));
  CL_CHECK(clReleaseMemObject(y_cl));
  CL_CHECK(clReleaseCommandQueue(q));

  checkMsg(cudaFreeHost((void *)net_input_cuda_h));
  checkMsg(cudaFreeHost((void *)v_cuda_h));
  checkMsg(cudaFreeHost((void *)u_cuda_h));
  checkMsg(cudaFreeHost((void *)y_cuda_h));
}

void softmax(const float* input, float* output, size_t len) {
  const float max_val = *std::max_element(input, input + len);
  float denominator = 0;
  for(int i = 0; i < len; i++) {
    float const v_exp = expf(input[i] - max_val);
    denominator += v_exp;
    output[i] = v_exp;
  }

  const float inv_denominator = 1. / denominator;
  for(int i = 0; i < len; i++) {
    output[i] *= inv_denominator;
  }
}

float sigmoid(float input) {
  return 1 / (1 + expf(-input));
}
