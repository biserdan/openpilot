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

// ModelFrame::ModelFrame(cl_device_id device_id, cl_context context) {
ModelFrame::ModelFrame() {
  input_frames = std::make_unique<float[]>(buf_size);
  /*checkMsg(cudaMalloc((void **)&test_gpu,sizeof(float)));
  checkMsg(cudaHostAlloc((void **)&test_cpu,sizeof(float),cudaHostAllocMapped));
  test_cpu[0] = 2.5;
  printf("test_cpu=%f\n",test_cpu[0]);
  checkMsg(cudaMemcpy(test_gpu, test_cpu, sizeof(float), cudaMemcpyHostToDevice));
  checkMsg(cudaMemcpy(test_cpu, test_gpu, sizeof(float), cudaMemcpyDeviceToHost));
  printf("test_cpu=%f\n",test_cpu[0]);*/

  /*q = CL_CHECK_ERR(clCreateCommandQueue(context, device_id, 0, &err));
  y_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, MODEL_WIDTH * MODEL_HEIGHT, NULL, &err));
  u_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));
  v_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), NULL, &err));
  net_input_cl = CL_CHECK_ERR(clCreateBuffer(context, CL_MEM_READ_WRITE, MODEL_FRAME_SIZE * sizeof(float), NULL, &err));*/

  checkMsg(cudaHostAlloc((void **)&y_cuda_h, MODEL_WIDTH * MODEL_HEIGHT, cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&y_cuda_d, (void *)y_cuda_h, 0));
  checkMsg(cudaHostAlloc((void **)&u_cuda_h, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&u_cuda_d, (void *)u_cuda_h, 0));
  checkMsg(cudaHostAlloc((void **)&v_cuda_h, (MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&v_cuda_d, (void *)v_cuda_h, 0));
  checkMsg(cudaHostAlloc((void **)&net_input_cuda_h, MODEL_FRAME_SIZE * sizeof(float), cudaHostAllocMapped));
  checkMsg(cudaHostGetDevicePointer((void **)&net_input_cuda_d, (void *)net_input_cuda_h, 0));

  /*checkMsg(cudaMalloc((void **)&y_cuda_d,MODEL_WIDTH * MODEL_HEIGHT));
  checkMsg(cudaMalloc((void **)&u_cuda_d,(MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2)));
  checkMsg(cudaMalloc((void **)&v_cuda_d,(MODEL_WIDTH / 2) * (MODEL_HEIGHT / 2)));
  checkMsg(cudaMalloc((void **)&net_input_cuda_d,MODEL_FRAME_SIZE * sizeof(float)));*/
  //printf("created: net_input_cuda_d\n");

  // transform_init(&transform, context, device_id);
  transform_init(&transform);
  loadyuv_init(&loadyuv, MODEL_WIDTH, MODEL_HEIGHT);
}

//float* ModelFrame::prepare(cl_mem yuv_cl, int frame_width, int frame_height, const mat3 &projection, cl_mem *output) {
float* ModelFrame::prepare(uint8_t *yuv_cl, int frame_width, int frame_height, const mat3 &projection, void **output) {
  /*transform_queue(&this->transform, q,
                  yuv_cl, frame_width, frame_height,
                  y_cl, u_cl, v_cl, MODEL_WIDTH, MODEL_HEIGHT, projection);*/
  transform_queue(&this->transform,
                  yuv_cl, frame_width, frame_height,
                  y_cuda_d, u_cuda_d, v_cuda_d, MODEL_WIDTH, MODEL_HEIGHT, projection);

  if (output == NULL) {
    // biserdan: openCL
    // loadyuv_queue(&loadyuv, q, y_cl, u_cl, v_cl, net_input_cl);
    // biserdan: CUDA
    loadyuv_queue(&loadyuv, y_cuda_d, u_cuda_d, v_cuda_d, net_input_cuda_d);

    std::memmove(&input_frames[0], &input_frames[MODEL_FRAME_SIZE], sizeof(float) * MODEL_FRAME_SIZE);
    //CL_CHECK(clEnqueueReadBuffer(q, net_input_cl, CL_TRUE, 0, MODEL_FRAME_SIZE * sizeof(float), &input_frames[MODEL_FRAME_SIZE], 0, nullptr, nullptr));
    // clFinish(q);
    // from host to device
    //printf("pointers: %p, %p\n",net_input_cuda_d,&input_frames[MODEL_FRAME_SIZE]);
    /*for(int i=0;i<101;i++) {
      printf("input_frame[0]: %f\n",input_frames[MODEL_FRAME_SIZE+i]);
      printf("input_frame[1]: %f\n",input_frames[0]);
    }*/
    //printf("buf_size: %d MODEL_FRAME_SIZE: %d\n",buf_size,MODEL_FRAME_SIZE);
    //buf_size: 393216 MODEL_FRAME_SIZE: 196608
    // cudaDeviceSynchronize();
    std::memmove(&input_frames[MODEL_FRAME_SIZE], net_input_cuda_h, MODEL_FRAME_SIZE * sizeof(float));
    //checkMsg(cudaMemcpy((void *)&input_frames[MODEL_FRAME_SIZE],(void *)net_input_cuda_d,MODEL_FRAME_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    //checkMsg(cudaMemcpy(&test[MODEL_FRAME_SIZE],&net_input_cuda_h[0],MODEL_FRAME_SIZE * sizeof(float), cudaMemcpyHostToHost));
    //cudaMemcpy((void *)&test_gpu[0],(void *)&test_cpu[0],sizeof(float), cudaMemcpyHostToDevice);
    //printf("net_input_cuda_h: %f\n",test_gpu[0]);
    return &input_frames[0];
  } else {
    loadyuv_queue(&loadyuv, y_cuda_d, u_cuda_d, v_cuda_d, net_input_cuda_d, true);
    // loadyuv_queue(&loadyuv, q, y_cl, u_cl, v_cl, *output, true);
    // NOTE: Since thneed is using a different command queue, this clFinish is needed to ensure the image is ready.
    //clFinish(q);
    return NULL;
  }
}

ModelFrame::~ModelFrame() {
  transform_destroy(&transform);
  //loadyuv_destroy(&loadyuv);
  /*CL_CHECK(clReleaseMemObject(net_input_cl));
  CL_CHECK(clReleaseMemObject(v_cl));
  CL_CHECK(clReleaseMemObject(u_cl));
  CL_CHECK(clReleaseMemObject(y_cl));
  CL_CHECK(clReleaseCommandQueue(q));*/

  /*checkMsg(cudaFreeHost((void *)net_input_cuda_h));
  checkMsg(cudaFreeHost((void *)v_cuda_h));
  checkMsg(cudaFreeHost((void *)u_cuda_h));
  checkMsg(cudaFreeHost((void *)y_cuda_h));*/

  checkMsg(cudaFreeHost((void *)net_input_cuda_h));
  cudaFreeHost((void *)y_cuda_h);
  cudaFreeHost((void *)u_cuda_h);
  cudaFreeHost((void *)v_cuda_h);
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
