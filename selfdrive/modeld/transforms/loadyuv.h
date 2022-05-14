#pragma once

#include "selfdrive/common/clutil.h"

typedef struct {
  int width, height;
  cl_kernel loadys_krnl, loaduv_krnl, copy_krnl;
} LoadYUVState;

void loadyuv_init(LoadYUVState* s, cl_context ctx, cl_device_id device_id, int width, int height);

void loadyuv_destroy(LoadYUVState* s);

// biserdan: openCL
/*
void loadyuv_queue(LoadYUVState* s, cl_command_queue q,
                   cl_mem y_cl, cl_mem u_cl, cl_mem v_cl,
                   cl_mem out_cl, bool do_shift = false);
*/
// biserdan: CUDA
void loadyuv_queue(LoadYUVState* s,uint16_t *y_cuda_d,
                   uint16_t *u_cuda_d, uint16_t *v_cuda_d,
                   uint16_t *out_cuda, bool do_shift = false);