#pragma once
#include <math.h>
#include "selfdrive/common/clutil.h"
#include "selfdrive/common/mat.h"

typedef struct {
  int width, height;
  //cl_kernel loadys_krnl, loaduv_krnl, copy_krnl;
} LoadYUVState;

void loadyuv_init(LoadYUVState* s, int width, int height);

void loadyuv_destroy(LoadYUVState* s);

// biserdan: openCL
/*
void loadyuv_queue(LoadYUVState* s, cl_command_queue q,
                   cl_mem y_cl, cl_mem u_cl, cl_mem v_cl,
                   cl_mem out_cl, bool do_shift = false);
*/
// biserdan: CUDA
void loadyuv_queue(LoadYUVState* s,uint8_t *y_cuda_d,
                   uint8_t *u_cuda_d, uint8_t *v_cuda_d,
                   float_t *out_cuda, bool do_shift = false);