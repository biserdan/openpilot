#include <stdio.h>        // C programming header file
#include <unistd.h>       // C programming header file
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <inttypes.h>

#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)
#define checkMsgNoFail(msg) __checkMsgNoFail(msg, __FILE__, __LINE__)

#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f / INTER_TAB_SIZE

#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

void start_warpPerspective(uint8_t *y_cuda_d, int src_step, int src_offset,
        int src_rows, int src_cols, uint8_t *dst, int dst_step, int dst_offset,
        int dst_rows,int dst_cols, float_t * M, const size_t *work_size_y);