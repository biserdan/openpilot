#include <stdint.h>
#include <stdio.h>

#define RGB_TO_Y(r, g, b) ((((__mul24(b, 13) + __mul24(g, 65) + __mul24(r, 33)) + 64) >> 7) + 16)
#define RGB_TO_U(r, g, b) ((__mul24(b, 56) - __mul24(g, 37) - __mul24(r, 19) + 0x8080) >> 8)
#define RGB_TO_V(r, g, b) ((__mul24(r, 56) - __mul24(g, 47) - __mul24(b, 9) + 0x8080) >> 8)
#define AVERAGE(x, y, z, w) (((uint16_t)(x) + (uint16_t)(y) + (uint16_t)(z) + (uint16_t)(w) + 1) >> 1)

#define CL_DEBUG 1  

__device__ void convert_2_ys(uint8_t * out_yuv, int yi, const uint8_t *rgbs1, int rgb_size) {
  uint8_t yy[2] = {
    (uint8_t) RGB_TO_Y(rgbs1[2], rgbs1[1], rgbs1[0]), 
    (uint8_t) RGB_TO_Y(rgbs1[5], rgbs1[4], rgbs1[3])
    };
#ifdef CL_DEBUG
  if(yi >= rgb_size)
    printf("Y vector2 overflow, %d > %d\n", yi, rgb_size);
#endif
  out_yuv[yi] = yy[0];
  out_yuv[yi+1] = yy[1];
}


__device__ void convert_4_ys(uint8_t * out_yuv, int yi, const uint8_t *rgbs1, const uint8_t *rgbs3, int rgb_size) {
  const uint8_t yy[4] = {
    (uint8_t) RGB_TO_Y(rgbs1[2], rgbs1[1], rgbs1[0]),
    (uint8_t) RGB_TO_Y(rgbs1[5], rgbs1[4], rgbs1[3]),
    (uint8_t) RGB_TO_Y(rgbs3[0], rgbs1[7], rgbs1[6]),
    (uint8_t) RGB_TO_Y(rgbs3[3], rgbs3[2], rgbs3[1])
  };
#ifdef CL_DEBUG
  if(yi > rgb_size - 4)
    printf("Y vector4 overflow, %d > %d\n", yi, rgb_size - 4);
#endif
  out_yuv[yi] = yy[0];
  out_yuv[yi+1] = yy[0];
  out_yuv[yi+2] = yy[0];
  out_yuv[yi+3] = yy[0];
}


__device__ void convert_uv(uint8_t * out_yuv, int ui, int vi,
                    const uint8_t * rgbs1, const uint8_t * rgbs2, int rgb_size) {
  // U & V: average of 2x2 pixels square
  const short ab = AVERAGE(rgbs1[0], rgbs1[3], rgbs2[0], rgbs2[3]);
  const short ag = AVERAGE(rgbs1[1], rgbs1[4], rgbs2[1], rgbs2[4]);
  const short ar = AVERAGE(rgbs1[2], rgbs1[5], rgbs2[2], rgbs2[5]);
#ifdef CL_DEBUG
  if(ui >= rgb_size  + rgb_size / 4)
    printf("U overflow, %d >= %d\n", ui, rgb_size  + rgb_size / 4);
  if(vi >= rgb_size  + rgb_size / 2)
    printf("V overflow, %d >= %d\n", vi, rgb_size  + rgb_size / 2);
#endif
  out_yuv[ui] = (uint8_t) RGB_TO_U(ar, ag, ab);
  out_yuv[vi] = (uint8_t) RGB_TO_V(ar, ag, ab);
}


__device__ void convert_2_uvs(uint8_t * out_yuv, int ui, int vi,
                    const uint8_t *rgbs1, const uint8_t *rgbs2, const uint8_t *rgbs3, const uint8_t *rgbs4, int rgb_size) {
  // U & V: average of 2x2 pixels square
  const short ab1 = AVERAGE(rgbs1[0], rgbs1[3], rgbs2[0], rgbs2[3]);
  const short ag1 = AVERAGE(rgbs1[1], rgbs1[4], rgbs2[1], rgbs2[4]);
  const short ar1 = AVERAGE(rgbs1[2], rgbs1[5], rgbs2[2], rgbs2[5]);
  const short ab2 = AVERAGE(rgbs1[6], rgbs3[1], rgbs2[6], rgbs4[1]);
  const short ag2 = AVERAGE(rgbs1[7], rgbs3[2], rgbs2[7], rgbs4[2]);
  const short ar2 = AVERAGE(rgbs3[0], rgbs3[3], rgbs4[0], rgbs4[3]);
  uint8_t u2[2] = {
    (uint8_t) RGB_TO_U(ar1, ag1, ab1),
    (uint8_t) RGB_TO_U(ar2, ag2, ab2)
  };
  uint8_t v2[2] = {
    (uint8_t) RGB_TO_V(ar1, ag1, ab1),
    (uint8_t) RGB_TO_V(ar2, ag2, ab2)
  };
#ifdef CL_DEBUG
  if(ui > rgb_size  + rgb_size / 4 - 2)
    printf("U 2 overflow, %d >= %d\n", ui, rgb_size  + rgb_size / 4 - 2);
  if(vi > rgb_size  + rgb_size / 2 - 2)
    printf("V 2 overflow, %d >= %d\n", vi, rgb_size  + rgb_size / 2 - 2);
#endif
  out_yuv[ui] = u2[0];
  out_yuv[ui+1] = u2[1];
  out_yuv[vi] = v2[0];
  out_yuv[vi+1] = v2[1];
}


__global__ void rgb_to_yuv(uint8_t const * const rgb, uint8_t * out_yuv, int width, int height, int rgb_stride)
{
  int uv_width = width / 2;
  int uv_height = height / 2;
  int rgb_size = width * height;
  
  const int dx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dy = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = __mul24(dx, 4); // Current column in rgb image
  const int row = __mul24(dy, 4); // Current row in rgb image
  const int bgri_start = __mul24(row, rgb_stride) + mul24(col, 3); // Start offset of rgb data being converted
  const int yi_start = __mul24(row,  width) + (col); // Start offset in the target yuv buffer
  int ui = __mul24(row / 2, uv_width) + (rgb_size + col / 2);
  int vi = __mul24(row / 2 , uv_width) + (rgb_size + uv_width * uv_height + col / 2);
  int num_col = min(width - col, 4);
  int num_row = min(height - row, 4);
  uint8_t rgbs0_0[8]; 
  uint8_t rgbs0_1[8];
  uint8_t rgbs1_0[8];
  uint8_t rgbs1_1[8];
  uint8_t rgbs2_0[8];
  uint8_t rgbs2_1[8];
  uint8_t rgbs3_0[8];
  uint8_t rgbs3_1[8];
  memcpy(rgbs0_0, rgb + bgri_start, 8);
  memcpy(rgbs0_1, rgb + bgri_start + 8, 8);
  memcpy(rgbs1_0, rgb + bgri_start + rgb_stride, 8);
  memcpy(rgbs1_1, rgb + bgri_start + rgb_stride + 8, 8);
  if(num_row == 4) {  
    memcpy(rgbs2_0, rgb + bgri_start + rgb_stride * 2, 8);
    memcpy(rgbs2_1, rgb + bgri_start + rgb_stride * 2 + 8, 8);
    memcpy(rgbs3_0, rgb + bgri_start + rgb_stride * 3, 8);
    memcpy(rgbs3_1, rgb + bgri_start + rgb_stride * 3 + 8, 8);
    if(num_col == 4) {
      convert_4_ys(out_yuv, yi_start, rgbs0_0, rgbs0_1, rgb_size);
      convert_4_ys(out_yuv, yi_start + width, rgbs1_0, rgbs1_1, rgb_size);
      convert_4_ys(out_yuv, yi_start + width * 2, rgbs2_0, rgbs2_1, rgb_size);
      convert_4_ys(out_yuv, yi_start + width * 3, rgbs3_0, rgbs3_1, rgb_size);
      convert_2_uvs(out_yuv, ui, vi, rgbs0_0, rgbs1_0, rgbs0_1, rgbs1_1, rgb_size);
      convert_2_uvs(out_yuv, ui + uv_width, vi + uv_width, rgbs2_0, rgbs3_0, rgbs2_1, rgbs3_1, rgb_size);
    } else if(num_col == 2) {
      convert_2_ys(out_yuv, yi_start, rgbs0_0, rgb_size);
      convert_2_ys(out_yuv, yi_start + width, rgbs1_0, rgb_size);
      convert_2_ys(out_yuv, yi_start + width * 2, rgbs2_0, rgb_size);
      convert_2_ys(out_yuv, yi_start + width * 3, rgbs3_0, rgb_size);
      convert_uv(out_yuv, ui, vi, rgbs0_0, rgbs1_0, rgb_size);
      convert_uv(out_yuv, ui + uv_width, vi + uv_width, rgbs2_0, rgbs3_0, rgb_size);
    }
  } else {
    if(num_col == 4) {
      convert_4_ys(out_yuv, yi_start, rgbs0_0, rgbs0_1, rgb_size);
      convert_4_ys(out_yuv, yi_start + width, rgbs1_0, rgbs1_1, rgb_size);
      convert_2_uvs(out_yuv, ui, vi, rgbs0_0, rgbs1_0, rgbs0_1, rgbs1_1, rgb_size);
    } else if(num_col == 2) {
      convert_2_ys(out_yuv, yi_start, rgbs0_0, rgb_size);
      convert_2_ys(out_yuv, yi_start + width, rgbs1_0, rgb_size);
      convert_uv(out_yuv, ui, vi, rgbs0_0, rgbs1_0, rgb_size);
    }
  }
}

void start_kernel(size_t *work_size, void *rgb_cuda, void *yuv_cuda, int width, int heigth, int rgb_stride) {
  dim3 grid = {(uint)work_size[0], (uint)work_size[1]};
  dim3 block = {1,1,1};
  rgb_to_yuv<<<grid, block >>>((uint8_t*)rgb_cuda, (uint8_t*)yuv_cuda, width, heigth, rgb_stride);
}