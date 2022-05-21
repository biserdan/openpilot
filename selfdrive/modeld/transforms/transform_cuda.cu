#include "transform_cuda.cuh"

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

__global__ void warpPerspective(const uint8_t * src,
                              int src_step, int src_offset, int src_rows, int src_cols,
                              uint8_t * dst,
                              int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              float * M) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;

  if (dx < dst_cols && dy < dst_rows)
    {
        float X0 = M[0] * dx + M[1] * dy + M[2];
        float Y0 = M[3] * dx + M[4] * dy + M[5];
        float W = M[6] * dx + M[7] * dy + M[8];
        W = W != 0.0f ? INTER_TAB_SIZE / W : 0.0f;
        int X = rint(X0 * W), Y = rint(Y0 * W);
        //short sx = convert_short_sat(X >> INTER_BITS);
        short sx = ((X >> INTER_BITS) > INT16_MAX) ? INT16_MAX : 
          static_cast<short>(X >> INTER_BITS);
        if(sx != sx) sx = 0;

        //short sy = convert_short_sat(Y >> INTER_BITS);
        short sy = ((Y >> INTER_BITS) > INT16_MAX) ? INT16_MAX : 
          static_cast<short>(Y >> INTER_BITS);
        if(sy != sy) sy = 0;

        short ay = (short)(Y & (INTER_TAB_SIZE - 1));
        short ax = (short)(X & (INTER_TAB_SIZE - 1));

        int v0 = (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) ?
            //convert_int(src[mad24(sy, src_step, src_offset + sx)]) : 0;
            static_cast<int>(src[sy * src_step + src_offset + sx]) : 0;
        int v1 = (sx+1 >= 0 && sx+1 < src_cols && sy >= 0 && sy < src_rows) ?
            //convert_int(src[mad24(sy, src_step, src_offset + (sx+1))]) : 0;
            static_cast<int>(src[sy * src_step + src_offset + (sx+1)]) : 0;
        int v2 = (sx >= 0 && sx < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            //convert_int(src[mad24(sy+1, src_step, src_offset + sx)]) : 0;
            static_cast<int>(src[(sy+1) * src_step + src_offset + sx]) : 0;
        int v3 = (sx+1 >= 0 && sx+1 < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            //convert_int(src[mad24(sy+1, src_step, src_offset + (sx+1))]) : 0;
            static_cast<int>(src[(sy+1) * src_step + src_offset + (sx+1)]) : 0;

        float taby = 1.f/INTER_TAB_SIZE*ay;
        float tabx = 1.f/INTER_TAB_SIZE*ax;

        // int dst_index = mad24(dy, dst_step, dst_offset + dx);
        int dst_index = dy * dst_step + dst_offset + dx;

        int itab0 = ((1.0f-taby)*(1.0f-tabx) * INTER_REMAP_COEF_SCALE ) > UINT16_MAX ? UINT16_MAX :
                   static_cast<int16_t>(nearbyint((1.0f-taby)*(1.0f-tabx) * INTER_REMAP_COEF_SCALE));
        if(itab0 != itab0) itab0 = 0;  
        // int itab0 = convert_short_sat_rte( (1.0f-taby)*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab1 = ((1.0f-taby)*tabx * INTER_REMAP_COEF_SCALE ) > UINT16_MAX ? UINT16_MAX :
                   static_cast<int16_t>(nearbyint((1.0f-taby)*tabx * INTER_REMAP_COEF_SCALE ));
        if(itab1 != itab1) itab1 = 0; 
        // int itab1 = convert_short_sat_rte( (1.0f-taby)*tabx * INTER_REMAP_COEF_SCALE );
        int itab2 = (taby*(1.0f-tabx) * INTER_REMAP_COEF_SCALE ) > UINT16_MAX ? UINT16_MAX :
                   static_cast<int16_t>(nearbyint(taby*(1.0f-tabx) * INTER_REMAP_COEF_SCALE ));
        if(itab2 != itab2) itab2 = 0; 
        // int itab2 = convert_short_sat_rte( taby*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab3 = (taby*tabx * INTER_REMAP_COEF_SCALE ) > UINT16_MAX ? UINT16_MAX :
                   static_cast<int16_t>(nearbyint(taby*tabx * INTER_REMAP_COEF_SCALE ));
        if(itab3 != itab3) itab3 = 0; 
        // int itab3 = convert_short_sat_rte( taby*tabx * INTER_REMAP_COEF_SCALE );

        int val = v0 * itab0 +  v1 * itab1 + v2 * itab2 + v3 * itab3;

        uint8_t pix = ((val + (1 << (INTER_REMAP_COEF_BITS-1))) > UINT8_MAX) ? UINT8_MAX :
                      static_cast<uint8_t>(val + (1 << (INTER_REMAP_COEF_BITS-1)));  
        if(pix != pix) pix = 0;
        // uchar pix = convert_uchar_sat((val + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS);
        dst[dst_index] = pix;
    }

}


void start_warpPerspective(uint8_t *y_cuda_d, int src_step, int src_offset,
        int src_rows, int src_cols, uint8_t *dst, int dst_step, int dst_offset,
        int dst_rows,int dst_cols, float_t * M, const size_t *work_size_y)
{ 
  dim3 gridShape = dim3 (work_size_y[0],work_size_y[1]);  
  warpPerspective<<< gridShape, 1>>>(y_cuda_d,src_step,src_offset,src_rows,
      src_cols,dst,dst_step,dst_offset,dst_rows,dst_cols,M);
   sleep(1);   // Necessary to give time to let GPU threads run !!!
}