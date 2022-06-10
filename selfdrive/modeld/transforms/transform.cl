#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f / INTER_TAB_SIZE

#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

__kernel void warpPerspective(__global const uchar * src,
                              int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * dst,
                              int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __constant float * M)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        //printf("M0 %f M1 %f M2 %f\n", M[0],M[1],M[2]);
        float X0 = M[0] * dx + M[1] * dy + M[2];
        float Y0 = M[3] * dx + M[4] * dy + M[5];
        float W = M[6] * dx + M[7] * dy + M[8];
        W = W != 0.0f ? INTER_TAB_SIZE / W : 0.0f;
        int X = rint(X0 * W), Y = rint(Y0 * W);
        /*if(dx > 250 && dx < 255 && dy > 130 && dy < 133) {
            printf("dx %d dy %d X0 %f Y0 %f W %f\n",dx,dy,X0,Y0,W);
        }*/
        

        short sx = convert_short_sat(X >> INTER_BITS);
        short sy = convert_short_sat(Y >> INTER_BITS);
        short ay = (short)(Y & (INTER_TAB_SIZE - 1));
        short ax = (short)(X & (INTER_TAB_SIZE - 1));

        /*if(dx > 250 && dx < 255 && dy > 130 && dy < 133) {
            printf("dx %d dy %d sx %d sy %d ay %d ax %d\n",dx,dy,sx,sy,ay,ax);
        }*/

        int v0 = (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) ?
            convert_int(src[mad24(sy, src_step, src_offset + sx)]) : 0;
        int v1 = (sx+1 >= 0 && sx+1 < src_cols && sy >= 0 && sy < src_rows) ?
            convert_int(src[mad24(sy, src_step, src_offset + (sx+1))]) : 0;
        int v2 = (sx >= 0 && sx < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            convert_int(src[mad24(sy+1, src_step, src_offset + sx)]) : 0;
        int v3 = (sx+1 >= 0 && sx+1 < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            convert_int(src[mad24(sy+1, src_step, src_offset + (sx+1))]) : 0;

        /*if(dx > 250 && dx < 255 && dy > 130 && dy < 133) {
            printf("dx %d dy %d v0 %d v1 %d v2 %d v3 %d\n",dx,dy,v0,v1,v2,v3);
        }*/

        float taby = 1.f/INTER_TAB_SIZE*ay;
        float tabx = 1.f/INTER_TAB_SIZE*ax;

        int dst_index = mad24(dy, dst_step, dst_offset + dx);

        /*if(dx > 250 && dx < 255 && dy > 130 && dy < 133) {
            printf("dx %d dy %d taby %f tabx %f dst_index %d\n",dx,dy,taby,tabx,dst_index);
        }*/

        int itab0 = convert_short_sat_rte( (1.0f-taby)*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab1 = convert_short_sat_rte( (1.0f-taby)*tabx * INTER_REMAP_COEF_SCALE );
        int itab2 = convert_short_sat_rte( taby*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab3 = convert_short_sat_rte( taby*tabx * INTER_REMAP_COEF_SCALE );

        /*if(dx > 250 && dx < 255 && dy > 130 && dy < 133) {
            printf("dx %d dy %d tab0 %d tab1 %d itab2 %d itab3 %d\n",dx,dy,itab0,itab1,itab2,itab3);
        }*/

        int val = v0 * itab0 +  v1 * itab1 + v2 * itab2 + v3 * itab3;

        uchar pix = convert_uchar_sat((val + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS);
        dst[dst_index] = pix;
    }
}
