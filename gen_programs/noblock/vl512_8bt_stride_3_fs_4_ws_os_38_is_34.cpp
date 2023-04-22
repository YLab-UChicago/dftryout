#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <arm_neon.h>
#include <m5ops.h>
#include <algorithm>
using namespace std;



int main (int argc, char *argv[]) {
    int height;
    int width;
    int depth;
    int filter_height;
    int filter_width;
    int num_filters;
    int padding;
    int strides;
    int h_block;
    int w_block;
    int f_block;
    int d_block;
    int curr;
    int64_t* inputs;
    int64_t* outputs;
    int64_t* filters;
    int output_depth;
    
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    depth = atoi(argv[3]);
    num_filters = atoi(argv[4]);
    filter_height = 4;
    filter_width = 4;
    padding = 3;
    strides = 3;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    int h;
    int w;
    int input_h;
    int input_w;
    
    int64x2x4_t data1;
    int64x2x4_t data2;
    int64x2x4_t input_cache_0;
    int64x2x4_t input_cache_1;
    int64x2x4_t input_cache_2;
    int64x2x4_t input_cache_3;
    int64x2x4_t input_cache_4;
    int64x2x4_t input_cache_5;
    int64x2x4_t input_cache_6;
    int64x2x4_t input_cache_7;
    int64x2x4_t input_cache_8;
    int64x2x4_t input_cache_9;
    int64x2x4_t input_cache_10;
    int64x2x4_t input_cache_11;
    int64x2x4_t input_cache_12;
    int64x2x4_t input_cache_13;
    int64x2x4_t input_cache_14;
    int64x2x4_t input_cache_15;
    int64x2x4_t input_cache_16;
    int64x2x4_t input_cache_17;
    int64x2x4_t input_cache_18;
    int64x2x4_t input_cache_19;
    int64x2x4_t input_cache_20;
    int64x2x4_t input_cache_21;
    int64x2x4_t input_cache_22;
    int64x2x4_t input_cache_23;
    int64x2x4_t input_cache_24;
    int64x2x4_t input_cache_25;
    int64x2x4_t input_cache_26;
    int64x2x4_t input_cache_27;
    int64x2x4_t input_cache_28;
    int64x2x4_t input_cache_29;
    int64x2x4_t input_cache_30;
    int64x2x4_t input_cache_31;
    int64x2x4_t input_cache_32;
    int64x2x4_t input_cache_33;
    
    int64x2x4_t output_cache_0;
    int64x2x4_t output_cache_1;
    int64x2x4_t output_cache_2;
    int64x2x4_t output_cache_3;
    int64x2x4_t output_cache_4;
    int64x2x4_t output_cache_5;
    int64x2x4_t output_cache_6;
    int64x2x4_t output_cache_7;
    int64x2x4_t output_cache_8;
    int64x2x4_t output_cache_9;
    int64x2x4_t output_cache_10;
    int64x2x4_t output_cache_11;
    int64x2x4_t output_cache_12;
    int64x2x4_t output_cache_13;
    int64x2x4_t output_cache_14;
    int64x2x4_t output_cache_15;
    int64x2x4_t output_cache_16;
    int64x2x4_t output_cache_17;
    int64x2x4_t output_cache_18;
    int64x2x4_t output_cache_19;
    int64x2x4_t output_cache_20;
    int64x2x4_t output_cache_21;
    int64x2x4_t output_cache_22;
    int64x2x4_t output_cache_23;
    int64x2x4_t output_cache_24;
    int64x2x4_t output_cache_25;
    int64x2x4_t output_cache_26;
    int64x2x4_t output_cache_27;
    int64x2x4_t output_cache_28;
    int64x2x4_t output_cache_29;
    int64x2x4_t output_cache_30;
    int64x2x4_t output_cache_31;
    int64x2x4_t output_cache_32;
    int64x2x4_t output_cache_33;
    int64x2x4_t output_cache_34;
    int64x2x4_t output_cache_35;
    int64x2x4_t output_cache_36;
    int64x2x4_t output_cache_37;
    
    m5_reset_stats(0, 0);
    
    for (int f = 0; f < num_filters; f++) {
        output_cache_0.val[0]=vdupq_n_u64(0);
        output_cache_0.val[1]=vdupq_n_u64(0);
        output_cache_0.val[2]=vdupq_n_u64(0);
        output_cache_0.val[3]=vdupq_n_u64(0);
        output_cache_1.val[0]=vdupq_n_u64(0);
        output_cache_1.val[1]=vdupq_n_u64(0);
        output_cache_1.val[2]=vdupq_n_u64(0);
        output_cache_1.val[3]=vdupq_n_u64(0);
        output_cache_2.val[0]=vdupq_n_u64(0);
        output_cache_2.val[1]=vdupq_n_u64(0);
        output_cache_2.val[2]=vdupq_n_u64(0);
        output_cache_2.val[3]=vdupq_n_u64(0);
        output_cache_3.val[0]=vdupq_n_u64(0);
        output_cache_3.val[1]=vdupq_n_u64(0);
        output_cache_3.val[2]=vdupq_n_u64(0);
        output_cache_3.val[3]=vdupq_n_u64(0);
        output_cache_4.val[0]=vdupq_n_u64(0);
        output_cache_4.val[1]=vdupq_n_u64(0);
        output_cache_4.val[2]=vdupq_n_u64(0);
        output_cache_4.val[3]=vdupq_n_u64(0);
        output_cache_5.val[0]=vdupq_n_u64(0);
        output_cache_5.val[1]=vdupq_n_u64(0);
        output_cache_5.val[2]=vdupq_n_u64(0);
        output_cache_5.val[3]=vdupq_n_u64(0);
        output_cache_6.val[0]=vdupq_n_u64(0);
        output_cache_6.val[1]=vdupq_n_u64(0);
        output_cache_6.val[2]=vdupq_n_u64(0);
        output_cache_6.val[3]=vdupq_n_u64(0);
        output_cache_7.val[0]=vdupq_n_u64(0);
        output_cache_7.val[1]=vdupq_n_u64(0);
        output_cache_7.val[2]=vdupq_n_u64(0);
        output_cache_7.val[3]=vdupq_n_u64(0);
        output_cache_8.val[0]=vdupq_n_u64(0);
        output_cache_8.val[1]=vdupq_n_u64(0);
        output_cache_8.val[2]=vdupq_n_u64(0);
        output_cache_8.val[3]=vdupq_n_u64(0);
        output_cache_9.val[0]=vdupq_n_u64(0);
        output_cache_9.val[1]=vdupq_n_u64(0);
        output_cache_9.val[2]=vdupq_n_u64(0);
        output_cache_9.val[3]=vdupq_n_u64(0);
        output_cache_10.val[0]=vdupq_n_u64(0);
        output_cache_10.val[1]=vdupq_n_u64(0);
        output_cache_10.val[2]=vdupq_n_u64(0);
        output_cache_10.val[3]=vdupq_n_u64(0);
        output_cache_11.val[0]=vdupq_n_u64(0);
        output_cache_11.val[1]=vdupq_n_u64(0);
        output_cache_11.val[2]=vdupq_n_u64(0);
        output_cache_11.val[3]=vdupq_n_u64(0);
        output_cache_12.val[0]=vdupq_n_u64(0);
        output_cache_12.val[1]=vdupq_n_u64(0);
        output_cache_12.val[2]=vdupq_n_u64(0);
        output_cache_12.val[3]=vdupq_n_u64(0);
        output_cache_13.val[0]=vdupq_n_u64(0);
        output_cache_13.val[1]=vdupq_n_u64(0);
        output_cache_13.val[2]=vdupq_n_u64(0);
        output_cache_13.val[3]=vdupq_n_u64(0);
        output_cache_14.val[0]=vdupq_n_u64(0);
        output_cache_14.val[1]=vdupq_n_u64(0);
        output_cache_14.val[2]=vdupq_n_u64(0);
        output_cache_14.val[3]=vdupq_n_u64(0);
        output_cache_15.val[0]=vdupq_n_u64(0);
        output_cache_15.val[1]=vdupq_n_u64(0);
        output_cache_15.val[2]=vdupq_n_u64(0);
        output_cache_15.val[3]=vdupq_n_u64(0);
        output_cache_16.val[0]=vdupq_n_u64(0);
        output_cache_16.val[1]=vdupq_n_u64(0);
        output_cache_16.val[2]=vdupq_n_u64(0);
        output_cache_16.val[3]=vdupq_n_u64(0);
        output_cache_17.val[0]=vdupq_n_u64(0);
        output_cache_17.val[1]=vdupq_n_u64(0);
        output_cache_17.val[2]=vdupq_n_u64(0);
        output_cache_17.val[3]=vdupq_n_u64(0);
        output_cache_18.val[0]=vdupq_n_u64(0);
        output_cache_18.val[1]=vdupq_n_u64(0);
        output_cache_18.val[2]=vdupq_n_u64(0);
        output_cache_18.val[3]=vdupq_n_u64(0);
        output_cache_19.val[0]=vdupq_n_u64(0);
        output_cache_19.val[1]=vdupq_n_u64(0);
        output_cache_19.val[2]=vdupq_n_u64(0);
        output_cache_19.val[3]=vdupq_n_u64(0);
        output_cache_20.val[0]=vdupq_n_u64(0);
        output_cache_20.val[1]=vdupq_n_u64(0);
        output_cache_20.val[2]=vdupq_n_u64(0);
        output_cache_20.val[3]=vdupq_n_u64(0);
        output_cache_21.val[0]=vdupq_n_u64(0);
        output_cache_21.val[1]=vdupq_n_u64(0);
        output_cache_21.val[2]=vdupq_n_u64(0);
        output_cache_21.val[3]=vdupq_n_u64(0);
        output_cache_22.val[0]=vdupq_n_u64(0);
        output_cache_22.val[1]=vdupq_n_u64(0);
        output_cache_22.val[2]=vdupq_n_u64(0);
        output_cache_22.val[3]=vdupq_n_u64(0);
        output_cache_23.val[0]=vdupq_n_u64(0);
        output_cache_23.val[1]=vdupq_n_u64(0);
        output_cache_23.val[2]=vdupq_n_u64(0);
        output_cache_23.val[3]=vdupq_n_u64(0);
        output_cache_24.val[0]=vdupq_n_u64(0);
        output_cache_24.val[1]=vdupq_n_u64(0);
        output_cache_24.val[2]=vdupq_n_u64(0);
        output_cache_24.val[3]=vdupq_n_u64(0);
        output_cache_25.val[0]=vdupq_n_u64(0);
        output_cache_25.val[1]=vdupq_n_u64(0);
        output_cache_25.val[2]=vdupq_n_u64(0);
        output_cache_25.val[3]=vdupq_n_u64(0);
        output_cache_26.val[0]=vdupq_n_u64(0);
        output_cache_26.val[1]=vdupq_n_u64(0);
        output_cache_26.val[2]=vdupq_n_u64(0);
        output_cache_26.val[3]=vdupq_n_u64(0);
        output_cache_27.val[0]=vdupq_n_u64(0);
        output_cache_27.val[1]=vdupq_n_u64(0);
        output_cache_27.val[2]=vdupq_n_u64(0);
        output_cache_27.val[3]=vdupq_n_u64(0);
        output_cache_28.val[0]=vdupq_n_u64(0);
        output_cache_28.val[1]=vdupq_n_u64(0);
        output_cache_28.val[2]=vdupq_n_u64(0);
        output_cache_28.val[3]=vdupq_n_u64(0);
        output_cache_29.val[0]=vdupq_n_u64(0);
        output_cache_29.val[1]=vdupq_n_u64(0);
        output_cache_29.val[2]=vdupq_n_u64(0);
        output_cache_29.val[3]=vdupq_n_u64(0);
        output_cache_30.val[0]=vdupq_n_u64(0);
        output_cache_30.val[1]=vdupq_n_u64(0);
        output_cache_30.val[2]=vdupq_n_u64(0);
        output_cache_30.val[3]=vdupq_n_u64(0);
        output_cache_31.val[0]=vdupq_n_u64(0);
        output_cache_31.val[1]=vdupq_n_u64(0);
        output_cache_31.val[2]=vdupq_n_u64(0);
        output_cache_31.val[3]=vdupq_n_u64(0);
        output_cache_32.val[0]=vdupq_n_u64(0);
        output_cache_32.val[1]=vdupq_n_u64(0);
        output_cache_32.val[2]=vdupq_n_u64(0);
        output_cache_32.val[3]=vdupq_n_u64(0);
        output_cache_33.val[0]=vdupq_n_u64(0);
        output_cache_33.val[1]=vdupq_n_u64(0);
        output_cache_33.val[2]=vdupq_n_u64(0);
        output_cache_33.val[3]=vdupq_n_u64(0);
        output_cache_34.val[0]=vdupq_n_u64(0);
        output_cache_34.val[1]=vdupq_n_u64(0);
        output_cache_34.val[2]=vdupq_n_u64(0);
        output_cache_34.val[3]=vdupq_n_u64(0);
        output_cache_35.val[0]=vdupq_n_u64(0);
        output_cache_35.val[1]=vdupq_n_u64(0);
        output_cache_35.val[2]=vdupq_n_u64(0);
        output_cache_35.val[3]=vdupq_n_u64(0);
        output_cache_36.val[0]=vdupq_n_u64(0);
        output_cache_36.val[1]=vdupq_n_u64(0);
        output_cache_36.val[2]=vdupq_n_u64(0);
        output_cache_36.val[3]=vdupq_n_u64(0);
        output_cache_37.val[0]=vdupq_n_u64(0);
        output_cache_37.val[1]=vdupq_n_u64(0);
        output_cache_37.val[2]=vdupq_n_u64(0);
        output_cache_37.val[3]=vdupq_n_u64(0);
        int64x2x4_t input_cache_0 = vld1q_s64_x4((const int64_t *) &inputs[((0-padding) * width * depth /256 + (0-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_1 = vld1q_s64_x4((const int64_t *) &inputs[((0-padding) * width * depth /256 + (1-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_2 = vld1q_s64_x4((const int64_t *) &inputs[((0-padding) * width * depth /256 + (2-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_3 = vld1q_s64_x4((const int64_t *) &inputs[((0-padding) * width * depth /256 + (3-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_4 = vld1q_s64_x4((const int64_t *) &inputs[((1-padding) * width * depth /256 + (0-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_5 = vld1q_s64_x4((const int64_t *) &inputs[((1-padding) * width * depth /256 + (1-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_6 = vld1q_s64_x4((const int64_t *) &inputs[((1-padding) * width * depth /256 + (2-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_7 = vld1q_s64_x4((const int64_t *) &inputs[((1-padding) * width * depth /256 + (3-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_8 = vld1q_s64_x4((const int64_t *) &inputs[((2-padding) * width * depth /256 + (0-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_9 = vld1q_s64_x4((const int64_t *) &inputs[((2-padding) * width * depth /256 + (1-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_10 = vld1q_s64_x4((const int64_t *) &inputs[((2-padding) * width * depth /256 + (2-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_11 = vld1q_s64_x4((const int64_t *) &inputs[((2-padding) * width * depth /256 + (3-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_12 = vld1q_s64_x4((const int64_t *) &inputs[((3-padding) * width * depth /256 + (0-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_13 = vld1q_s64_x4((const int64_t *) &inputs[((3-padding) * width * depth /256 + (1-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_14 = vld1q_s64_x4((const int64_t *) &inputs[((3-padding) * width * depth /256 + (2-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_15 = vld1q_s64_x4((const int64_t *) &inputs[((3-padding) * width * depth /256 + (3-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_16 = vld1q_s64_x4((const int64_t *) &inputs[((4-padding) * width * depth /256 + (0-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_17 = vld1q_s64_x4((const int64_t *) &inputs[((4-padding) * width * depth /256 + (1-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_18 = vld1q_s64_x4((const int64_t *) &inputs[((4-padding) * width * depth /256 + (2-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_19 = vld1q_s64_x4((const int64_t *) &inputs[((4-padding) * width * depth /256 + (3-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_20 = vld1q_s64_x4((const int64_t *) &inputs[((5-padding) * width * depth /256 + (0-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_21 = vld1q_s64_x4((const int64_t *) &inputs[((5-padding) * width * depth /256 + (1-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_22 = vld1q_s64_x4((const int64_t *) &inputs[((5-padding) * width * depth /256 + (2-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_23 = vld1q_s64_x4((const int64_t *) &inputs[((5-padding) * width * depth /256 + (3-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_24 = vld1q_s64_x4((const int64_t *) &inputs[((6-padding) * width * depth /256 + (0-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_25 = vld1q_s64_x4((const int64_t *) &inputs[((6-padding) * width * depth /256 + (1-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_26 = vld1q_s64_x4((const int64_t *) &inputs[((6-padding) * width * depth /256 + (2-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_27 = vld1q_s64_x4((const int64_t *) &inputs[((6-padding) * width * depth /256 + (3-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_28 = vld1q_s64_x4((const int64_t *) &inputs[((7-padding) * width * depth /256 + (0-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_29 = vld1q_s64_x4((const int64_t *) &inputs[((7-padding) * width * depth /256 + (1-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_30 = vld1q_s64_x4((const int64_t *) &inputs[((7-padding) * width * depth /256 + (2-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_31 = vld1q_s64_x4((const int64_t *) &inputs[((7-padding) * width * depth /256 + (3-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_32 = vld1q_s64_x4((const int64_t *) &inputs[((8-padding) * width * depth /256 + (0-padding) * depth /256) * 512 /64]);
        int64x2x4_t input_cache_33 = vld1q_s64_x4((const int64_t *) &inputs[((8-padding) * width * depth /256 + (1-padding) * depth /256) * 512 /64]);
        int i;
        int j;
        for (i = 0; i < filter_height - 1; i ++) {
            for (j = 0; j < filter_width; j ++) {
                h = 0;
                w = 0;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
                
                
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_0.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_0.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_0.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_0.val[3],data2.val[3]);
                output_cache_0.val[0] = vaddq_u8(output_cache_0.val[0],data1.val[0]);
                output_cache_0.val[1] = vaddq_u8(output_cache_0.val[1],data1.val[1]);
                output_cache_0.val[2] = vaddq_u8(output_cache_0.val[2],data1.val[2]);
                output_cache_0.val[3] = vaddq_u8(output_cache_0.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_1.val[3],data2.val[3]);
                output_cache_1.val[0] = vaddq_u8(output_cache_1.val[0],data1.val[0]);
                output_cache_1.val[1] = vaddq_u8(output_cache_1.val[1],data1.val[1]);
                output_cache_1.val[2] = vaddq_u8(output_cache_1.val[2],data1.val[2]);
                output_cache_1.val[3] = vaddq_u8(output_cache_1.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_2.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_2.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_2.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_2.val[3],data2.val[3]);
                output_cache_2.val[0] = vaddq_u8(output_cache_2.val[0],data1.val[0]);
                output_cache_2.val[1] = vaddq_u8(output_cache_2.val[1],data1.val[1]);
                output_cache_2.val[2] = vaddq_u8(output_cache_2.val[2],data1.val[2]);
                output_cache_2.val[3] = vaddq_u8(output_cache_2.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_3.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_3.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_3.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_3.val[3],data2.val[3]);
                output_cache_3.val[0] = vaddq_u8(output_cache_3.val[0],data1.val[0]);
                output_cache_3.val[1] = vaddq_u8(output_cache_3.val[1],data1.val[1]);
                output_cache_3.val[2] = vaddq_u8(output_cache_3.val[2],data1.val[2]);
                output_cache_3.val[3] = vaddq_u8(output_cache_3.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_4.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_4.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_4.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_4.val[3],data2.val[3]);
                output_cache_4.val[0] = vaddq_u8(output_cache_4.val[0],data1.val[0]);
                output_cache_4.val[1] = vaddq_u8(output_cache_4.val[1],data1.val[1]);
                output_cache_4.val[2] = vaddq_u8(output_cache_4.val[2],data1.val[2]);
                output_cache_4.val[3] = vaddq_u8(output_cache_4.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_5.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_5.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_5.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_5.val[3],data2.val[3]);
                output_cache_5.val[0] = vaddq_u8(output_cache_5.val[0],data1.val[0]);
                output_cache_5.val[1] = vaddq_u8(output_cache_5.val[1],data1.val[1]);
                output_cache_5.val[2] = vaddq_u8(output_cache_5.val[2],data1.val[2]);
                output_cache_5.val[3] = vaddq_u8(output_cache_5.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_6.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_6.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_6.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_6.val[3],data2.val[3]);
                output_cache_6.val[0] = vaddq_u8(output_cache_6.val[0],data1.val[0]);
                output_cache_6.val[1] = vaddq_u8(output_cache_6.val[1],data1.val[1]);
                output_cache_6.val[2] = vaddq_u8(output_cache_6.val[2],data1.val[2]);
                output_cache_6.val[3] = vaddq_u8(output_cache_6.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_7.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_7.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_7.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_7.val[3],data2.val[3]);
                output_cache_7.val[0] = vaddq_u8(output_cache_7.val[0],data1.val[0]);
                output_cache_7.val[1] = vaddq_u8(output_cache_7.val[1],data1.val[1]);
                output_cache_7.val[2] = vaddq_u8(output_cache_7.val[2],data1.val[2]);
                output_cache_7.val[3] = vaddq_u8(output_cache_7.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_8.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_8.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_8.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_8.val[3],data2.val[3]);
                output_cache_8.val[0] = vaddq_u8(output_cache_8.val[0],data1.val[0]);
                output_cache_8.val[1] = vaddq_u8(output_cache_8.val[1],data1.val[1]);
                output_cache_8.val[2] = vaddq_u8(output_cache_8.val[2],data1.val[2]);
                output_cache_8.val[3] = vaddq_u8(output_cache_8.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_9.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_9.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_9.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_9.val[3],data2.val[3]);
                output_cache_9.val[0] = vaddq_u8(output_cache_9.val[0],data1.val[0]);
                output_cache_9.val[1] = vaddq_u8(output_cache_9.val[1],data1.val[1]);
                output_cache_9.val[2] = vaddq_u8(output_cache_9.val[2],data1.val[2]);
                output_cache_9.val[3] = vaddq_u8(output_cache_9.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_10.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_10.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_10.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_10.val[3],data2.val[3]);
                output_cache_10.val[0] = vaddq_u8(output_cache_10.val[0],data1.val[0]);
                output_cache_10.val[1] = vaddq_u8(output_cache_10.val[1],data1.val[1]);
                output_cache_10.val[2] = vaddq_u8(output_cache_10.val[2],data1.val[2]);
                output_cache_10.val[3] = vaddq_u8(output_cache_10.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_11.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_11.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_11.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_11.val[3],data2.val[3]);
                output_cache_11.val[0] = vaddq_u8(output_cache_11.val[0],data1.val[0]);
                output_cache_11.val[1] = vaddq_u8(output_cache_11.val[1],data1.val[1]);
                output_cache_11.val[2] = vaddq_u8(output_cache_11.val[2],data1.val[2]);
                output_cache_11.val[3] = vaddq_u8(output_cache_11.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_12.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_12.val[3],data2.val[3]);
                output_cache_12.val[0] = vaddq_u8(output_cache_12.val[0],data1.val[0]);
                output_cache_12.val[1] = vaddq_u8(output_cache_12.val[1],data1.val[1]);
                output_cache_12.val[2] = vaddq_u8(output_cache_12.val[2],data1.val[2]);
                output_cache_12.val[3] = vaddq_u8(output_cache_12.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_13.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_13.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_13.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_13.val[3],data2.val[3]);
                output_cache_13.val[0] = vaddq_u8(output_cache_13.val[0],data1.val[0]);
                output_cache_13.val[1] = vaddq_u8(output_cache_13.val[1],data1.val[1]);
                output_cache_13.val[2] = vaddq_u8(output_cache_13.val[2],data1.val[2]);
                output_cache_13.val[3] = vaddq_u8(output_cache_13.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_14.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_14.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_14.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_14.val[3],data2.val[3]);
                output_cache_14.val[0] = vaddq_u8(output_cache_14.val[0],data1.val[0]);
                output_cache_14.val[1] = vaddq_u8(output_cache_14.val[1],data1.val[1]);
                output_cache_14.val[2] = vaddq_u8(output_cache_14.val[2],data1.val[2]);
                output_cache_14.val[3] = vaddq_u8(output_cache_14.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_15.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_15.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_15.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_15.val[3],data2.val[3]);
                output_cache_15.val[0] = vaddq_u8(output_cache_15.val[0],data1.val[0]);
                output_cache_15.val[1] = vaddq_u8(output_cache_15.val[1],data1.val[1]);
                output_cache_15.val[2] = vaddq_u8(output_cache_15.val[2],data1.val[2]);
                output_cache_15.val[3] = vaddq_u8(output_cache_15.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_16.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_16.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_16.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_16.val[3],data2.val[3]);
                output_cache_16.val[0] = vaddq_u8(output_cache_16.val[0],data1.val[0]);
                output_cache_16.val[1] = vaddq_u8(output_cache_16.val[1],data1.val[1]);
                output_cache_16.val[2] = vaddq_u8(output_cache_16.val[2],data1.val[2]);
                output_cache_16.val[3] = vaddq_u8(output_cache_16.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_17.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_17.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_17.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_17.val[3],data2.val[3]);
                output_cache_17.val[0] = vaddq_u8(output_cache_17.val[0],data1.val[0]);
                output_cache_17.val[1] = vaddq_u8(output_cache_17.val[1],data1.val[1]);
                output_cache_17.val[2] = vaddq_u8(output_cache_17.val[2],data1.val[2]);
                output_cache_17.val[3] = vaddq_u8(output_cache_17.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_18.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_18.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_18.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_18.val[3],data2.val[3]);
                output_cache_18.val[0] = vaddq_u8(output_cache_18.val[0],data1.val[0]);
                output_cache_18.val[1] = vaddq_u8(output_cache_18.val[1],data1.val[1]);
                output_cache_18.val[2] = vaddq_u8(output_cache_18.val[2],data1.val[2]);
                output_cache_18.val[3] = vaddq_u8(output_cache_18.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_19.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_19.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_19.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_19.val[3],data2.val[3]);
                output_cache_19.val[0] = vaddq_u8(output_cache_19.val[0],data1.val[0]);
                output_cache_19.val[1] = vaddq_u8(output_cache_19.val[1],data1.val[1]);
                output_cache_19.val[2] = vaddq_u8(output_cache_19.val[2],data1.val[2]);
                output_cache_19.val[3] = vaddq_u8(output_cache_19.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_20.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_20.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_20.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_20.val[3],data2.val[3]);
                output_cache_20.val[0] = vaddq_u8(output_cache_20.val[0],data1.val[0]);
                output_cache_20.val[1] = vaddq_u8(output_cache_20.val[1],data1.val[1]);
                output_cache_20.val[2] = vaddq_u8(output_cache_20.val[2],data1.val[2]);
                output_cache_20.val[3] = vaddq_u8(output_cache_20.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_21.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_21.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_21.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_21.val[3],data2.val[3]);
                output_cache_21.val[0] = vaddq_u8(output_cache_21.val[0],data1.val[0]);
                output_cache_21.val[1] = vaddq_u8(output_cache_21.val[1],data1.val[1]);
                output_cache_21.val[2] = vaddq_u8(output_cache_21.val[2],data1.val[2]);
                output_cache_21.val[3] = vaddq_u8(output_cache_21.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_22.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_22.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_22.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_22.val[3],data2.val[3]);
                output_cache_22.val[0] = vaddq_u8(output_cache_22.val[0],data1.val[0]);
                output_cache_22.val[1] = vaddq_u8(output_cache_22.val[1],data1.val[1]);
                output_cache_22.val[2] = vaddq_u8(output_cache_22.val[2],data1.val[2]);
                output_cache_22.val[3] = vaddq_u8(output_cache_22.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_23.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_23.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_23.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_23.val[3],data2.val[3]);
                output_cache_23.val[0] = vaddq_u8(output_cache_23.val[0],data1.val[0]);
                output_cache_23.val[1] = vaddq_u8(output_cache_23.val[1],data1.val[1]);
                output_cache_23.val[2] = vaddq_u8(output_cache_23.val[2],data1.val[2]);
                output_cache_23.val[3] = vaddq_u8(output_cache_23.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_24.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_24.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_24.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_24.val[3],data2.val[3]);
                output_cache_24.val[0] = vaddq_u8(output_cache_24.val[0],data1.val[0]);
                output_cache_24.val[1] = vaddq_u8(output_cache_24.val[1],data1.val[1]);
                output_cache_24.val[2] = vaddq_u8(output_cache_24.val[2],data1.val[2]);
                output_cache_24.val[3] = vaddq_u8(output_cache_24.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_25.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_25.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_25.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_25.val[3],data2.val[3]);
                output_cache_25.val[0] = vaddq_u8(output_cache_25.val[0],data1.val[0]);
                output_cache_25.val[1] = vaddq_u8(output_cache_25.val[1],data1.val[1]);
                output_cache_25.val[2] = vaddq_u8(output_cache_25.val[2],data1.val[2]);
                output_cache_25.val[3] = vaddq_u8(output_cache_25.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_26.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_26.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_26.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_26.val[3],data2.val[3]);
                output_cache_26.val[0] = vaddq_u8(output_cache_26.val[0],data1.val[0]);
                output_cache_26.val[1] = vaddq_u8(output_cache_26.val[1],data1.val[1]);
                output_cache_26.val[2] = vaddq_u8(output_cache_26.val[2],data1.val[2]);
                output_cache_26.val[3] = vaddq_u8(output_cache_26.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_27.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_27.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_27.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_27.val[3],data2.val[3]);
                output_cache_27.val[0] = vaddq_u8(output_cache_27.val[0],data1.val[0]);
                output_cache_27.val[1] = vaddq_u8(output_cache_27.val[1],data1.val[1]);
                output_cache_27.val[2] = vaddq_u8(output_cache_27.val[2],data1.val[2]);
                output_cache_27.val[3] = vaddq_u8(output_cache_27.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_28.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_28.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_28.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_28.val[3],data2.val[3]);
                output_cache_28.val[0] = vaddq_u8(output_cache_28.val[0],data1.val[0]);
                output_cache_28.val[1] = vaddq_u8(output_cache_28.val[1],data1.val[1]);
                output_cache_28.val[2] = vaddq_u8(output_cache_28.val[2],data1.val[2]);
                output_cache_28.val[3] = vaddq_u8(output_cache_28.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_29.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_29.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_29.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_29.val[3],data2.val[3]);
                output_cache_29.val[0] = vaddq_u8(output_cache_29.val[0],data1.val[0]);
                output_cache_29.val[1] = vaddq_u8(output_cache_29.val[1],data1.val[1]);
                output_cache_29.val[2] = vaddq_u8(output_cache_29.val[2],data1.val[2]);
                output_cache_29.val[3] = vaddq_u8(output_cache_29.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_30.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_30.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_30.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_30.val[3],data2.val[3]);
                output_cache_30.val[0] = vaddq_u8(output_cache_30.val[0],data1.val[0]);
                output_cache_30.val[1] = vaddq_u8(output_cache_30.val[1],data1.val[1]);
                output_cache_30.val[2] = vaddq_u8(output_cache_30.val[2],data1.val[2]);
                output_cache_30.val[3] = vaddq_u8(output_cache_30.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_31.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_31.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_31.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_31.val[3],data2.val[3]);
                output_cache_31.val[0] = vaddq_u8(output_cache_31.val[0],data1.val[0]);
                output_cache_31.val[1] = vaddq_u8(output_cache_31.val[1],data1.val[1]);
                output_cache_31.val[2] = vaddq_u8(output_cache_31.val[2],data1.val[2]);
                output_cache_31.val[3] = vaddq_u8(output_cache_31.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_32.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_32.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_32.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_32.val[3],data2.val[3]);
                output_cache_32.val[0] = vaddq_u8(output_cache_32.val[0],data1.val[0]);
                output_cache_32.val[1] = vaddq_u8(output_cache_32.val[1],data1.val[1]);
                output_cache_32.val[2] = vaddq_u8(output_cache_32.val[2],data1.val[2]);
                output_cache_32.val[3] = vaddq_u8(output_cache_32.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1.val[0] = vmulq_s8(input_cache_33.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(input_cache_33.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(input_cache_33.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(input_cache_33.val[3],data2.val[3]);
                output_cache_33.val[0] = vaddq_u8(output_cache_33.val[0],data1.val[0]);
                output_cache_33.val[1] = vaddq_u8(output_cache_33.val[1],data1.val[1]);
                output_cache_33.val[2] = vaddq_u8(output_cache_33.val[2],data1.val[2]);
                output_cache_33.val[3] = vaddq_u8(output_cache_33.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output_cache_34.val[0] = vaddq_u8(output_cache_34.val[0],data1.val[0]);
                output_cache_34.val[1] = vaddq_u8(output_cache_34.val[1],data1.val[1]);
                output_cache_34.val[2] = vaddq_u8(output_cache_34.val[2],data1.val[2]);
                output_cache_34.val[3] = vaddq_u8(output_cache_34.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output_cache_35.val[0] = vaddq_u8(output_cache_35.val[0],data1.val[0]);
                output_cache_35.val[1] = vaddq_u8(output_cache_35.val[1],data1.val[1]);
                output_cache_35.val[2] = vaddq_u8(output_cache_35.val[2],data1.val[2]);
                output_cache_35.val[3] = vaddq_u8(output_cache_35.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output_cache_36.val[0] = vaddq_u8(output_cache_36.val[0],data1.val[0]);
                output_cache_36.val[1] = vaddq_u8(output_cache_36.val[1],data1.val[1]);
                output_cache_36.val[2] = vaddq_u8(output_cache_36.val[2],data1.val[2]);
                output_cache_36.val[3] = vaddq_u8(output_cache_36.val[3],data1.val[3]);
                
                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                output_cache_37.val[0] = vaddq_u8(output_cache_37.val[0],data1.val[0]);
                output_cache_37.val[1] = vaddq_u8(output_cache_37.val[1],data1.val[1]);
                output_cache_37.val[2] = vaddq_u8(output_cache_37.val[2],data1.val[2]);
                output_cache_37.val[3] = vaddq_u8(output_cache_37.val[3],data1.val[3]);
                for (h = 0; h < out_height; h++) {
                    for (w = 38; w < out_width; w++) {
                        input_h = h * strides + i - padding;
                        input_w = w * strides + j - padding;
                        data1 = vld1q_s64_x4((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * 512 /64]);
                        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                        data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                        data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                        data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    }
                }
            }
        }
        
        
        for (j = 0; j < filter_width - 1; j ++) {
            data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
            
            
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_0.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_0.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_0.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_0.val[3], data2.val[3]);
            output_cache_0.val[0] = vaddq_u8(output_cache_0.val[0],data1.val[0]);
            output_cache_0.val[1] = vaddq_u8(output_cache_0.val[1],data1.val[1]);
            output_cache_0.val[2] = vaddq_u8(output_cache_0.val[2],data1.val[2]);
            output_cache_0.val[3] = vaddq_u8(output_cache_0.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_1.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_1.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_1.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_1.val[3], data2.val[3]);
            output_cache_1.val[0] = vaddq_u8(output_cache_1.val[0],data1.val[0]);
            output_cache_1.val[1] = vaddq_u8(output_cache_1.val[1],data1.val[1]);
            output_cache_1.val[2] = vaddq_u8(output_cache_1.val[2],data1.val[2]);
            output_cache_1.val[3] = vaddq_u8(output_cache_1.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_2.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_2.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_2.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_2.val[3], data2.val[3]);
            output_cache_2.val[0] = vaddq_u8(output_cache_2.val[0],data1.val[0]);
            output_cache_2.val[1] = vaddq_u8(output_cache_2.val[1],data1.val[1]);
            output_cache_2.val[2] = vaddq_u8(output_cache_2.val[2],data1.val[2]);
            output_cache_2.val[3] = vaddq_u8(output_cache_2.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_3.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_3.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_3.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_3.val[3], data2.val[3]);
            output_cache_3.val[0] = vaddq_u8(output_cache_3.val[0],data1.val[0]);
            output_cache_3.val[1] = vaddq_u8(output_cache_3.val[1],data1.val[1]);
            output_cache_3.val[2] = vaddq_u8(output_cache_3.val[2],data1.val[2]);
            output_cache_3.val[3] = vaddq_u8(output_cache_3.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_4.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_4.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_4.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_4.val[3], data2.val[3]);
            output_cache_4.val[0] = vaddq_u8(output_cache_4.val[0],data1.val[0]);
            output_cache_4.val[1] = vaddq_u8(output_cache_4.val[1],data1.val[1]);
            output_cache_4.val[2] = vaddq_u8(output_cache_4.val[2],data1.val[2]);
            output_cache_4.val[3] = vaddq_u8(output_cache_4.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_5.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_5.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_5.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_5.val[3], data2.val[3]);
            output_cache_5.val[0] = vaddq_u8(output_cache_5.val[0],data1.val[0]);
            output_cache_5.val[1] = vaddq_u8(output_cache_5.val[1],data1.val[1]);
            output_cache_5.val[2] = vaddq_u8(output_cache_5.val[2],data1.val[2]);
            output_cache_5.val[3] = vaddq_u8(output_cache_5.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_6.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_6.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_6.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_6.val[3], data2.val[3]);
            output_cache_6.val[0] = vaddq_u8(output_cache_6.val[0],data1.val[0]);
            output_cache_6.val[1] = vaddq_u8(output_cache_6.val[1],data1.val[1]);
            output_cache_6.val[2] = vaddq_u8(output_cache_6.val[2],data1.val[2]);
            output_cache_6.val[3] = vaddq_u8(output_cache_6.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_7.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_7.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_7.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_7.val[3], data2.val[3]);
            output_cache_7.val[0] = vaddq_u8(output_cache_7.val[0],data1.val[0]);
            output_cache_7.val[1] = vaddq_u8(output_cache_7.val[1],data1.val[1]);
            output_cache_7.val[2] = vaddq_u8(output_cache_7.val[2],data1.val[2]);
            output_cache_7.val[3] = vaddq_u8(output_cache_7.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_8.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_8.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_8.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_8.val[3], data2.val[3]);
            output_cache_8.val[0] = vaddq_u8(output_cache_8.val[0],data1.val[0]);
            output_cache_8.val[1] = vaddq_u8(output_cache_8.val[1],data1.val[1]);
            output_cache_8.val[2] = vaddq_u8(output_cache_8.val[2],data1.val[2]);
            output_cache_8.val[3] = vaddq_u8(output_cache_8.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_9.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_9.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_9.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_9.val[3], data2.val[3]);
            output_cache_9.val[0] = vaddq_u8(output_cache_9.val[0],data1.val[0]);
            output_cache_9.val[1] = vaddq_u8(output_cache_9.val[1],data1.val[1]);
            output_cache_9.val[2] = vaddq_u8(output_cache_9.val[2],data1.val[2]);
            output_cache_9.val[3] = vaddq_u8(output_cache_9.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_10.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_10.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_10.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_10.val[3], data2.val[3]);
            output_cache_10.val[0] = vaddq_u8(output_cache_10.val[0],data1.val[0]);
            output_cache_10.val[1] = vaddq_u8(output_cache_10.val[1],data1.val[1]);
            output_cache_10.val[2] = vaddq_u8(output_cache_10.val[2],data1.val[2]);
            output_cache_10.val[3] = vaddq_u8(output_cache_10.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_11.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_11.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_11.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_11.val[3], data2.val[3]);
            output_cache_11.val[0] = vaddq_u8(output_cache_11.val[0],data1.val[0]);
            output_cache_11.val[1] = vaddq_u8(output_cache_11.val[1],data1.val[1]);
            output_cache_11.val[2] = vaddq_u8(output_cache_11.val[2],data1.val[2]);
            output_cache_11.val[3] = vaddq_u8(output_cache_11.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_12.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_12.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_12.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_12.val[3], data2.val[3]);
            output_cache_12.val[0] = vaddq_u8(output_cache_12.val[0],data1.val[0]);
            output_cache_12.val[1] = vaddq_u8(output_cache_12.val[1],data1.val[1]);
            output_cache_12.val[2] = vaddq_u8(output_cache_12.val[2],data1.val[2]);
            output_cache_12.val[3] = vaddq_u8(output_cache_12.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_13.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_13.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_13.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_13.val[3], data2.val[3]);
            output_cache_13.val[0] = vaddq_u8(output_cache_13.val[0],data1.val[0]);
            output_cache_13.val[1] = vaddq_u8(output_cache_13.val[1],data1.val[1]);
            output_cache_13.val[2] = vaddq_u8(output_cache_13.val[2],data1.val[2]);
            output_cache_13.val[3] = vaddq_u8(output_cache_13.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_14.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_14.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_14.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_14.val[3], data2.val[3]);
            output_cache_14.val[0] = vaddq_u8(output_cache_14.val[0],data1.val[0]);
            output_cache_14.val[1] = vaddq_u8(output_cache_14.val[1],data1.val[1]);
            output_cache_14.val[2] = vaddq_u8(output_cache_14.val[2],data1.val[2]);
            output_cache_14.val[3] = vaddq_u8(output_cache_14.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_15.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_15.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_15.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_15.val[3], data2.val[3]);
            output_cache_15.val[0] = vaddq_u8(output_cache_15.val[0],data1.val[0]);
            output_cache_15.val[1] = vaddq_u8(output_cache_15.val[1],data1.val[1]);
            output_cache_15.val[2] = vaddq_u8(output_cache_15.val[2],data1.val[2]);
            output_cache_15.val[3] = vaddq_u8(output_cache_15.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_16.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_16.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_16.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_16.val[3], data2.val[3]);
            output_cache_16.val[0] = vaddq_u8(output_cache_16.val[0],data1.val[0]);
            output_cache_16.val[1] = vaddq_u8(output_cache_16.val[1],data1.val[1]);
            output_cache_16.val[2] = vaddq_u8(output_cache_16.val[2],data1.val[2]);
            output_cache_16.val[3] = vaddq_u8(output_cache_16.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_17.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_17.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_17.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_17.val[3], data2.val[3]);
            output_cache_17.val[0] = vaddq_u8(output_cache_17.val[0],data1.val[0]);
            output_cache_17.val[1] = vaddq_u8(output_cache_17.val[1],data1.val[1]);
            output_cache_17.val[2] = vaddq_u8(output_cache_17.val[2],data1.val[2]);
            output_cache_17.val[3] = vaddq_u8(output_cache_17.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_18.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_18.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_18.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_18.val[3], data2.val[3]);
            output_cache_18.val[0] = vaddq_u8(output_cache_18.val[0],data1.val[0]);
            output_cache_18.val[1] = vaddq_u8(output_cache_18.val[1],data1.val[1]);
            output_cache_18.val[2] = vaddq_u8(output_cache_18.val[2],data1.val[2]);
            output_cache_18.val[3] = vaddq_u8(output_cache_18.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_19.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_19.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_19.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_19.val[3], data2.val[3]);
            output_cache_19.val[0] = vaddq_u8(output_cache_19.val[0],data1.val[0]);
            output_cache_19.val[1] = vaddq_u8(output_cache_19.val[1],data1.val[1]);
            output_cache_19.val[2] = vaddq_u8(output_cache_19.val[2],data1.val[2]);
            output_cache_19.val[3] = vaddq_u8(output_cache_19.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_20.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_20.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_20.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_20.val[3], data2.val[3]);
            output_cache_20.val[0] = vaddq_u8(output_cache_20.val[0],data1.val[0]);
            output_cache_20.val[1] = vaddq_u8(output_cache_20.val[1],data1.val[1]);
            output_cache_20.val[2] = vaddq_u8(output_cache_20.val[2],data1.val[2]);
            output_cache_20.val[3] = vaddq_u8(output_cache_20.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_21.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_21.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_21.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_21.val[3], data2.val[3]);
            output_cache_21.val[0] = vaddq_u8(output_cache_21.val[0],data1.val[0]);
            output_cache_21.val[1] = vaddq_u8(output_cache_21.val[1],data1.val[1]);
            output_cache_21.val[2] = vaddq_u8(output_cache_21.val[2],data1.val[2]);
            output_cache_21.val[3] = vaddq_u8(output_cache_21.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_22.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_22.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_22.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_22.val[3], data2.val[3]);
            output_cache_22.val[0] = vaddq_u8(output_cache_22.val[0],data1.val[0]);
            output_cache_22.val[1] = vaddq_u8(output_cache_22.val[1],data1.val[1]);
            output_cache_22.val[2] = vaddq_u8(output_cache_22.val[2],data1.val[2]);
            output_cache_22.val[3] = vaddq_u8(output_cache_22.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_23.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_23.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_23.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_23.val[3], data2.val[3]);
            output_cache_23.val[0] = vaddq_u8(output_cache_23.val[0],data1.val[0]);
            output_cache_23.val[1] = vaddq_u8(output_cache_23.val[1],data1.val[1]);
            output_cache_23.val[2] = vaddq_u8(output_cache_23.val[2],data1.val[2]);
            output_cache_23.val[3] = vaddq_u8(output_cache_23.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_24.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_24.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_24.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_24.val[3], data2.val[3]);
            output_cache_24.val[0] = vaddq_u8(output_cache_24.val[0],data1.val[0]);
            output_cache_24.val[1] = vaddq_u8(output_cache_24.val[1],data1.val[1]);
            output_cache_24.val[2] = vaddq_u8(output_cache_24.val[2],data1.val[2]);
            output_cache_24.val[3] = vaddq_u8(output_cache_24.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_25.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_25.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_25.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_25.val[3], data2.val[3]);
            output_cache_25.val[0] = vaddq_u8(output_cache_25.val[0],data1.val[0]);
            output_cache_25.val[1] = vaddq_u8(output_cache_25.val[1],data1.val[1]);
            output_cache_25.val[2] = vaddq_u8(output_cache_25.val[2],data1.val[2]);
            output_cache_25.val[3] = vaddq_u8(output_cache_25.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_26.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_26.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_26.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_26.val[3], data2.val[3]);
            output_cache_26.val[0] = vaddq_u8(output_cache_26.val[0],data1.val[0]);
            output_cache_26.val[1] = vaddq_u8(output_cache_26.val[1],data1.val[1]);
            output_cache_26.val[2] = vaddq_u8(output_cache_26.val[2],data1.val[2]);
            output_cache_26.val[3] = vaddq_u8(output_cache_26.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_27.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_27.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_27.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_27.val[3], data2.val[3]);
            output_cache_27.val[0] = vaddq_u8(output_cache_27.val[0],data1.val[0]);
            output_cache_27.val[1] = vaddq_u8(output_cache_27.val[1],data1.val[1]);
            output_cache_27.val[2] = vaddq_u8(output_cache_27.val[2],data1.val[2]);
            output_cache_27.val[3] = vaddq_u8(output_cache_27.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_28.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_28.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_28.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_28.val[3], data2.val[3]);
            output_cache_28.val[0] = vaddq_u8(output_cache_28.val[0],data1.val[0]);
            output_cache_28.val[1] = vaddq_u8(output_cache_28.val[1],data1.val[1]);
            output_cache_28.val[2] = vaddq_u8(output_cache_28.val[2],data1.val[2]);
            output_cache_28.val[3] = vaddq_u8(output_cache_28.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_29.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_29.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_29.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_29.val[3], data2.val[3]);
            output_cache_29.val[0] = vaddq_u8(output_cache_29.val[0],data1.val[0]);
            output_cache_29.val[1] = vaddq_u8(output_cache_29.val[1],data1.val[1]);
            output_cache_29.val[2] = vaddq_u8(output_cache_29.val[2],data1.val[2]);
            output_cache_29.val[3] = vaddq_u8(output_cache_29.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_30.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_30.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_30.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_30.val[3], data2.val[3]);
            output_cache_30.val[0] = vaddq_u8(output_cache_30.val[0],data1.val[0]);
            output_cache_30.val[1] = vaddq_u8(output_cache_30.val[1],data1.val[1]);
            output_cache_30.val[2] = vaddq_u8(output_cache_30.val[2],data1.val[2]);
            output_cache_30.val[3] = vaddq_u8(output_cache_30.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_31.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_31.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_31.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_31.val[3], data2.val[3]);
            output_cache_31.val[0] = vaddq_u8(output_cache_31.val[0],data1.val[0]);
            output_cache_31.val[1] = vaddq_u8(output_cache_31.val[1],data1.val[1]);
            output_cache_31.val[2] = vaddq_u8(output_cache_31.val[2],data1.val[2]);
            output_cache_31.val[3] = vaddq_u8(output_cache_31.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_32.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_32.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_32.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_32.val[3], data2.val[3]);
            output_cache_32.val[0] = vaddq_u8(output_cache_32.val[0],data1.val[0]);
            output_cache_32.val[1] = vaddq_u8(output_cache_32.val[1],data1.val[1]);
            output_cache_32.val[2] = vaddq_u8(output_cache_32.val[2],data1.val[2]);
            output_cache_32.val[3] = vaddq_u8(output_cache_32.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1.val[0] = vmulq_s8(input_cache_33.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(input_cache_33.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(input_cache_33.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(input_cache_33.val[3], data2.val[3]);
            output_cache_33.val[0] = vaddq_u8(output_cache_33.val[0],data1.val[0]);
            output_cache_33.val[1] = vaddq_u8(output_cache_33.val[1],data1.val[1]);
            output_cache_33.val[2] = vaddq_u8(output_cache_33.val[2],data1.val[2]);
            output_cache_33.val[3] = vaddq_u8(output_cache_33.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
            data1.val[0] = vmulq_s8(data1.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(data1.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(data1.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(data1.val[3], data2.val[3]);
            output_cache_34.val[0] = vaddq_u8(output_cache_34.val[0],data1.val[0]);
            output_cache_34.val[1] = vaddq_u8(output_cache_34.val[1],data1.val[1]);
            output_cache_34.val[2] = vaddq_u8(output_cache_34.val[2],data1.val[2]);
            output_cache_34.val[3] = vaddq_u8(output_cache_34.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
            data1.val[0] = vmulq_s8(data1.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(data1.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(data1.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(data1.val[3], data2.val[3]);
            output_cache_35.val[0] = vaddq_u8(output_cache_35.val[0],data1.val[0]);
            output_cache_35.val[1] = vaddq_u8(output_cache_35.val[1],data1.val[1]);
            output_cache_35.val[2] = vaddq_u8(output_cache_35.val[2],data1.val[2]);
            output_cache_35.val[3] = vaddq_u8(output_cache_35.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
            data1.val[0] = vmulq_s8(data1.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(data1.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(data1.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(data1.val[3], data2.val[3]);
            output_cache_36.val[0] = vaddq_u8(output_cache_36.val[0],data1.val[0]);
            output_cache_36.val[1] = vaddq_u8(output_cache_36.val[1],data1.val[1]);
            output_cache_36.val[2] = vaddq_u8(output_cache_36.val[2],data1.val[2]);
            output_cache_36.val[3] = vaddq_u8(output_cache_36.val[3],data1.val[3]);
            
            w++;
            input_h = h * strides + i - padding;
            input_w = w * strides + j - padding;
            data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
            data1.val[0] = vmulq_s8(data1.val[0], data2.val[0]);
            data1.val[1] = vmulq_s8(data1.val[1], data2.val[1]);
            data1.val[2] = vmulq_s8(data1.val[2], data2.val[2]);
            data1.val[3] = vmulq_s8(data1.val[3], data2.val[3]);
            output_cache_37.val[0] = vaddq_u8(output_cache_37.val[0],data1.val[0]);
            output_cache_37.val[1] = vaddq_u8(output_cache_37.val[1],data1.val[1]);
            output_cache_37.val[2] = vaddq_u8(output_cache_37.val[2],data1.val[2]);
            output_cache_37.val[3] = vaddq_u8(output_cache_37.val[3],data1.val[3]);
            
            for (h = 0; h < out_height; h++) {
                for (w = 38; w < out_width; w++) {
                    input_h = h * strides + i - padding;
                    input_w = w * strides + j - padding;
                    data1 = vld1q_s64_x4((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * 512 /64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                    outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                }
            }
        }
        data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
        
        
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_0.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_0.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_0.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_0.val[3],data2.val[3]);
        output_cache_0.val[0] = vaddq_u8(output_cache_0.val[0],data1.val[0]);
        output_cache_0.val[1] = vaddq_u8(output_cache_0.val[1],data1.val[1]);
        output_cache_0.val[2] = vaddq_u8(output_cache_0.val[2],data1.val[2]);
        output_cache_0.val[3] = vaddq_u8(output_cache_0.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_0.val[0])+vaddvq_u8(output_cache_0.val[1])+vaddvq_u8(output_cache_0.val[2])+vaddvq_u8(output_cache_0.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_1.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_1.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_1.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_1.val[3],data2.val[3]);
        output_cache_1.val[0] = vaddq_u8(output_cache_1.val[0],data1.val[0]);
        output_cache_1.val[1] = vaddq_u8(output_cache_1.val[1],data1.val[1]);
        output_cache_1.val[2] = vaddq_u8(output_cache_1.val[2],data1.val[2]);
        output_cache_1.val[3] = vaddq_u8(output_cache_1.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_1.val[0])+vaddvq_u8(output_cache_1.val[1])+vaddvq_u8(output_cache_1.val[2])+vaddvq_u8(output_cache_1.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_2.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_2.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_2.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_2.val[3],data2.val[3]);
        output_cache_2.val[0] = vaddq_u8(output_cache_2.val[0],data1.val[0]);
        output_cache_2.val[1] = vaddq_u8(output_cache_2.val[1],data1.val[1]);
        output_cache_2.val[2] = vaddq_u8(output_cache_2.val[2],data1.val[2]);
        output_cache_2.val[3] = vaddq_u8(output_cache_2.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_2.val[0])+vaddvq_u8(output_cache_2.val[1])+vaddvq_u8(output_cache_2.val[2])+vaddvq_u8(output_cache_2.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_3.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_3.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_3.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_3.val[3],data2.val[3]);
        output_cache_3.val[0] = vaddq_u8(output_cache_3.val[0],data1.val[0]);
        output_cache_3.val[1] = vaddq_u8(output_cache_3.val[1],data1.val[1]);
        output_cache_3.val[2] = vaddq_u8(output_cache_3.val[2],data1.val[2]);
        output_cache_3.val[3] = vaddq_u8(output_cache_3.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_3.val[0])+vaddvq_u8(output_cache_3.val[1])+vaddvq_u8(output_cache_3.val[2])+vaddvq_u8(output_cache_3.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_4.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_4.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_4.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_4.val[3],data2.val[3]);
        output_cache_4.val[0] = vaddq_u8(output_cache_4.val[0],data1.val[0]);
        output_cache_4.val[1] = vaddq_u8(output_cache_4.val[1],data1.val[1]);
        output_cache_4.val[2] = vaddq_u8(output_cache_4.val[2],data1.val[2]);
        output_cache_4.val[3] = vaddq_u8(output_cache_4.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_4.val[0])+vaddvq_u8(output_cache_4.val[1])+vaddvq_u8(output_cache_4.val[2])+vaddvq_u8(output_cache_4.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_5.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_5.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_5.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_5.val[3],data2.val[3]);
        output_cache_5.val[0] = vaddq_u8(output_cache_5.val[0],data1.val[0]);
        output_cache_5.val[1] = vaddq_u8(output_cache_5.val[1],data1.val[1]);
        output_cache_5.val[2] = vaddq_u8(output_cache_5.val[2],data1.val[2]);
        output_cache_5.val[3] = vaddq_u8(output_cache_5.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_5.val[0])+vaddvq_u8(output_cache_5.val[1])+vaddvq_u8(output_cache_5.val[2])+vaddvq_u8(output_cache_5.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_6.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_6.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_6.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_6.val[3],data2.val[3]);
        output_cache_6.val[0] = vaddq_u8(output_cache_6.val[0],data1.val[0]);
        output_cache_6.val[1] = vaddq_u8(output_cache_6.val[1],data1.val[1]);
        output_cache_6.val[2] = vaddq_u8(output_cache_6.val[2],data1.val[2]);
        output_cache_6.val[3] = vaddq_u8(output_cache_6.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_6.val[0])+vaddvq_u8(output_cache_6.val[1])+vaddvq_u8(output_cache_6.val[2])+vaddvq_u8(output_cache_6.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_7.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_7.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_7.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_7.val[3],data2.val[3]);
        output_cache_7.val[0] = vaddq_u8(output_cache_7.val[0],data1.val[0]);
        output_cache_7.val[1] = vaddq_u8(output_cache_7.val[1],data1.val[1]);
        output_cache_7.val[2] = vaddq_u8(output_cache_7.val[2],data1.val[2]);
        output_cache_7.val[3] = vaddq_u8(output_cache_7.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_7.val[0])+vaddvq_u8(output_cache_7.val[1])+vaddvq_u8(output_cache_7.val[2])+vaddvq_u8(output_cache_7.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_8.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_8.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_8.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_8.val[3],data2.val[3]);
        output_cache_8.val[0] = vaddq_u8(output_cache_8.val[0],data1.val[0]);
        output_cache_8.val[1] = vaddq_u8(output_cache_8.val[1],data1.val[1]);
        output_cache_8.val[2] = vaddq_u8(output_cache_8.val[2],data1.val[2]);
        output_cache_8.val[3] = vaddq_u8(output_cache_8.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_8.val[0])+vaddvq_u8(output_cache_8.val[1])+vaddvq_u8(output_cache_8.val[2])+vaddvq_u8(output_cache_8.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_9.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_9.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_9.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_9.val[3],data2.val[3]);
        output_cache_9.val[0] = vaddq_u8(output_cache_9.val[0],data1.val[0]);
        output_cache_9.val[1] = vaddq_u8(output_cache_9.val[1],data1.val[1]);
        output_cache_9.val[2] = vaddq_u8(output_cache_9.val[2],data1.val[2]);
        output_cache_9.val[3] = vaddq_u8(output_cache_9.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_9.val[0])+vaddvq_u8(output_cache_9.val[1])+vaddvq_u8(output_cache_9.val[2])+vaddvq_u8(output_cache_9.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_10.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_10.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_10.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_10.val[3],data2.val[3]);
        output_cache_10.val[0] = vaddq_u8(output_cache_10.val[0],data1.val[0]);
        output_cache_10.val[1] = vaddq_u8(output_cache_10.val[1],data1.val[1]);
        output_cache_10.val[2] = vaddq_u8(output_cache_10.val[2],data1.val[2]);
        output_cache_10.val[3] = vaddq_u8(output_cache_10.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_10.val[0])+vaddvq_u8(output_cache_10.val[1])+vaddvq_u8(output_cache_10.val[2])+vaddvq_u8(output_cache_10.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_11.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_11.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_11.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_11.val[3],data2.val[3]);
        output_cache_11.val[0] = vaddq_u8(output_cache_11.val[0],data1.val[0]);
        output_cache_11.val[1] = vaddq_u8(output_cache_11.val[1],data1.val[1]);
        output_cache_11.val[2] = vaddq_u8(output_cache_11.val[2],data1.val[2]);
        output_cache_11.val[3] = vaddq_u8(output_cache_11.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_11.val[0])+vaddvq_u8(output_cache_11.val[1])+vaddvq_u8(output_cache_11.val[2])+vaddvq_u8(output_cache_11.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_12.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_12.val[3],data2.val[3]);
        output_cache_12.val[0] = vaddq_u8(output_cache_12.val[0],data1.val[0]);
        output_cache_12.val[1] = vaddq_u8(output_cache_12.val[1],data1.val[1]);
        output_cache_12.val[2] = vaddq_u8(output_cache_12.val[2],data1.val[2]);
        output_cache_12.val[3] = vaddq_u8(output_cache_12.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_12.val[0])+vaddvq_u8(output_cache_12.val[1])+vaddvq_u8(output_cache_12.val[2])+vaddvq_u8(output_cache_12.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_13.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_13.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_13.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_13.val[3],data2.val[3]);
        output_cache_13.val[0] = vaddq_u8(output_cache_13.val[0],data1.val[0]);
        output_cache_13.val[1] = vaddq_u8(output_cache_13.val[1],data1.val[1]);
        output_cache_13.val[2] = vaddq_u8(output_cache_13.val[2],data1.val[2]);
        output_cache_13.val[3] = vaddq_u8(output_cache_13.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_13.val[0])+vaddvq_u8(output_cache_13.val[1])+vaddvq_u8(output_cache_13.val[2])+vaddvq_u8(output_cache_13.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_14.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_14.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_14.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_14.val[3],data2.val[3]);
        output_cache_14.val[0] = vaddq_u8(output_cache_14.val[0],data1.val[0]);
        output_cache_14.val[1] = vaddq_u8(output_cache_14.val[1],data1.val[1]);
        output_cache_14.val[2] = vaddq_u8(output_cache_14.val[2],data1.val[2]);
        output_cache_14.val[3] = vaddq_u8(output_cache_14.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_14.val[0])+vaddvq_u8(output_cache_14.val[1])+vaddvq_u8(output_cache_14.val[2])+vaddvq_u8(output_cache_14.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_15.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_15.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_15.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_15.val[3],data2.val[3]);
        output_cache_15.val[0] = vaddq_u8(output_cache_15.val[0],data1.val[0]);
        output_cache_15.val[1] = vaddq_u8(output_cache_15.val[1],data1.val[1]);
        output_cache_15.val[2] = vaddq_u8(output_cache_15.val[2],data1.val[2]);
        output_cache_15.val[3] = vaddq_u8(output_cache_15.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_15.val[0])+vaddvq_u8(output_cache_15.val[1])+vaddvq_u8(output_cache_15.val[2])+vaddvq_u8(output_cache_15.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_16.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_16.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_16.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_16.val[3],data2.val[3]);
        output_cache_16.val[0] = vaddq_u8(output_cache_16.val[0],data1.val[0]);
        output_cache_16.val[1] = vaddq_u8(output_cache_16.val[1],data1.val[1]);
        output_cache_16.val[2] = vaddq_u8(output_cache_16.val[2],data1.val[2]);
        output_cache_16.val[3] = vaddq_u8(output_cache_16.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_16.val[0])+vaddvq_u8(output_cache_16.val[1])+vaddvq_u8(output_cache_16.val[2])+vaddvq_u8(output_cache_16.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_17.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_17.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_17.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_17.val[3],data2.val[3]);
        output_cache_17.val[0] = vaddq_u8(output_cache_17.val[0],data1.val[0]);
        output_cache_17.val[1] = vaddq_u8(output_cache_17.val[1],data1.val[1]);
        output_cache_17.val[2] = vaddq_u8(output_cache_17.val[2],data1.val[2]);
        output_cache_17.val[3] = vaddq_u8(output_cache_17.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_17.val[0])+vaddvq_u8(output_cache_17.val[1])+vaddvq_u8(output_cache_17.val[2])+vaddvq_u8(output_cache_17.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_18.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_18.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_18.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_18.val[3],data2.val[3]);
        output_cache_18.val[0] = vaddq_u8(output_cache_18.val[0],data1.val[0]);
        output_cache_18.val[1] = vaddq_u8(output_cache_18.val[1],data1.val[1]);
        output_cache_18.val[2] = vaddq_u8(output_cache_18.val[2],data1.val[2]);
        output_cache_18.val[3] = vaddq_u8(output_cache_18.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_18.val[0])+vaddvq_u8(output_cache_18.val[1])+vaddvq_u8(output_cache_18.val[2])+vaddvq_u8(output_cache_18.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_19.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_19.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_19.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_19.val[3],data2.val[3]);
        output_cache_19.val[0] = vaddq_u8(output_cache_19.val[0],data1.val[0]);
        output_cache_19.val[1] = vaddq_u8(output_cache_19.val[1],data1.val[1]);
        output_cache_19.val[2] = vaddq_u8(output_cache_19.val[2],data1.val[2]);
        output_cache_19.val[3] = vaddq_u8(output_cache_19.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_19.val[0])+vaddvq_u8(output_cache_19.val[1])+vaddvq_u8(output_cache_19.val[2])+vaddvq_u8(output_cache_19.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_20.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_20.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_20.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_20.val[3],data2.val[3]);
        output_cache_20.val[0] = vaddq_u8(output_cache_20.val[0],data1.val[0]);
        output_cache_20.val[1] = vaddq_u8(output_cache_20.val[1],data1.val[1]);
        output_cache_20.val[2] = vaddq_u8(output_cache_20.val[2],data1.val[2]);
        output_cache_20.val[3] = vaddq_u8(output_cache_20.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_20.val[0])+vaddvq_u8(output_cache_20.val[1])+vaddvq_u8(output_cache_20.val[2])+vaddvq_u8(output_cache_20.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_21.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_21.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_21.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_21.val[3],data2.val[3]);
        output_cache_21.val[0] = vaddq_u8(output_cache_21.val[0],data1.val[0]);
        output_cache_21.val[1] = vaddq_u8(output_cache_21.val[1],data1.val[1]);
        output_cache_21.val[2] = vaddq_u8(output_cache_21.val[2],data1.val[2]);
        output_cache_21.val[3] = vaddq_u8(output_cache_21.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_21.val[0])+vaddvq_u8(output_cache_21.val[1])+vaddvq_u8(output_cache_21.val[2])+vaddvq_u8(output_cache_21.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_22.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_22.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_22.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_22.val[3],data2.val[3]);
        output_cache_22.val[0] = vaddq_u8(output_cache_22.val[0],data1.val[0]);
        output_cache_22.val[1] = vaddq_u8(output_cache_22.val[1],data1.val[1]);
        output_cache_22.val[2] = vaddq_u8(output_cache_22.val[2],data1.val[2]);
        output_cache_22.val[3] = vaddq_u8(output_cache_22.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_22.val[0])+vaddvq_u8(output_cache_22.val[1])+vaddvq_u8(output_cache_22.val[2])+vaddvq_u8(output_cache_22.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_23.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_23.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_23.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_23.val[3],data2.val[3]);
        output_cache_23.val[0] = vaddq_u8(output_cache_23.val[0],data1.val[0]);
        output_cache_23.val[1] = vaddq_u8(output_cache_23.val[1],data1.val[1]);
        output_cache_23.val[2] = vaddq_u8(output_cache_23.val[2],data1.val[2]);
        output_cache_23.val[3] = vaddq_u8(output_cache_23.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_23.val[0])+vaddvq_u8(output_cache_23.val[1])+vaddvq_u8(output_cache_23.val[2])+vaddvq_u8(output_cache_23.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_24.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_24.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_24.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_24.val[3],data2.val[3]);
        output_cache_24.val[0] = vaddq_u8(output_cache_24.val[0],data1.val[0]);
        output_cache_24.val[1] = vaddq_u8(output_cache_24.val[1],data1.val[1]);
        output_cache_24.val[2] = vaddq_u8(output_cache_24.val[2],data1.val[2]);
        output_cache_24.val[3] = vaddq_u8(output_cache_24.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_24.val[0])+vaddvq_u8(output_cache_24.val[1])+vaddvq_u8(output_cache_24.val[2])+vaddvq_u8(output_cache_24.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_25.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_25.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_25.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_25.val[3],data2.val[3]);
        output_cache_25.val[0] = vaddq_u8(output_cache_25.val[0],data1.val[0]);
        output_cache_25.val[1] = vaddq_u8(output_cache_25.val[1],data1.val[1]);
        output_cache_25.val[2] = vaddq_u8(output_cache_25.val[2],data1.val[2]);
        output_cache_25.val[3] = vaddq_u8(output_cache_25.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_25.val[0])+vaddvq_u8(output_cache_25.val[1])+vaddvq_u8(output_cache_25.val[2])+vaddvq_u8(output_cache_25.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_26.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_26.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_26.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_26.val[3],data2.val[3]);
        output_cache_26.val[0] = vaddq_u8(output_cache_26.val[0],data1.val[0]);
        output_cache_26.val[1] = vaddq_u8(output_cache_26.val[1],data1.val[1]);
        output_cache_26.val[2] = vaddq_u8(output_cache_26.val[2],data1.val[2]);
        output_cache_26.val[3] = vaddq_u8(output_cache_26.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_26.val[0])+vaddvq_u8(output_cache_26.val[1])+vaddvq_u8(output_cache_26.val[2])+vaddvq_u8(output_cache_26.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_27.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_27.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_27.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_27.val[3],data2.val[3]);
        output_cache_27.val[0] = vaddq_u8(output_cache_27.val[0],data1.val[0]);
        output_cache_27.val[1] = vaddq_u8(output_cache_27.val[1],data1.val[1]);
        output_cache_27.val[2] = vaddq_u8(output_cache_27.val[2],data1.val[2]);
        output_cache_27.val[3] = vaddq_u8(output_cache_27.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_27.val[0])+vaddvq_u8(output_cache_27.val[1])+vaddvq_u8(output_cache_27.val[2])+vaddvq_u8(output_cache_27.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_28.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_28.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_28.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_28.val[3],data2.val[3]);
        output_cache_28.val[0] = vaddq_u8(output_cache_28.val[0],data1.val[0]);
        output_cache_28.val[1] = vaddq_u8(output_cache_28.val[1],data1.val[1]);
        output_cache_28.val[2] = vaddq_u8(output_cache_28.val[2],data1.val[2]);
        output_cache_28.val[3] = vaddq_u8(output_cache_28.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_28.val[0])+vaddvq_u8(output_cache_28.val[1])+vaddvq_u8(output_cache_28.val[2])+vaddvq_u8(output_cache_28.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_29.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_29.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_29.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_29.val[3],data2.val[3]);
        output_cache_29.val[0] = vaddq_u8(output_cache_29.val[0],data1.val[0]);
        output_cache_29.val[1] = vaddq_u8(output_cache_29.val[1],data1.val[1]);
        output_cache_29.val[2] = vaddq_u8(output_cache_29.val[2],data1.val[2]);
        output_cache_29.val[3] = vaddq_u8(output_cache_29.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_29.val[0])+vaddvq_u8(output_cache_29.val[1])+vaddvq_u8(output_cache_29.val[2])+vaddvq_u8(output_cache_29.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_30.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_30.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_30.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_30.val[3],data2.val[3]);
        output_cache_30.val[0] = vaddq_u8(output_cache_30.val[0],data1.val[0]);
        output_cache_30.val[1] = vaddq_u8(output_cache_30.val[1],data1.val[1]);
        output_cache_30.val[2] = vaddq_u8(output_cache_30.val[2],data1.val[2]);
        output_cache_30.val[3] = vaddq_u8(output_cache_30.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_30.val[0])+vaddvq_u8(output_cache_30.val[1])+vaddvq_u8(output_cache_30.val[2])+vaddvq_u8(output_cache_30.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_31.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_31.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_31.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_31.val[3],data2.val[3]);
        output_cache_31.val[0] = vaddq_u8(output_cache_31.val[0],data1.val[0]);
        output_cache_31.val[1] = vaddq_u8(output_cache_31.val[1],data1.val[1]);
        output_cache_31.val[2] = vaddq_u8(output_cache_31.val[2],data1.val[2]);
        output_cache_31.val[3] = vaddq_u8(output_cache_31.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_31.val[0])+vaddvq_u8(output_cache_31.val[1])+vaddvq_u8(output_cache_31.val[2])+vaddvq_u8(output_cache_31.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_32.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_32.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_32.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_32.val[3],data2.val[3]);
        output_cache_32.val[0] = vaddq_u8(output_cache_32.val[0],data1.val[0]);
        output_cache_32.val[1] = vaddq_u8(output_cache_32.val[1],data1.val[1]);
        output_cache_32.val[2] = vaddq_u8(output_cache_32.val[2],data1.val[2]);
        output_cache_32.val[3] = vaddq_u8(output_cache_32.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_32.val[0])+vaddvq_u8(output_cache_32.val[1])+vaddvq_u8(output_cache_32.val[2])+vaddvq_u8(output_cache_32.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1.val[0] = vmulq_s8(input_cache_33.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(input_cache_33.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(input_cache_33.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(input_cache_33.val[3],data2.val[3]);
        output_cache_33.val[0] = vaddq_u8(output_cache_33.val[0],data1.val[0]);
        output_cache_33.val[1] = vaddq_u8(output_cache_33.val[1],data1.val[1]);
        output_cache_33.val[2] = vaddq_u8(output_cache_33.val[2],data1.val[2]);
        output_cache_33.val[3] = vaddq_u8(output_cache_33.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_33.val[0])+vaddvq_u8(output_cache_33.val[1])+vaddvq_u8(output_cache_33.val[2])+vaddvq_u8(output_cache_33.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
        output_cache_34.val[0] = vaddq_u8(output_cache_34.val[0],data1.val[0]);
        output_cache_34.val[1] = vaddq_u8(output_cache_34.val[1],data1.val[1]);
        output_cache_34.val[2] = vaddq_u8(output_cache_34.val[2],data1.val[2]);
        output_cache_34.val[3] = vaddq_u8(output_cache_34.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_34.val[0])+vaddvq_u8(output_cache_34.val[1])+vaddvq_u8(output_cache_34.val[2])+vaddvq_u8(output_cache_34.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
        output_cache_35.val[0] = vaddq_u8(output_cache_35.val[0],data1.val[0]);
        output_cache_35.val[1] = vaddq_u8(output_cache_35.val[1],data1.val[1]);
        output_cache_35.val[2] = vaddq_u8(output_cache_35.val[2],data1.val[2]);
        output_cache_35.val[3] = vaddq_u8(output_cache_35.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_35.val[0])+vaddvq_u8(output_cache_35.val[1])+vaddvq_u8(output_cache_35.val[2])+vaddvq_u8(output_cache_35.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
        output_cache_36.val[0] = vaddq_u8(output_cache_36.val[0],data1.val[0]);
        output_cache_36.val[1] = vaddq_u8(output_cache_36.val[1],data1.val[1]);
        output_cache_36.val[2] = vaddq_u8(output_cache_36.val[2],data1.val[2]);
        output_cache_36.val[3] = vaddq_u8(output_cache_36.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_36.val[0])+vaddvq_u8(output_cache_36.val[1])+vaddvq_u8(output_cache_36.val[2])+vaddvq_u8(output_cache_36.val[3]);
        
        
        w++;
        input_h = h * strides + i - padding;
        input_w = w * strides + j - padding;
        data1 = vld1q_s64_x4((const int64_t*)& filters[(f * filter_height * filter_width + i * filter_width + j)*512/64]);
        data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
        data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
        data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
        data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
        output_cache_37.val[0] = vaddq_u8(output_cache_37.val[0],data1.val[0]);
        output_cache_37.val[1] = vaddq_u8(output_cache_37.val[1],data1.val[1]);
        output_cache_37.val[2] = vaddq_u8(output_cache_37.val[2],data1.val[2]);
        output_cache_37.val[3] = vaddq_u8(output_cache_37.val[3],data1.val[3]);
        outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(output_cache_37.val[0])+vaddvq_u8(output_cache_37.val[1])+vaddvq_u8(output_cache_37.val[2])+vaddvq_u8(output_cache_37.val[3]);
        
        
        for (h = 0; h < out_height; h++) {
            for (w = 38; w < out_width; w++) {
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_s64_x4((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * 512 /64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                data1.val[2] = vmulq_s8(data1.val[2],data2.val[2]);
                data1.val[3] = vmulq_s8(data1.val[3],data2.val[3]);
                outputs[h * out_width * num_filters + w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
            }
        }
    }
}