#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <arm_neon.h>
#include <algorithm>
#include <m5ops.h>

using namespace std;



int main(int argc, char *argv[])
{

    /* Please type the following in the command line
        ./<program_name> <input_height> <input_width> <input_depth (plase use 256)> <num_filters>
    */

    FILE *pFile = fopen("../durations/ws_os0_is0.txt", "a");
    int height;
    int width;
    int depth;
    int filter_height;
    int filter_width;
    int num_filters;
    int padding;
    int strides;
    int curr;
    int64_t *inputs;
    int *outputs;
    int64_t *filters;

    int output_depth;
    int pool_size;

    int input_size;

    int layer_counter = 0;
    double time_elapsed_ms;
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    depth = atoi(argv[3]);
    num_filters = atoi(argv[4]);
    filter_height = 3;
    filter_width = 3;
    padding = 2;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int *)malloc(sizeof(int) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    uint64x2x2_t data1;
    uint64x2x2_t data2;
    uint64x2x2_t output_cache_1;
    uint64x2x2_t output_cache_2;
    uint64x2x2_t output_cache_3;
    uint64x2x2_t output_cache_4;
    uint64x2x2_t output_cache_5;
    uint64x2x2_t output_cache_6;
    uint64x2x2_t output_cache_7;
    uint64x2x2_t output_cache_8;
    uint64x2x2_t output_cache_9;
    uint64x2x2_t output_cache_10;
    uint64x2x2_t output_cache_11;
    uint64x2x2_t output_cache_12;
    uint64x2x2_t output_cache_13;
    uint64x2x2_t output_cache_14;
    uint64x2x2_t output_cache_15;
    uint64x2x2_t output_cache_16;
    uint64x2x2_t output_cache_17;
    uint64x2x2_t output_cache_18;
    uint64x2x2_t output_cache_19;
    uint64x2x2_t output_cache_20;
    uint64x2x2_t output_cache_21;
    uint64x2x2_t output_cache_22;
    uint64x2x2_t output_cache_23;
    uint64x2x2_t output_cache_24;
    uint64x2x2_t output_cache_25;
    uint64x2x2_t output_cache_26;
    uint64x2x2_t output_cache_27;
    uint64x2x2_t output_cache_28;
    uint64x2x2_t output_cache_29;
    uint64x2x2_t output_cache_30;
    

    m5_reset_stats(0, 0);


    for (int f = 0; f < num_filters; f++) {
        output_cache_1.val[0] = vdupq_n_u64(0);
        output_cache_1.val[1] = vdupq_n_u64(0);
        output_cache_2.val[0] = vdupq_n_u64(0);
        output_cache_2.val[1] = vdupq_n_u64(0);
        output_cache_3.val[0] = vdupq_n_u64(0);
        output_cache_3.val[1] = vdupq_n_u64(0);
        output_cache_4.val[0] = vdupq_n_u64(0);
        output_cache_4.val[1] = vdupq_n_u64(0);
        output_cache_5.val[0] = vdupq_n_u64(0);
        output_cache_5.val[1] = vdupq_n_u64(0);
        output_cache_6.val[0] = vdupq_n_u64(0);
        output_cache_6.val[1] = vdupq_n_u64(0);
        output_cache_7.val[0] = vdupq_n_u64(0);
        output_cache_7.val[1] = vdupq_n_u64(0);
        output_cache_8.val[0] = vdupq_n_u64(0);
        output_cache_8.val[1] = vdupq_n_u64(0);
        output_cache_9.val[0] = vdupq_n_u64(0);
        output_cache_9.val[1] = vdupq_n_u64(0);
        output_cache_10.val[0] = vdupq_n_u64(0);
        output_cache_10.val[1] = vdupq_n_u64(0);
        output_cache_11.val[0] = vdupq_n_u64(0);
        output_cache_11.val[1] = vdupq_n_u64(0);
        output_cache_12.val[0] = vdupq_n_u64(0);
        output_cache_12.val[1] = vdupq_n_u64(0);
        output_cache_13.val[0] = vdupq_n_u64(0);
        output_cache_13.val[1] = vdupq_n_u64(0);
        output_cache_14.val[0] = vdupq_n_u64(0);
        output_cache_14.val[1] = vdupq_n_u64(0);
        output_cache_15.val[0] = vdupq_n_u64(0);
        output_cache_15.val[1] = vdupq_n_u64(0);
        output_cache_16.val[0] = vdupq_n_u64(0);
        output_cache_16.val[1] = vdupq_n_u64(0);
        output_cache_17.val[0] = vdupq_n_u64(0);
        output_cache_17.val[1] = vdupq_n_u64(0);
        output_cache_18.val[0] = vdupq_n_u64(0);
        output_cache_18.val[1] = vdupq_n_u64(0);
        output_cache_19.val[0] = vdupq_n_u64(0);
        output_cache_19.val[1] = vdupq_n_u64(0);
        output_cache_20.val[0] = vdupq_n_u64(0);
        output_cache_20.val[1] = vdupq_n_u64(0);
        output_cache_21.val[0] = vdupq_n_u64(0);
        output_cache_21.val[1] = vdupq_n_u64(0);
        output_cache_22.val[0] = vdupq_n_u64(0);
        output_cache_22.val[1] = vdupq_n_u64(0);
        output_cache_23.val[0] = vdupq_n_u64(0);
        output_cache_23.val[1] = vdupq_n_u64(0);
        output_cache_24.val[0] = vdupq_n_u64(0);
        output_cache_24.val[1] = vdupq_n_u64(0);
        output_cache_25.val[0] = vdupq_n_u64(0);
        output_cache_25.val[1] = vdupq_n_u64(0);
        output_cache_26.val[0] = vdupq_n_u64(0);
        output_cache_26.val[1] = vdupq_n_u64(0);
        output_cache_27.val[0] = vdupq_n_u64(0);
        output_cache_27.val[1] = vdupq_n_u64(0);
        output_cache_28.val[0] = vdupq_n_u64(0);
        output_cache_28.val[1] = vdupq_n_u64(0);
        output_cache_29.val[0] = vdupq_n_u64(0);
        output_cache_29.val[1] = vdupq_n_u64(0);
        output_cache_30.val[0] = vdupq_n_u64(0);
        output_cache_30.val[1] = vdupq_n_u64(0);

        int i;
        int j;
        for (i = 0; i < filter_height - 1; i ++) {
            for (j = 0; j < filter_width; j ++) {
                int h = 0;
                int w = 0;
                data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);

                int input_h;
                int input_w;
                
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_1.val[0] = vaddq_u8(output_cache_1.val[0],data1.val[0]);
                output_cache_1.val[1] = vaddq_u8(output_cache_1.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_2.val[0] = vaddq_u8(output_cache_2.val[0],data1.val[0]);
                output_cache_2.val[1] = vaddq_u8(output_cache_2.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_3.val[0] = vaddq_u8(output_cache_3.val[0],data1.val[0]);
                output_cache_3.val[1] = vaddq_u8(output_cache_3.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_4.val[0] = vaddq_u8(output_cache_4.val[0],data1.val[0]);
                output_cache_4.val[1] = vaddq_u8(output_cache_4.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_5.val[0] = vaddq_u8(output_cache_5.val[0],data1.val[0]);
                output_cache_5.val[1] = vaddq_u8(output_cache_5.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_6.val[0] = vaddq_u8(output_cache_6.val[0],data1.val[0]);
                output_cache_6.val[1] = vaddq_u8(output_cache_6.val[0],data1.val[1]);

                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_7.val[0] = vaddq_u8(output_cache_7.val[0],data1.val[0]);
                output_cache_7.val[1] = vaddq_u8(output_cache_7.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_8.val[0] = vaddq_u8(output_cache_8.val[0],data1.val[0]);
                output_cache_8.val[1] = vaddq_u8(output_cache_8.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_9.val[0] = vaddq_u8(output_cache_9.val[0],data1.val[0]);
                output_cache_9.val[1] = vaddq_u8(output_cache_9.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_10.val[0] = vaddq_u8(output_cache_10.val[0],data1.val[0]);
                output_cache_10.val[1] = vaddq_u8(output_cache_10.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_11.val[0] = vaddq_u8(output_cache_11.val[0],data1.val[0]);
                output_cache_11.val[1] = vaddq_u8(output_cache_11.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_12.val[0] = vaddq_u8(output_cache_12.val[0],data1.val[0]);
                output_cache_12.val[1] = vaddq_u8(output_cache_12.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_13.val[0] = vaddq_u8(output_cache_13.val[0],data1.val[0]);
                output_cache_13.val[1] = vaddq_u8(output_cache_13.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_14.val[0] = vaddq_u8(output_cache_14.val[0],data1.val[0]);
                output_cache_14.val[1] = vaddq_u8(output_cache_14.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_15.val[0] = vaddq_u8(output_cache_15.val[0],data1.val[0]);
                output_cache_15.val[1] = vaddq_u8(output_cache_15.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_16.val[0] = vaddq_u8(output_cache_16.val[0],data1.val[0]);
                output_cache_16.val[1] = vaddq_u8(output_cache_16.val[0],data1.val[1]);

                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_17.val[0] = vaddq_u8(output_cache_17.val[0],data1.val[0]);
                output_cache_17.val[1] = vaddq_u8(output_cache_17.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_18.val[0] = vaddq_u8(output_cache_18.val[0],data1.val[0]);
                output_cache_18.val[1] = vaddq_u8(output_cache_18.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_19.val[0] = vaddq_u8(output_cache_19.val[0],data1.val[0]);
                output_cache_19.val[1] = vaddq_u8(output_cache_19.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_20.val[0] = vaddq_u8(output_cache_20.val[0],data1.val[0]);
                output_cache_20.val[1] = vaddq_u8(output_cache_20.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_21.val[0] = vaddq_u8(output_cache_21.val[0],data1.val[0]);
                output_cache_21.val[1] = vaddq_u8(output_cache_21.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_22.val[0] = vaddq_u8(output_cache_22.val[0],data1.val[0]);
                output_cache_22.val[1] = vaddq_u8(output_cache_22.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_23.val[0] = vaddq_u8(output_cache_23.val[0],data1.val[0]);
                output_cache_23.val[1] = vaddq_u8(output_cache_23.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_24.val[0] = vaddq_u8(output_cache_24.val[0],data1.val[0]);
                output_cache_24.val[1] = vaddq_u8(output_cache_24.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_25.val[0] = vaddq_u8(output_cache_25.val[0],data1.val[0]);
                output_cache_25.val[1] = vaddq_u8(output_cache_25.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_26.val[0] = vaddq_u8(output_cache_26.val[0],data1.val[0]);
                output_cache_26.val[1] = vaddq_u8(output_cache_26.val[0],data1.val[1]);

                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_27.val[0] = vaddq_u8(output_cache_27.val[0],data1.val[0]);
                output_cache_27.val[1] = vaddq_u8(output_cache_27.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_28.val[0] = vaddq_u8(output_cache_28.val[0],data1.val[0]);
                output_cache_28.val[1] = vaddq_u8(output_cache_28.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_29.val[0] = vaddq_u8(output_cache_29.val[0],data1.val[0]);
                output_cache_29.val[1] = vaddq_u8(output_cache_29.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_30.val[0] = vaddq_u8(output_cache_30.val[0],data1.val[0]);
                output_cache_30.val[1] = vaddq_u8(output_cache_30.val[0],data1.val[1]);
                

                for (h = 0; h < out_height; h++) {
                    for (w = 30; w < out_width; w++) {
                        input_h = h * strides + i - padding;
                        input_w = w * strides + j - padding;
                        data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                        data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                        data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));
                    }
                }
            }
        for (j = 0; j < filter_width - 1; j ++) {
                data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);

                int input_h;
                int input_w;
                int h = 0;
                int w = 0;
                
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_1.val[0] = vaddq_u8(output_cache_1.val[0],data1.val[0]);
                output_cache_1.val[1] = vaddq_u8(output_cache_1.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_2.val[0] = vaddq_u8(output_cache_2.val[0],data1.val[0]);
                output_cache_2.val[1] = vaddq_u8(output_cache_2.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_3.val[0] = vaddq_u8(output_cache_3.val[0],data1.val[0]);
                output_cache_3.val[1] = vaddq_u8(output_cache_3.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_4.val[0] = vaddq_u8(output_cache_4.val[0],data1.val[0]);
                output_cache_4.val[1] = vaddq_u8(output_cache_4.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_5.val[0] = vaddq_u8(output_cache_5.val[0],data1.val[0]);
                output_cache_5.val[1] = vaddq_u8(output_cache_5.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_6.val[0] = vaddq_u8(output_cache_6.val[0],data1.val[0]);
                output_cache_6.val[1] = vaddq_u8(output_cache_6.val[0],data1.val[1]);

                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_7.val[0] = vaddq_u8(output_cache_7.val[0],data1.val[0]);
                output_cache_7.val[1] = vaddq_u8(output_cache_7.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_8.val[0] = vaddq_u8(output_cache_8.val[0],data1.val[0]);
                output_cache_8.val[1] = vaddq_u8(output_cache_8.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_9.val[0] = vaddq_u8(output_cache_9.val[0],data1.val[0]);
                output_cache_9.val[1] = vaddq_u8(output_cache_9.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_10.val[0] = vaddq_u8(output_cache_10.val[0],data1.val[0]);
                output_cache_10.val[1] = vaddq_u8(output_cache_10.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_11.val[0] = vaddq_u8(output_cache_11.val[0],data1.val[0]);
                output_cache_11.val[1] = vaddq_u8(output_cache_11.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_12.val[0] = vaddq_u8(output_cache_12.val[0],data1.val[0]);
                output_cache_12.val[1] = vaddq_u8(output_cache_12.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_13.val[0] = vaddq_u8(output_cache_13.val[0],data1.val[0]);
                output_cache_13.val[1] = vaddq_u8(output_cache_13.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_14.val[0] = vaddq_u8(output_cache_14.val[0],data1.val[0]);
                output_cache_14.val[1] = vaddq_u8(output_cache_14.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_15.val[0] = vaddq_u8(output_cache_15.val[0],data1.val[0]);
                output_cache_15.val[1] = vaddq_u8(output_cache_15.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_16.val[0] = vaddq_u8(output_cache_16.val[0],data1.val[0]);
                output_cache_16.val[1] = vaddq_u8(output_cache_16.val[0],data1.val[1]);

                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_17.val[0] = vaddq_u8(output_cache_17.val[0],data1.val[0]);
                output_cache_17.val[1] = vaddq_u8(output_cache_17.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_18.val[0] = vaddq_u8(output_cache_18.val[0],data1.val[0]);
                output_cache_18.val[1] = vaddq_u8(output_cache_18.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_19.val[0] = vaddq_u8(output_cache_19.val[0],data1.val[0]);
                output_cache_19.val[1] = vaddq_u8(output_cache_19.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_20.val[0] = vaddq_u8(output_cache_20.val[0],data1.val[0]);
                output_cache_20.val[1] = vaddq_u8(output_cache_20.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_21.val[0] = vaddq_u8(output_cache_21.val[0],data1.val[0]);
                output_cache_21.val[1] = vaddq_u8(output_cache_21.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_22.val[0] = vaddq_u8(output_cache_22.val[0],data1.val[0]);
                output_cache_22.val[1] = vaddq_u8(output_cache_22.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_23.val[0] = vaddq_u8(output_cache_23.val[0],data1.val[0]);
                output_cache_23.val[1] = vaddq_u8(output_cache_23.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_24.val[0] = vaddq_u8(output_cache_24.val[0],data1.val[0]);
                output_cache_24.val[1] = vaddq_u8(output_cache_24.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_25.val[0] = vaddq_u8(output_cache_25.val[0],data1.val[0]);
                output_cache_25.val[1] = vaddq_u8(output_cache_25.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_26.val[0] = vaddq_u8(output_cache_26.val[0],data1.val[0]);
                output_cache_26.val[1] = vaddq_u8(output_cache_26.val[0],data1.val[1]);

                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_27.val[0] = vaddq_u8(output_cache_27.val[0],data1.val[0]);
                output_cache_27.val[1] = vaddq_u8(output_cache_27.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_28.val[0] = vaddq_u8(output_cache_28.val[0],data1.val[0]);
                output_cache_28.val[1] = vaddq_u8(output_cache_28.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_29.val[0] = vaddq_u8(output_cache_29.val[0],data1.val[0]);
                output_cache_29.val[1] = vaddq_u8(output_cache_29.val[0],data1.val[1]);

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_30.val[0] = vaddq_u8(output_cache_30.val[0],data1.val[0]);
                output_cache_30.val[1] = vaddq_u8(output_cache_30.val[0],data1.val[1]);
                

                for (int h = 0; h < out_height; h++) {
                    for (int w = 30; w < out_width; w++) {
                        input_h = h * strides + i - padding;
                        input_w = w * strides + j - padding;
                        data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                        data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                        data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));
                    }
                }
                data2 = vld1q_u64_x2((const uint64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);

                
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_1.val[0] = vaddq_u8(output_cache_1.val[0],data1.val[0]);
                output_cache_1.val[1] = vaddq_u8(output_cache_1.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_1.val[0])+vaddvq_u8(output_cache_1.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_2.val[0] = vaddq_u8(output_cache_2.val[0],data1.val[0]);
                output_cache_2.val[1] = vaddq_u8(output_cache_2.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_2.val[0])+vaddvq_u8(output_cache_2.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_3.val[0] = vaddq_u8(output_cache_3.val[0],data1.val[0]);
                output_cache_3.val[1] = vaddq_u8(output_cache_3.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_3.val[0])+vaddvq_u8(output_cache_3.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_4.val[0] = vaddq_u8(output_cache_4.val[0],data1.val[0]);
                output_cache_4.val[1] = vaddq_u8(output_cache_4.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_4.val[0])+vaddvq_u8(output_cache_4.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_5.val[0] = vaddq_u8(output_cache_5.val[0],data1.val[0]);
                output_cache_5.val[1] = vaddq_u8(output_cache_5.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_5.val[0])+vaddvq_u8(output_cache_5.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_6.val[0] = vaddq_u8(output_cache_6.val[0],data1.val[0]);
                output_cache_6.val[1] = vaddq_u8(output_cache_6.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_6.val[0])+vaddvq_u8(output_cache_6.val[1]));

                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_7.val[0] = vaddq_u8(output_cache_7.val[0],data1.val[0]);
                output_cache_7.val[1] = vaddq_u8(output_cache_7.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_7.val[0])+vaddvq_u8(output_cache_7.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_8.val[0] = vaddq_u8(output_cache_8.val[0],data1.val[0]);
                output_cache_8.val[1] = vaddq_u8(output_cache_8.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_8.val[0])+vaddvq_u8(output_cache_8.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_9.val[0] = vaddq_u8(output_cache_9.val[0],data1.val[0]);
                output_cache_9.val[1] = vaddq_u8(output_cache_9.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_9.val[0])+vaddvq_u8(output_cache_9.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_10.val[0] = vaddq_u8(output_cache_10.val[0],data1.val[0]);
                output_cache_10.val[1] = vaddq_u8(output_cache_10.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_10.val[0])+vaddvq_u8(output_cache_10.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_11.val[0] = vaddq_u8(output_cache_11.val[0],data1.val[0]);
                output_cache_11.val[1] = vaddq_u8(output_cache_11.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_11.val[0])+vaddvq_u8(output_cache_11.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_12.val[0] = vaddq_u8(output_cache_12.val[0],data1.val[0]);
                output_cache_12.val[1] = vaddq_u8(output_cache_12.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_12.val[0])+vaddvq_u8(output_cache_12.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_13.val[0] = vaddq_u8(output_cache_13.val[0],data1.val[0]);
                output_cache_13.val[1] = vaddq_u8(output_cache_13.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_13.val[0])+vaddvq_u8(output_cache_13.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_14.val[0] = vaddq_u8(output_cache_14.val[0],data1.val[0]);
                output_cache_14.val[1] = vaddq_u8(output_cache_14.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_14.val[0])+vaddvq_u8(output_cache_14.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_15.val[0] = vaddq_u8(output_cache_15.val[0],data1.val[0]);
                output_cache_15.val[1] = vaddq_u8(output_cache_15.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_15.val[0])+vaddvq_u8(output_cache_15.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_16.val[0] = vaddq_u8(output_cache_16.val[0],data1.val[0]);
                output_cache_16.val[1] = vaddq_u8(output_cache_16.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_16.val[0])+vaddvq_u8(output_cache_16.val[1]));

                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_17.val[0] = vaddq_u8(output_cache_17.val[0],data1.val[0]);
                output_cache_17.val[1] = vaddq_u8(output_cache_17.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_17.val[0])+vaddvq_u8(output_cache_17.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_18.val[0] = vaddq_u8(output_cache_18.val[0],data1.val[0]);
                output_cache_18.val[1] = vaddq_u8(output_cache_18.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_18.val[0])+vaddvq_u8(output_cache_18.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_19.val[0] = vaddq_u8(output_cache_19.val[0],data1.val[0]);
                output_cache_19.val[1] = vaddq_u8(output_cache_19.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_19.val[0])+vaddvq_u8(output_cache_19.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_20.val[0] = vaddq_u8(output_cache_20.val[0],data1.val[0]);
                output_cache_20.val[1] = vaddq_u8(output_cache_20.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_20.val[0])+vaddvq_u8(output_cache_20.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_21.val[0] = vaddq_u8(output_cache_21.val[0],data1.val[0]);
                output_cache_21.val[1] = vaddq_u8(output_cache_21.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_21.val[0])+vaddvq_u8(output_cache_21.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_22.val[0] = vaddq_u8(output_cache_22.val[0],data1.val[0]);
                output_cache_22.val[1] = vaddq_u8(output_cache_22.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_22.val[0])+vaddvq_u8(output_cache_22.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_23.val[0] = vaddq_u8(output_cache_23.val[0],data1.val[0]);
                output_cache_23.val[1] = vaddq_u8(output_cache_23.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_23.val[0])+vaddvq_u8(output_cache_23.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_24.val[0] = vaddq_u8(output_cache_24.val[0],data1.val[0]);
                output_cache_24.val[1] = vaddq_u8(output_cache_24.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_24.val[0])+vaddvq_u8(output_cache_24.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_25.val[0] = vaddq_u8(output_cache_25.val[0],data1.val[0]);
                output_cache_25.val[1] = vaddq_u8(output_cache_25.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_25.val[0])+vaddvq_u8(output_cache_25.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_26.val[0] = vaddq_u8(output_cache_26.val[0],data1.val[0]);
                output_cache_26.val[1] = vaddq_u8(output_cache_26.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_26.val[0])+vaddvq_u8(output_cache_26.val[1]));

                w++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_27.val[0] = vaddq_u8(output_cache_27.val[0],data1.val[0]);
                output_cache_27.val[1] = vaddq_u8(output_cache_27.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_27.val[0])+vaddvq_u8(output_cache_27.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_28.val[0] = vaddq_u8(output_cache_28.val[0],data1.val[0]);
                output_cache_28.val[1] = vaddq_u8(output_cache_28.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_28.val[0])+vaddvq_u8(output_cache_28.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_29.val[0] = vaddq_u8(output_cache_29.val[0],data1.val[0]);
                output_cache_29.val[1] = vaddq_u8(output_cache_29.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_29.val[0])+vaddvq_u8(output_cache_29.val[1]));

                w ++;
                input_h = h * strides + i - padding;
                input_w = w * strides + j - padding;
                data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                output_cache_30.val[0] = vaddq_u8(output_cache_30.val[0],data1.val[0]);
                output_cache_30.val[1] = vaddq_u8(output_cache_30.val[0],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] = 256- 2*(vaddvq_u8(output_cache_30.val[0])+vaddvq_u8(output_cache_30.val[1]));
                

                for (int h = 0; h < out_height; h++) {
                    for (int w = 30; w < out_width; w++) {
                        input_h = h * strides + i - padding;
                        input_w = w * strides + j - padding;
                        data1 = vld1q_u64_x2((const uint64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                        data1.val[0] = veorq_u64(data1.val[0], data2.val[0]);
                        data1.val[1] = veorq_u64(data1.val[1], data2.val[1]);
                        outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[0]))) + vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(data1.val[1]))));
                    }
                }
            }
        }
    }

    m5_dump_reset_stats(0, 0);
    std::fprintf(pFile, "%lf\n", time_elapsed_ms);

    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}