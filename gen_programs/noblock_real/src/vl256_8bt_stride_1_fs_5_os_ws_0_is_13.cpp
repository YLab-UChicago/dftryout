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
    std::clock_t c_start;
    std::clock_t c_end;
    double time_elapsed_ms;
    int curr;
    int64_t* inputs;
    int64_t* outputs;
    int64_t* filters;
    int output_depth;
    
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    depth = 256;
    num_filters = atoi(argv[3]);
    filter_height = 5;
    filter_width = 5;
    padding = 4;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    
    int64x2x2_t data1;
    int64x2x2_t data2;
    int64x2x2_t input_cache_0;
    int64x2x2_t input_cache_1;
    int64x2x2_t input_cache_2;
    int64x2x2_t input_cache_3;
    int64x2x2_t input_cache_4;
    int64x2x2_t input_cache_5;
    int64x2x2_t input_cache_6;
    int64x2x2_t input_cache_7;
    int64x2x2_t input_cache_8;
    int64x2x2_t input_cache_9;
    int64x2x2_t input_cache_10;
    int64x2x2_t input_cache_11;
    int64x2x2_t input_cache_12;
    
    
    c_start = std::clock();
    
    for (int f = 0; f < num_filters; f++) {
        input_cache_0 = vld1q_s64_x2((const int64_t *) &inputs[(0 * width * depth /256 + 0) * 256 /64]);
        input_cache_1 = vld1q_s64_x2((const int64_t *) &inputs[(1 * width * depth /256 + 1) * 256 /64]);
        input_cache_2 = vld1q_s64_x2((const int64_t *) &inputs[(2 * width * depth /256 + 2) * 256 /64]);
        input_cache_3 = vld1q_s64_x2((const int64_t *) &inputs[(3 * width * depth /256 + 3) * 256 /64]);
        input_cache_4 = vld1q_s64_x2((const int64_t *) &inputs[(4 * width * depth /256 + 4) * 256 /64]);
        input_cache_5 = vld1q_s64_x2((const int64_t *) &inputs[(5 * width * depth /256 + 5) * 256 /64]);
        input_cache_6 = vld1q_s64_x2((const int64_t *) &inputs[(6 * width * depth /256 + 6) * 256 /64]);
        input_cache_7 = vld1q_s64_x2((const int64_t *) &inputs[(7 * width * depth /256 + 7) * 256 /64]);
        input_cache_8 = vld1q_s64_x2((const int64_t *) &inputs[(8 * width * depth /256 + 8) * 256 /64]);
        input_cache_9 = vld1q_s64_x2((const int64_t *) &inputs[(9 * width * depth /256 + 9) * 256 /64]);
        input_cache_10 = vld1q_s64_x2((const int64_t *) &inputs[(10 * width * depth /256 + 10) * 256 /64]);
        input_cache_11 = vld1q_s64_x2((const int64_t *) &inputs[(11 * width * depth /256 + 11) * 256 /64]);
        input_cache_12 = vld1q_s64_x2((const int64_t *) &inputs[(12 * width * depth /256 + 12) * 256 /64]);
        int64x2x2_t output;
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                int i = 0;
                int j = 0;
                int input_h;
                int input_w;
                 
                i = 0;
                j = 0;
                input_h = h * strides +0;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_0.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_0.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 1;
                input_h = h * strides +0;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_2.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_2.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 3;
                input_h = h * strides +0;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_3.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_3.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 4;
                input_h = h * strides +0;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_0 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_0.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_0.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 0;
                input_h = h * strides +1;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_4.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_4.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 1;
                input_h = h * strides +1;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_5.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_5.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_6.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_6.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 3;
                input_h = h * strides +1;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_7.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_7.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 4;
                input_h = h * strides +1;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_4 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_4.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_4.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_8.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_8.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_9.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_9.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_10.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_10.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 3;
                input_h = h * strides +2;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_11.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_11.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 4;
                input_h = h * strides +2;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_8 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_8.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_8.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 0;
                input_h = h * strides +3;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 1;
                input_h = h * strides +3;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_12 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 2;
                input_h = h * strides +3;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 3;
                input_h = h * strides +3;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 4;
                input_h = h * strides +3;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 0;
                input_h = h * strides +4;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 1;
                input_h = h * strides +4;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 2;
                input_h = h * strides +4;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 3;
                input_h = h * strides +4;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 4;
                input_h = h * strides +4;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
                w ++;
                i = 0;
                j = 0;
                input_h = h * strides +0;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 1;
                input_h = h * strides +0;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_2.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_2.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_3.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_3.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 3;
                input_h = h * strides +0;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_0.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_0.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 4;
                input_h = h * strides +0;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 0;
                input_h = h * strides +1;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_5.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_5.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 1;
                input_h = h * strides +1;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_6.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_6.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_7.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_7.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 3;
                input_h = h * strides +1;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_4.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_4.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 4;
                input_h = h * strides +1;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_5 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_5.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_5.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_9.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_9.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_10.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_10.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_11.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_11.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 3;
                input_h = h * strides +2;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_8.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_8.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 4;
                input_h = h * strides +2;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_9 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_9.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_9.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 0;
                input_h = h * strides +3;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 1;
                input_h = h * strides +3;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_12 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 2;
                input_h = h * strides +3;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 3;
                input_h = h * strides +3;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 4;
                input_h = h * strides +3;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 0;
                input_h = h * strides +4;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 1;
                input_h = h * strides +4;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 2;
                input_h = h * strides +4;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 3;
                input_h = h * strides +4;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 4;
                input_h = h * strides +4;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
                w ++;
                i = 0;
                j = 0;
                input_h = h * strides +0;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_2.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_2.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 1;
                input_h = h * strides +0;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_3.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_3.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_0.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_0.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 3;
                input_h = h * strides +0;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 4;
                input_h = h * strides +0;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_2 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_2.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_2.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 0;
                input_h = h * strides +1;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_6.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_6.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 1;
                input_h = h * strides +1;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_7.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_7.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_4.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_4.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 3;
                input_h = h * strides +1;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_5.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_5.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 4;
                input_h = h * strides +1;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_6 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_6.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_6.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_10.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_10.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_11.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_11.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_8.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_8.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 3;
                input_h = h * strides +2;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_9.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_9.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 4;
                input_h = h * strides +2;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_10 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_10.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_10.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 0;
                input_h = h * strides +3;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 1;
                input_h = h * strides +3;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_12 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 2;
                input_h = h * strides +3;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 3;
                input_h = h * strides +3;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 4;
                input_h = h * strides +3;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 0;
                input_h = h * strides +4;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 1;
                input_h = h * strides +4;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 2;
                input_h = h * strides +4;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 3;
                input_h = h * strides +4;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 4;
                input_h = h * strides +4;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
                w ++;
                i = 0;
                j = 0;
                input_h = h * strides +0;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_3.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_3.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 1;
                input_h = h * strides +0;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_0.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_0.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 3;
                input_h = h * strides +0;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_2.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_2.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 0;
                j = 4;
                input_h = h * strides +0;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_3 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_3.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_3.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 0;
                input_h = h * strides +1;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_7.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_7.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 1;
                input_h = h * strides +1;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_4.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_4.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_5.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_5.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 3;
                input_h = h * strides +1;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_6.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_6.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 1;
                j = 4;
                input_h = h * strides +1;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_7 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_7.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_7.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_11.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_11.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_8.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_8.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_9.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_9.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 3;
                input_h = h * strides +2;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_10.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_10.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 2;
                j = 4;
                input_h = h * strides +2;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_11 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_11.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_11.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 0;
                input_h = h * strides +3;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 1;
                input_h = h * strides +3;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    input_cache_12 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 2;
                input_h = h * strides +3;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 3;
                input_h = h * strides +3;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 3;
                j = 4;
                input_h = h * strides +3;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input_cache_12.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input_cache_12.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 0;
                input_h = h * strides +4;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 1;
                input_h = h * strides +4;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 2;
                input_h = h * strides +4;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 3;
                input_h = h * strides +4;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                i = 4;
                j = 4;
                input_h = h * strides +4;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64_x2((const int64_t *) &inputs[(input_h * width * depth /256 + input_w * depth /256) * depth /64]);
                    data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(data1.val[1],data2.val[1]);
                    output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                    output.val[1] = vaddq_u8(output.val[1],data1.val[1]);
                    
                }
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]) + vaddvq_u8(output.val[1]);
                
            }
        }
    }
    c_end = std::clock();
    time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    printf("%lf\n", time_elapsed_ms);
    
    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}