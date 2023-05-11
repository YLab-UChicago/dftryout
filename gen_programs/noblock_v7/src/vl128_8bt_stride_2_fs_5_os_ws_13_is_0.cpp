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
    depth = 128;
    num_filters = atoi(argv[3]);
    filter_height = 5;
    filter_width = 5;
    padding = 4;
    strides = 2;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    
    int64x2_t data1;
    int64x2_t data2;
    
    int64x2_t weight_cache_0;
    int64x2_t weight_cache_1;
    int64x2_t weight_cache_2;
    int64x2_t weight_cache_3;
    int64x2_t weight_cache_4;
    int64x2_t weight_cache_5;
    int64x2_t weight_cache_6;
    int64x2_t weight_cache_7;
    int64x2_t weight_cache_8;
    int64x2_t weight_cache_9;
    int64x2_t weight_cache_10;
    int64x2_t weight_cache_11;
    int64x2_t weight_cache_12;
    
    m5_reset_stats(0, 0);
    
    for (int f = 0; f < num_filters; f++) {
        weight_cache_0 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +0)*128/64]);
        weight_cache_1 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +1)*128/64]);
        weight_cache_2 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +2)*128/64]);
        weight_cache_3 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +3)*128/64]);
        weight_cache_4 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +4)*128/64]);
        weight_cache_5 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +5)*128/64]);
        weight_cache_6 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +6)*128/64]);
        weight_cache_7 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +7)*128/64]);
        weight_cache_8 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +8)*128/64]);
        weight_cache_9 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +9)*128/64]);
        weight_cache_10 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +10)*128/64]);
        weight_cache_11 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +11)*128/64]);
        weight_cache_12 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +12)*128/64]);
        int64x2_t output;
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
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_0);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 1;
                input_h = h * strides +0;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_1);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 3;
                input_h = h * strides +0;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_3);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 4;
                input_h = h * strides +0;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_4);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 0;
                input_h = h * strides +1;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_5);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 1;
                input_h = h * strides +1;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_6);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_7);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 3;
                input_h = h * strides +1;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_8);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 4;
                input_h = h * strides +1;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_9);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_10);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_11);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_12);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 3;
                input_h = h * strides +2;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 4;
                input_h = h * strides +2;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 0;
                input_h = h * strides +3;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 1;
                input_h = h * strides +3;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 2;
                input_h = h * strides +3;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 3;
                input_h = h * strides +3;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 4;
                input_h = h * strides +3;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 0;
                input_h = h * strides +4;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 1;
                input_h = h * strides +4;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 2;
                input_h = h * strides +4;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 3;
                input_h = h * strides +4;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 4;
                input_h = h * strides +4;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                
                w ++;
                i = 0;
                j = 0;
                input_h = h * strides +0;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_0);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 1;
                input_h = h * strides +0;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_1);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 3;
                input_h = h * strides +0;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_3);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 4;
                input_h = h * strides +0;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_4);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 0;
                input_h = h * strides +1;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_5);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 1;
                input_h = h * strides +1;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_6);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_7);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 3;
                input_h = h * strides +1;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_8);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 4;
                input_h = h * strides +1;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_9);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_10);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_11);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_12);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 3;
                input_h = h * strides +2;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 4;
                input_h = h * strides +2;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 0;
                input_h = h * strides +3;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 1;
                input_h = h * strides +3;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 2;
                input_h = h * strides +3;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 3;
                input_h = h * strides +3;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 4;
                input_h = h * strides +3;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 0;
                input_h = h * strides +4;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 1;
                input_h = h * strides +4;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 2;
                input_h = h * strides +4;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 3;
                input_h = h * strides +4;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 4;
                input_h = h * strides +4;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                
                w ++;
                i = 0;
                j = 0;
                input_h = h * strides +0;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_0);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 1;
                input_h = h * strides +0;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_1);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 2;
                input_h = h * strides +0;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 3;
                input_h = h * strides +0;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_3);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 0;
                j = 4;
                input_h = h * strides +0;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_4);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 0;
                input_h = h * strides +1;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_5);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 1;
                input_h = h * strides +1;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_6);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 2;
                input_h = h * strides +1;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_7);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 3;
                input_h = h * strides +1;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_8);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 1;
                j = 4;
                input_h = h * strides +1;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_9);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 0;
                input_h = h * strides +2;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_10);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 1;
                input_h = h * strides +2;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_11);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 2;
                input_h = h * strides +2;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data1 = vmulq_s8(data1,weight_cache_12);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 3;
                input_h = h * strides +2;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 2;
                j = 4;
                input_h = h * strides +2;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 0;
                input_h = h * strides +3;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 1;
                input_h = h * strides +3;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 2;
                input_h = h * strides +3;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 3;
                input_h = h * strides +3;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 3;
                j = 4;
                input_h = h * strides +3;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 0;
                input_h = h * strides +4;
                input_w = w * strides +0;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 1;
                input_h = h * strides +4;
                input_w = w * strides +1;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 2;
                input_h = h * strides +4;
                input_w = w * strides +2;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 3;
                input_h = h * strides +4;
                input_w = w * strides +3;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                i = 4;
                j = 4;
                input_h = h * strides +4;
                input_w = w * strides +4;
                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                    data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1 = vmulq_s8(data1,data2);
                    output = vaddq_u8(output,data1);
                    
                }
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output);
                
            }
        }
    }
    
    m5_dump_reset_stats(0, 0);
    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}