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
    
    int64x2_t data1;
    int64x2_t data2;
    int64x2_t input_cache_0;
    int64x2_t input_cache_1;
    
    int64x2_t weight_cache_0;
    int64x2_t weight_cache_1;
    int64x2_t weight_cache_2;
    int64x2_t weight_cache_3;
    int64x2_t weight_cache_4;
    int64x2_t weight_cache_5;
    
    m5_reset_stats(0, 0);
    
    for (int f = 0; f < num_filters; f++) {
        input_cache_0 = vld1q_s64((const int64_t *) &inputs[((0-padding) * width * depth /256 + (0-padding) * depth /256) * 128 /64]);
        input_cache_1 = vld1q_s64((const int64_t *) &inputs[((0-padding) * width * depth /256 + (1-padding) * depth /256) * 128 /64]);
        weight_cache_0 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +0)*128/64]);
        weight_cache_1 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +1)*128/64]);
        weight_cache_2 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +2)*128/64]);
        weight_cache_3 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +3)*128/64]);
        weight_cache_4 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +4)*128/64]);
        weight_cache_5 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +5)*128/64]);
        int64x2_t output;
        for (int h = 0; h < out_height; h++) {
            for (int w = 0; w < out_width; w ++) {
                int i = 0;
                int j = 0;
                int input_h;
                int input_w;
                 
                data1.val[0] = vmulq_s8(input_cache_0.val[0],weight_cache_0.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 0;
                j = 1;
                input_h = h * strides +0 - padding;
                input_w = w * strides +1 - padding;
                input_cache_0 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1.val[0] = vmulq_s8(input_cache_0.val[0],weight_cache_1.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 0;
                j = 2;
                input_h = h * strides +0 - padding;
                input_w = w * strides +2 - padding;
                data1.val[0] = vmulq_s8(input_cache_0.val[0],weight_cache_2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 0;
                j = 3;
                input_h = h * strides +0 - padding;
                input_w = w * strides +3 - padding;
                data1.val[0] = vmulq_s8(input_cache_0.val[0],weight_cache_3.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                data1.val[0] = vmulq_s8(input_cache_1.val[0],weight_cache_4.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 1;
                j = 1;
                input_h = h * strides +1 - padding;
                input_w = w * strides +1 - padding;
                input_cache_1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data1.val[0] = vmulq_s8(input_cache_1.val[0],weight_cache_5.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 1;
                j = 2;
                input_h = h * strides +1 - padding;
                input_w = w * strides +2 - padding;
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(input_cache_1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 1;
                j = 3;
                input_h = h * strides +1 - padding;
                input_w = w * strides +3 - padding;
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(input_cache_1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 2;
                j = 0;
                input_h = h * strides +2 - padding;
                input_w = w * strides +0 - padding;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 2;
                j = 1;
                input_h = h * strides +2 - padding;
                input_w = w * strides +1 - padding;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 2;
                j = 2;
                input_h = h * strides +2 - padding;
                input_w = w * strides +2 - padding;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 2;
                j = 3;
                input_h = h * strides +2 - padding;
                input_w = w * strides +3 - padding;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 3;
                j = 0;
                input_h = h * strides +3 - padding;
                input_w = w * strides +0 - padding;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 3;
                j = 1;
                input_h = h * strides +3 - padding;
                input_w = w * strides +1 - padding;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 3;
                j = 2;
                input_h = h * strides +3 - padding;
                input_w = w * strides +2 - padding;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                i = 3;
                j = 3;
                input_h = h * strides +3 - padding;
                input_w = w * strides +3 - padding;
                data1 = vld1q_s64((const int64_t *) &inputs[(input_h * width * depth /128 + input_w * depth /128) * depth /64]);
                data2 = vld1q_s64((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = vmulq_s8(data1.val[0],data2.val[0]);
                output.val[0] = vaddq_u8(output.val[0],data1.val[0]);
                
                outputs[h * out_width * num_filters + w * num_filters + f] = vaddvq_u8(output.val[0]);
                
            }
        }
    }
    
    m5_dump_reset_stats(0, 0);
    std::free(inputs);
    std::free(outputs);
    std::free(filters);
}