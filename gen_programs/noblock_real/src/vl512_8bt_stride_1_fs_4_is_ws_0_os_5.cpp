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
    int idx;
    int64_t* inputs;
    int64_t* outputs;
    int64_t* filters;
    int i = 0;
    int j = 0;
    int input_h;
    int input_w;
    int output_h;
    int output_w;
    int output_depth;
    std::clock_t c_start;
    std::clock_t c_end;
    double time_elapsed_ms;
    
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    depth = 512;
    num_filters = atoi(argv[3]);
    filter_height = 4;
    filter_width = 4;
    padding = 3;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    
    int64x2x4_t input;
    int64x2x4_t data1;
    int64x2x4_t data2;
    
    int64x2x4_t output_cache_0;
    int64x2x4_t output_cache_1;
    int64x2x4_t output_cache_2;
    int64x2x4_t output_cache_3;
    int64x2x4_t output_cache_4;
    
    
    c_start = std::clock();
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
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w ++) {
                idx = h * width * depth / 64 + w * depth / 64;
                input = vld1q_s64_x4((const int64_t *)&inputs[idx]);
                 
                i = 3;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_0.val[0]= vaddq_u8(output_cache_0.val[0],data1.val[0]);
                    output_cache_0.val[1]= vaddq_u8(output_cache_0.val[1],data1.val[1]);
                    output_cache_0.val[2]= vaddq_u8(output_cache_0.val[2],data1.val[2]);
                    output_cache_0.val[3]= vaddq_u8(output_cache_0.val[3],data1.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_0.val[0])+vaddvq_u8(output_cache_0.val[1])+vaddvq_u8(output_cache_0.val[2])+vaddvq_u8(output_cache_0.val[3]);
                    
                }
                
                i = 3;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_1.val[0]= vaddq_u8(output_cache_1.val[0],data1.val[0]);
                    output_cache_1.val[1]= vaddq_u8(output_cache_1.val[1],data1.val[1]);
                    output_cache_1.val[2]= vaddq_u8(output_cache_1.val[2],data1.val[2]);
                    output_cache_1.val[3]= vaddq_u8(output_cache_1.val[3],data1.val[3]);
                }
                
                i = 3;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_2.val[0]= vaddq_u8(output_cache_2.val[0],data1.val[0]);
                    output_cache_2.val[1]= vaddq_u8(output_cache_2.val[1],data1.val[1]);
                    output_cache_2.val[2]= vaddq_u8(output_cache_2.val[2],data1.val[2]);
                    output_cache_2.val[3]= vaddq_u8(output_cache_2.val[3],data1.val[3]);
                }
                
                i = 3;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    output_cache_0.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    output_cache_0.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    output_cache_0.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    output_cache_0.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                }
                
                i = 2;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_3.val[0]= vaddq_u8(output_cache_3.val[0],data1.val[0]);
                    output_cache_3.val[1]= vaddq_u8(output_cache_3.val[1],data1.val[1]);
                    output_cache_3.val[2]= vaddq_u8(output_cache_3.val[2],data1.val[2]);
                    output_cache_3.val[3]= vaddq_u8(output_cache_3.val[3],data1.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_3.val[0])+vaddvq_u8(output_cache_3.val[1])+vaddvq_u8(output_cache_3.val[2])+vaddvq_u8(output_cache_3.val[3]);
                    
                }
                
                i = 2;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_4.val[0]= vaddq_u8(output_cache_4.val[0],data1.val[0]);
                    output_cache_4.val[1]= vaddq_u8(output_cache_4.val[1],data1.val[1]);
                    output_cache_4.val[2]= vaddq_u8(output_cache_4.val[2],data1.val[2]);
                    output_cache_4.val[3]= vaddq_u8(output_cache_4.val[3],data1.val[3]);
                }
                
                i = 2;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    output_cache_3.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    output_cache_3.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    output_cache_3.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    output_cache_3.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                }
                
                i = 2;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                
                w ++;
                i = 3;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_1.val[0]= vaddq_u8(output_cache_1.val[0],data1.val[0]);
                    output_cache_1.val[1]= vaddq_u8(output_cache_1.val[1],data1.val[1]);
                    output_cache_1.val[2]= vaddq_u8(output_cache_1.val[2],data1.val[2]);
                    output_cache_1.val[3]= vaddq_u8(output_cache_1.val[3],data1.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_1.val[0])+vaddvq_u8(output_cache_1.val[1])+vaddvq_u8(output_cache_1.val[2])+vaddvq_u8(output_cache_1.val[3]);
                    
                }
                
                i = 3;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_2.val[0]= vaddq_u8(output_cache_2.val[0],data1.val[0]);
                    output_cache_2.val[1]= vaddq_u8(output_cache_2.val[1],data1.val[1]);
                    output_cache_2.val[2]= vaddq_u8(output_cache_2.val[2],data1.val[2]);
                    output_cache_2.val[3]= vaddq_u8(output_cache_2.val[3],data1.val[3]);
                }
                
                i = 3;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_0.val[0]= vaddq_u8(output_cache_0.val[0],data1.val[0]);
                    output_cache_0.val[1]= vaddq_u8(output_cache_0.val[1],data1.val[1]);
                    output_cache_0.val[2]= vaddq_u8(output_cache_0.val[2],data1.val[2]);
                    output_cache_0.val[3]= vaddq_u8(output_cache_0.val[3],data1.val[3]);
                }
                
                i = 3;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    output_cache_1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    output_cache_1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    output_cache_1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    output_cache_1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                }
                
                i = 2;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_4.val[0]= vaddq_u8(output_cache_4.val[0],data1.val[0]);
                    output_cache_4.val[1]= vaddq_u8(output_cache_4.val[1],data1.val[1]);
                    output_cache_4.val[2]= vaddq_u8(output_cache_4.val[2],data1.val[2]);
                    output_cache_4.val[3]= vaddq_u8(output_cache_4.val[3],data1.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_4.val[0])+vaddvq_u8(output_cache_4.val[1])+vaddvq_u8(output_cache_4.val[2])+vaddvq_u8(output_cache_4.val[3]);
                    
                }
                
                i = 2;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_3.val[0]= vaddq_u8(output_cache_3.val[0],data1.val[0]);
                    output_cache_3.val[1]= vaddq_u8(output_cache_3.val[1],data1.val[1]);
                    output_cache_3.val[2]= vaddq_u8(output_cache_3.val[2],data1.val[2]);
                    output_cache_3.val[3]= vaddq_u8(output_cache_3.val[3],data1.val[3]);
                }
                
                i = 2;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    output_cache_4.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    output_cache_4.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    output_cache_4.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    output_cache_4.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                }
                
                i = 2;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                
                w ++;
                i = 3;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_2.val[0]= vaddq_u8(output_cache_2.val[0],data1.val[0]);
                    output_cache_2.val[1]= vaddq_u8(output_cache_2.val[1],data1.val[1]);
                    output_cache_2.val[2]= vaddq_u8(output_cache_2.val[2],data1.val[2]);
                    output_cache_2.val[3]= vaddq_u8(output_cache_2.val[3],data1.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_2.val[0])+vaddvq_u8(output_cache_2.val[1])+vaddvq_u8(output_cache_2.val[2])+vaddvq_u8(output_cache_2.val[3]);
                    
                }
                
                i = 3;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_0.val[0]= vaddq_u8(output_cache_0.val[0],data1.val[0]);
                    output_cache_0.val[1]= vaddq_u8(output_cache_0.val[1],data1.val[1]);
                    output_cache_0.val[2]= vaddq_u8(output_cache_0.val[2],data1.val[2]);
                    output_cache_0.val[3]= vaddq_u8(output_cache_0.val[3],data1.val[3]);
                }
                
                i = 3;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_1.val[0]= vaddq_u8(output_cache_1.val[0],data1.val[0]);
                    output_cache_1.val[1]= vaddq_u8(output_cache_1.val[1],data1.val[1]);
                    output_cache_1.val[2]= vaddq_u8(output_cache_1.val[2],data1.val[2]);
                    output_cache_1.val[3]= vaddq_u8(output_cache_1.val[3],data1.val[3]);
                }
                
                i = 3;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    output_cache_2.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    output_cache_2.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    output_cache_2.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    output_cache_2.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                }
                
                i = 2;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_3.val[0]= vaddq_u8(output_cache_3.val[0],data1.val[0]);
                    output_cache_3.val[1]= vaddq_u8(output_cache_3.val[1],data1.val[1]);
                    output_cache_3.val[2]= vaddq_u8(output_cache_3.val[2],data1.val[2]);
                    output_cache_3.val[3]= vaddq_u8(output_cache_3.val[3],data1.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_3.val[0])+vaddvq_u8(output_cache_3.val[1])+vaddvq_u8(output_cache_3.val[2])+vaddvq_u8(output_cache_3.val[3]);
                    
                }
                
                i = 2;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    output_cache_4.val[0]= vaddq_u8(output_cache_4.val[0],data1.val[0]);
                    output_cache_4.val[1]= vaddq_u8(output_cache_4.val[1],data1.val[1]);
                    output_cache_4.val[2]= vaddq_u8(output_cache_4.val[2],data1.val[2]);
                    output_cache_4.val[3]= vaddq_u8(output_cache_4.val[3],data1.val[3]);
                }
                
                i = 2;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    output_cache_4.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    output_cache_4.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    output_cache_4.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    output_cache_4.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                }
                
                i = 2;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 1;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                i = 0;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                    data1.val[0] = vmulq_s8(input.val[0],data2.val[0]);
                    data1.val[1] = vmulq_s8(input.val[1],data2.val[1]);
                    data1.val[2] = vmulq_s8(input.val[2],data2.val[2]);
                    data1.val[3] = vmulq_s8(input.val[3],data2.val[3]);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1.val[0])+vaddvq_u8(data1.val[1])+vaddvq_u8(data1.val[2])+vaddvq_u8(data1.val[3]);
                    
                }
                
                
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