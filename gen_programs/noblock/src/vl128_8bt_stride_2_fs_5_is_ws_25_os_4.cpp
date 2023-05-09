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
    
    int64x2_t input;
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
    int64x2_t weight_cache_13;
    int64x2_t weight_cache_14;
    int64x2_t weight_cache_15;
    int64x2_t weight_cache_16;
    int64x2_t weight_cache_17;
    int64x2_t weight_cache_18;
    int64x2_t weight_cache_19;
    int64x2_t weight_cache_20;
    int64x2_t weight_cache_21;
    int64x2_t weight_cache_22;
    int64x2_t weight_cache_23;
    int64x2_t weight_cache_24;
    
    int64x2_t output_cache_0;
    int64x2_t output_cache_1;
    int64x2_t output_cache_2;
    int64x2_t output_cache_3;
    
    
    for (int f = 0; f < num_filters; f++) {
        output_cache_0=vdupq_n_u64(0);
        output_cache_1=vdupq_n_u64(0);
        output_cache_2=vdupq_n_u64(0);
        output_cache_3=vdupq_n_u64(0);
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
        weight_cache_13 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +13)*128/64]);
        weight_cache_14 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +14)*128/64]);
        weight_cache_15 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +15)*128/64]);
        weight_cache_16 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +16)*128/64]);
        weight_cache_17 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +17)*128/64]);
        weight_cache_18 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +18)*128/64]);
        weight_cache_19 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +19)*128/64]);
        weight_cache_20 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +20)*128/64]);
        weight_cache_21 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +21)*128/64]);
        weight_cache_22 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +22)*128/64]);
        weight_cache_23 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +23)*128/64]);
        weight_cache_24 = vld1q_s64((const int64_t*) &filters[(f * filter_height * filter_width +24)*128/64]);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w ++) {
                idx = h * width * depth / 64 + w * depth / 64;
                input = vld1q_s64((const int64_t *)&inputs[idx]);
                 
                i = 4;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_0);
                    output_cache_0 = vaddq_u8(output_cache_0,data1);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_0);
                    
                }
                
                i = 4;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_1);
                    output_cache_1 = vaddq_u8(output_cache_1,data1);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_1);
                    
                }
                
                i = 4;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_2);
                    output_cache_2 = vaddq_u8(output_cache_2,data1);
                }
                
                i = 4;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_cache_2 = vmulq_s8(input,weight_cache_3);
                }
                
                i = 4;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_cache_0 = vmulq_s8(input,weight_cache_4);
                }
                
                i = 3;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_5);
                    output_cache_3 = vaddq_u8(output_cache_3,data1);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_3);
                    
                }
                
                i = 3;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_6);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 3;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_cache_3 = vmulq_s8(input,weight_cache_7);
                }
                
                i = 3;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_8);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 3;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_9);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_10);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_11);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_12);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_13);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_14);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_15);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_16);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_17);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_18);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_19);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_20);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_21);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_22);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_23);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_24);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                
                w ++;
                i = 4;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_0);
                    output_cache_1 = vaddq_u8(output_cache_1,data1);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_1);
                    
                }
                
                i = 4;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_1);
                    output_cache_2 = vaddq_u8(output_cache_2,data1);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_2);
                    
                }
                
                i = 4;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_2);
                    output_cache_0 = vaddq_u8(output_cache_0,data1);
                }
                
                i = 4;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_cache_0 = vmulq_s8(input,weight_cache_3);
                }
                
                i = 4;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_cache_1 = vmulq_s8(input,weight_cache_4);
                }
                
                i = 3;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_5);
                    output_cache_3 = vaddq_u8(output_cache_3,data1);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_3);
                    
                }
                
                i = 3;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_6);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 3;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_cache_3 = vmulq_s8(input,weight_cache_7);
                }
                
                i = 3;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_8);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 3;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_9);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_10);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_11);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_12);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_13);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_14);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_15);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_16);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_17);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_18);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_19);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_20);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_21);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_22);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_23);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_24);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                
                w ++;
                i = 4;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_0);
                    output_cache_2 = vaddq_u8(output_cache_2,data1);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_2);
                    
                }
                
                i = 4;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_1);
                    output_cache_0 = vaddq_u8(output_cache_0,data1);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_0);
                    
                }
                
                i = 4;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_2);
                    output_cache_1 = vaddq_u8(output_cache_1,data1);
                }
                
                i = 4;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_cache_1 = vmulq_s8(input,weight_cache_3);
                }
                
                i = 4;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_cache_2 = vmulq_s8(input,weight_cache_4);
                }
                
                i = 3;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    data1 = vmulq_s8(input,weight_cache_5);
                    output_cache_3 = vaddq_u8(output_cache_3,data1);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(output_cache_3);
                    
                }
                
                i = 3;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_6);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 3;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_cache_3 = vmulq_s8(input,weight_cache_7);
                }
                
                i = 3;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_8);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 3;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_9);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_10);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_11);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_12);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_13);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 2;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_14);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_15);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_16);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_17);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_18);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 1;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_19);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 4;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_20);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 3;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_21);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 2;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_22);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 1;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_23);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                i = 0;
                j = 0;
                output_h = floor((h - i) / strides);
                output_w = floor((w - j) / strides);
                if (output_h >= 0 && output_h < out_height && output_w >= 0 && output_w < out_width) {
                    output_h = (h + padding - i) / strides;
                    output_w = (w + padding - j) / strides;
                    data1 = vmulq_s8(input,weight_cache_24);
                    outputs[output_h * out_width * num_filters + output_w * num_filters + f] += vaddvq_u8(data1);
                    
                }
                
                
            }
        }
    }
std::free(inputs);
std::free(outputs);
std::free(filters);
}