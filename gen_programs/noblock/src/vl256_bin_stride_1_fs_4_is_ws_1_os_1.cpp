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
    depth = atoi(argv[3]);
    num_filters = atoi(argv[4]);
    filter_height = 4;
    filter_width = 4;
    padding = 3;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    
    int64x2x2_t input;
    int64x2x2_t data1;
    int64x2x2_t data2;
    int64x2x2_t weight_cache_0;
    
    int64x2x2_t output_cache_0;
    
    m5_reset_stats(0, 0);
    
    for (int f = 0; f < num_filters; f++) {
        output_cache_0.val[0]=vdupq_n_u64(0);
        output_cache_0.val[1]=vdupq_n_u64(0);
        weight_cache_0 = vld1q_s64_x2((const int64_t*) &filters[(f * filter_height * filter_width +0)*256/64]);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w ++) {
                idx = h * width * depth / 64 + w * depth / 64;
                input = vld1q_s64_x2((const int64_t *)&inputs[idx]);
                 
                i = 3;
                j = 3;
                data1.val[0] = veorq_s64(input.val[0],weight_cache_0.val[0]);
                data1.val[1] = veorq_s64(input.val[1],weight_cache_0.val[1]);
                output_cache_0.val[0]= vaddq_u8(output_cache_0.val[0],data1.val[0]);
                output_cache_0.val[1]= vaddq_u8(output_cache_0.val[1],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(output_cache_0.val[0]))+vaddvq_u8(vcntq_u8(output_cache_0.val[1])));
                
                
                i = 3;
                j = 2;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                output_cache_0.val[0] = veorq_s64(input.val[0],data2.val[0]);
                output_cache_0.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(output_cache_0.val[0]))+vaddvq_u8(vcntq_u8(output_cache_0.val[1])));
                
                
                i = 3;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 3;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                
                w ++;
                i = 3;
                j = 3;
                data1.val[0] = veorq_s64(input.val[0],weight_cache_0.val[0]);
                data1.val[1] = veorq_s64(input.val[1],weight_cache_0.val[1]);
                output_cache_0.val[0]= vaddq_u8(output_cache_0.val[0],data1.val[0]);
                output_cache_0.val[1]= vaddq_u8(output_cache_0.val[1],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(output_cache_0.val[0]))+vaddvq_u8(vcntq_u8(output_cache_0.val[1])));
                
                
                i = 3;
                j = 2;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                output_cache_0.val[0] = veorq_s64(input.val[0],data2.val[0]);
                output_cache_0.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(output_cache_0.val[0]))+vaddvq_u8(vcntq_u8(output_cache_0.val[1])));
                
                
                i = 3;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 3;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                
                w ++;
                i = 3;
                j = 3;
                data1.val[0] = veorq_s64(input.val[0],weight_cache_0.val[0]);
                data1.val[1] = veorq_s64(input.val[1],weight_cache_0.val[1]);
                output_cache_0.val[0]= vaddq_u8(output_cache_0.val[0],data1.val[0]);
                output_cache_0.val[1]= vaddq_u8(output_cache_0.val[1],data1.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(output_cache_0.val[0]))+vaddvq_u8(vcntq_u8(output_cache_0.val[1])));
                
                
                i = 3;
                j = 2;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                output_cache_0.val[0] = veorq_s64(input.val[0],data2.val[0]);
                output_cache_0.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(output_cache_0.val[0]))+vaddvq_u8(vcntq_u8(output_cache_0.val[1])));
                
                
                i = 3;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 3;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 2;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 1;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                i = 0;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x2((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                outputs[h * out_width * num_filters + w * num_filters + f] += 256 - 2 * (vaddvq_u8(vcntq_u8(data1.val[0]))+vaddvq_u8(vcntq_u8(data1.val[1])));
                
                
                
            }
        }
    }
}