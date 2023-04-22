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
    int output_depth;
    
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    depth = atoi(argv[3]);
    num_filters = atoi(argv[4]);
    filter_height = 5;
    filter_width = 5;
    padding = 4;
    strides = 1;
    int out_height = ceil((height - filter_height + 2 * padding) / strides + 1);
    int out_width = ceil((width - filter_width + 2 * padding) / strides + 1);
    inputs = (int64_t *)malloc(sizeof(int64_t) * (height + 2 * padding) * (width + 2 * padding) * depth / 64);
    outputs = (int64_t *)malloc(sizeof(int64_t) * out_height * out_width * num_filters);
    filters = (int64_t *)malloc(sizeof(int64_t) * filter_height * filter_width * num_filters * depth / 64);
    
    int64x2x4_t input;
    int64x2x4_t data1;
    int64x2x4_t data2;
    
    
    m5_reset_stats(0, 0);
    
    for (int f = 0; f < num_filters; f++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w ++) {
                idx = h * width * depth / 64 + w * depth / 64;
                input = vld1q_s64_x4((const int64_t *)&inputs[idx]);
                 
                i = 4;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                
                w ++;
                i = 4;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                
                w ++;
                i = 4;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                
                w ++;
                i = 4;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 4;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 3;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 2;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 1;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 4;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 3;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 2;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 1;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                i = 0;
                j = 0;
                output_h = (h + padding - i) / strides;
                output_w = (w + padding - j) / strides;
                data2 = vld1q_s64_x4((const int64_t *) & filters[(f * filter_height * filter_width + i * filter_width + j)*depth/64]);
                data1.val[0] = veorq_s64(input.val[0],data2.val[0]);
                data1.val[1] = veorq_s64(input.val[1],data2.val[1]);
                data1.val[2] = veorq_s64(input.val[2],data2.val[2]);
                data1.val[3] = veorq_s64(input.val[3],data2.val[3]);
                
                
            }
        }
    }
}